import ArgumentParser
import Foundation
import Hummingbird
#if canImport(FoundationModels)
import FoundationModels
#endif

struct BridgeMessagePayload: Codable, Sendable {
    let role: String
    let content: String
    let metadata: [String: JSONValue]?
}

enum JSONValue: Codable, Sendable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .integer(value): try container.encode(value)
        case let .bool(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}

struct ToolDefinitionPayload: Codable, Sendable {
    let name: String
    let description: String?
    let input_schema: JSONValue?
}

struct SessionCreateRequest: Codable, Sendable {
    let session_id: String?
    let instructions: String?
    let tools: [ToolDefinitionPayload]?
}

struct SessionCreateResponse: Codable, Sendable {
    let session_id: String
}

struct SessionResponseRequest: Codable, Sendable {
    let prompt: String?
    let messages: [BridgeMessagePayload]
    let stream: Bool?
}

struct SessionResponseBody: Codable, Sendable {
    let text: String
}

struct ToolResultSubmitRequest: Codable, Sendable {
    let result: JSONValue
}

struct HealthResponse: Codable, Sendable {
    let status: String
    let availability: String
    let detail: String?
}

struct SSEEvent: Codable, Sendable {
    let type: String
    let text: String?
    let name: String?
    let callID: String?
    let args: JSONValue?
    let result: JSONValue?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case name
        case callID = "call_id"
        case args
        case result
        case message
    }
}

enum BridgeError: Error, LocalizedError {
    case unsupportedPlatform
    case modelUnavailable(String)
    case missingSession(String)
    case invalidRequest(String)
    case sessionBusy(String)
    case cancelled(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "Foundation Models requires macOS 26 or newer."
        case let .modelUnavailable(detail):
            return detail
        case let .missingSession(sessionID):
            return "Session not found: \(sessionID)"
        case let .invalidRequest(message):
            return message
        case let .sessionBusy(sessionID):
            return "Session already has an active response: \(sessionID)"
        case let .cancelled(sessionID):
            return "Response cancelled for session: \(sessionID)"
        }
    }
}

extension JSONValue {
    var stringValue: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }

    var objectValue: [String: JSONValue]? {
        guard case let .object(value) = self else { return nil }
        return value
    }

    var stringArrayValue: [String]? {
        guard case let .array(values) = self else { return nil }
        return values.compactMap { $0.stringValue }
    }

    var serializedString: String {
        guard let data = try? JSONEncoder().encode(self),
              let json = String(data: data, encoding: .utf8)
        else {
            return "null"
        }
        return json
    }

    var promptText: String {
        if case let .string(value) = self {
            return value
        }
        return self.serializedString
    }

    static func fromGeneratedContent(_ content: GeneratedContent) throws -> JSONValue {
        let data = Data(content.jsonString.utf8)
        let object = try JSONSerialization.jsonObject(with: data)
        return try fromJSONObject(object)
    }

    static func fromJSONObject(_ object: Any) throws -> JSONValue {
        switch object {
        case let value as String:
            return .string(value)
        case let value as Bool:
            return .bool(value)
        case let value as Int:
            return .integer(value)
        case let value as Double:
            return .number(value)
        case let value as NSNumber:
            if CFGetTypeID(value) == CFBooleanGetTypeID() {
                return .bool(value.boolValue)
            }
            let doubleValue = value.doubleValue
            if floor(doubleValue) == doubleValue {
                return .integer(value.intValue)
            }
            return .number(doubleValue)
        case let value as [Any]:
            return .array(try value.map { try fromJSONObject($0) })
        case let value as [String: Any]:
            return .object(try value.mapValues { try fromJSONObject($0) })
        case _ as NSNull:
            return .null
        default:
            throw BridgeError.invalidRequest("unsupported JSON value in tool payload")
        }
    }
}

#if canImport(FoundationModels)
@available(macOS 26.0, *)
actor ToolInvocationCoordinator {
    private let sessionID: String
    private var streamContinuation: AsyncStream<ByteBuffer>.Continuation?
    private var pendingCalls: [String: CheckedContinuation<JSONValue, Error>] = [:]

    init(sessionID: String) {
        self.sessionID = sessionID
    }

    func beginStreaming(_ continuation: AsyncStream<ByteBuffer>.Continuation) {
        self.streamContinuation = continuation
    }

    func finishStreaming() {
        self.streamContinuation = nil
    }

    func requestToolCall(name: String, arguments: JSONValue) async throws -> String {
        guard let continuation = self.streamContinuation else {
            throw BridgeError.invalidRequest("tool call requested without an active stream")
        }

        let callID = UUID().uuidString
        continuation.yield(
            encodeSSEBuffer(
                SSEEvent(
                    type: "tool_call",
                    text: nil,
                    name: name,
                    callID: callID,
                    args: arguments,
                    result: nil,
                    message: nil
                )
            )
        )

        let result = try await withCheckedThrowingContinuation { (toolContinuation: CheckedContinuation<JSONValue, Error>) in
            self.pendingCalls[callID] = toolContinuation
        }

        continuation.yield(
            encodeSSEBuffer(
                SSEEvent(
                    type: "tool_result",
                    text: nil,
                    name: name,
                    callID: callID,
                    args: nil,
                    result: result,
                    message: nil
                )
            )
        )
        return result.promptText
    }

    func submitToolResult(callID: String, result: JSONValue) -> Bool {
        guard let continuation = self.pendingCalls.removeValue(forKey: callID) else {
            return false
        }
        continuation.resume(returning: result)
        return true
    }

    func cancelPendingToolCalls() {
        let pending = Array(self.pendingCalls.values)
        self.pendingCalls.removeAll()
        for continuation in pending {
            continuation.resume(throwing: BridgeError.cancelled(sessionID))
        }
    }
}

@available(macOS 26.0, *)
struct RemoteTool: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = String

    let name: String
    let description: String
    let parameters: GenerationSchema
    let coordinator: ToolInvocationCoordinator

    func call(arguments: GeneratedContent) async throws -> String {
        let payload = try JSONValue.fromGeneratedContent(arguments)
        return try await coordinator.requestToolCall(name: name, arguments: payload)
    }
}

@available(macOS 26.0, *)
private func buildRemoteTools(
    from definitions: [ToolDefinitionPayload],
    coordinator: ToolInvocationCoordinator
) throws -> [any Tool] {
    try definitions.map { definition in
        let schema = try buildToolParametersSchema(definition.input_schema, toolName: definition.name)
        return RemoteTool(
            name: definition.name,
            description: definition.description ?? definition.name,
            parameters: schema,
            coordinator: coordinator
        ) as any Tool
    }
}

@available(macOS 26.0, *)
private func buildToolParametersSchema(_ inputSchema: JSONValue?, toolName: String) throws -> GenerationSchema {
    let payload = inputSchema ?? .object([
        "type": .string("object"),
        "properties": .object([:]),
    ])
    let root = try buildDynamicSchema(payload, name: "\(toolName)_arguments")
    return try GenerationSchema(root: root, dependencies: [])
}

@available(macOS 26.0, *)
private func buildDynamicSchema(_ schema: JSONValue, name: String) throws -> DynamicGenerationSchema {
    guard let object = schema.objectValue else {
        throw BridgeError.invalidRequest("tool schema for \(name) must be an object")
    }

    let type = object["type"]?.stringValue ?? (object["properties"] != nil ? "object" : "string")
    switch type {
    case "object":
        let properties = object["properties"]?.objectValue ?? [:]
        let required = Set(object["required"]?.stringArrayValue ?? [])
        let dynamicProperties = try properties.keys.sorted().map { propertyName in
            let propertySchema = properties[propertyName] ?? .object(["type": .string("string")])
            return DynamicGenerationSchema.Property(
                name: propertyName,
                description: propertySchema.objectValue?["description"]?.stringValue,
                schema: try buildDynamicSchema(propertySchema, name: "\(name)_\(propertyName)"),
                isOptional: !required.contains(propertyName)
            )
        }
        return DynamicGenerationSchema(
            name: name,
            description: object["description"]?.stringValue,
            properties: dynamicProperties
        )
    case "string":
        return DynamicGenerationSchema(type: String.self, guides: [GenerationGuide<String>]())
    case "integer":
        return DynamicGenerationSchema(type: Int.self, guides: [GenerationGuide<Int>]())
    case "number":
        return DynamicGenerationSchema(type: Double.self, guides: [GenerationGuide<Double>]())
    case "boolean":
        return DynamicGenerationSchema(type: Bool.self, guides: [GenerationGuide<Bool>]())
    case "array":
        let itemSchema = object["items"] ?? .object(["type": .string("string")])
        return DynamicGenerationSchema(
            arrayOf: try buildDynamicSchema(itemSchema, name: "\(name)_item"),
            minimumElements: nil,
            maximumElements: nil
        )
    default:
        throw BridgeError.invalidRequest("unsupported tool schema type: \(type)")
    }
}
#endif

@available(macOS 26.0, *)
actor BridgeSessionActor {
    let sessionID: String
    let tools: [ToolDefinitionPayload]

    private let session: LanguageModelSession
    #if canImport(FoundationModels)
    private let toolCoordinator: ToolInvocationCoordinator
    #endif
    private var activeResponseID: UUID?
    private var activeResponseTask: Task<Void, Never>?

    init(sessionID: String, instructions: String?, tools: [ToolDefinitionPayload]) throws {
        self.sessionID = sessionID
        self.tools = tools

        #if canImport(FoundationModels)
        let coordinator = ToolInvocationCoordinator(sessionID: sessionID)
        self.toolCoordinator = coordinator
        let remoteTools = try buildRemoteTools(from: tools, coordinator: coordinator)
        if let instructions, !instructions.isEmpty {
            self.session = LanguageModelSession(tools: remoteTools) {
                instructions
            }
        } else {
            self.session = LanguageModelSession(tools: remoteTools) {
                ""
            }
        }
        #else
        if let instructions, !instructions.isEmpty {
            self.session = LanguageModelSession(instructions: instructions)
        } else {
            self.session = LanguageModelSession()
        }
        #endif
    }

    func respond(prompt: String) async throws -> String {
        guard activeResponseTask == nil else {
            throw BridgeError.sessionBusy(sessionID)
        }
        let response = try await session.respond(to: prompt)
        return response.content
    }

    func stream(prompt: String) async throws -> AsyncStream<ByteBuffer> {
        guard activeResponseTask == nil else {
            throw BridgeError.sessionBusy(sessionID)
        }

        let responseID = UUID()
        let (stream, continuation) = AsyncStream<ByteBuffer>.makeStream()
        #if canImport(FoundationModels)
        await toolCoordinator.beginStreaming(continuation)
        #endif
        let task = Task { [self] in
            await self.runStreaming(prompt: prompt, responseID: responseID, continuation: continuation)
        }

        self.activeResponseID = responseID
        self.activeResponseTask = task

        continuation.onTermination = { @Sendable _ in
            task.cancel()
            Task {
                await self.finishResponse(responseID)
            }
        }

        return stream
    }

    func cancelActiveResponse() async -> Bool {
        let hadActiveResponse = activeResponseTask != nil
        activeResponseTask?.cancel()
        activeResponseTask = nil
        activeResponseID = nil
        #if canImport(FoundationModels)
        await toolCoordinator.cancelPendingToolCalls()
        #endif
        return hadActiveResponse
    }

    func close() async {
        _ = await cancelActiveResponse()
    }

    func submitToolResult(callID: String, result: JSONValue) async -> Bool {
        #if canImport(FoundationModels)
        return await toolCoordinator.submitToolResult(callID: callID, result: result)
        #else
        _ = callID
        _ = result
        return false
        #endif
    }

    private func runStreaming(
        prompt: String,
        responseID: UUID,
        continuation: AsyncStream<ByteBuffer>.Continuation
    ) async {
        defer {
            continuation.finish()
            Task {
                await self.finishResponse(responseID)
            }
        }

        do {
            var previousContent = ""
            let responseStream = session.streamResponse(to: prompt)
            for try await snapshot in responseStream {
                try Task.checkCancellation()
                let currentContent = snapshot.content
                guard currentContent.count > previousContent.count else { continue }
                let delta = String(currentContent.dropFirst(previousContent.count))
                previousContent = currentContent
                guard !delta.isEmpty else { continue }
                continuation.yield(
                    encodeSSEBuffer(
                        SSEEvent(
                            type: "response_chunk",
                            text: delta,
                            name: nil,
                            callID: nil,
                            args: nil,
                            result: nil,
                            message: nil
                        )
                    )
                )
            }

            continuation.yield(
                encodeSSEBuffer(
                    SSEEvent(
                        type: "response_end",
                        text: nil,
                        name: nil,
                        callID: nil,
                        args: nil,
                        result: nil,
                        message: nil
                    )
                )
            )
        } catch is CancellationError {
            continuation.yield(
                encodeSSEBuffer(
                    SSEEvent(
                        type: "error",
                        text: nil,
                        name: nil,
                        callID: nil,
                        args: nil,
                        result: nil,
                        message: BridgeError.cancelled(sessionID).localizedDescription
                    )
                )
            )
        } catch {
            continuation.yield(
                encodeSSEBuffer(
                    SSEEvent(
                        type: "error",
                        text: nil,
                        name: nil,
                        callID: nil,
                        args: nil,
                        result: nil,
                        message: error.localizedDescription
                    )
                )
            )
        }
    }

    private func finishResponse(_ responseID: UUID) async {
        #if canImport(FoundationModels)
        await toolCoordinator.finishStreaming()
        await toolCoordinator.cancelPendingToolCalls()
        #endif
        clearActiveResponse(responseID)
    }

    private func clearActiveResponse(_ responseID: UUID) {
        guard activeResponseID == responseID else { return }
        activeResponseTask = nil
        activeResponseID = nil
    }
}

actor BridgeSessionStore {
    private var sessions: [String: BridgeSessionActor] = [:]

    func create(_ payload: SessionCreateRequest) throws -> String {
        let sessionID = payload.session_id ?? UUID().uuidString
        let session = try BridgeSessionActor(
            sessionID: sessionID,
            instructions: payload.instructions,
            tools: payload.tools ?? []
        )
        sessions[sessionID] = session
        return sessionID
    }

    func get(_ sessionID: String) -> BridgeSessionActor? {
        sessions[sessionID]
    }

    func delete(_ sessionID: String) async {
        guard let session = sessions.removeValue(forKey: sessionID) else { return }
        await session.close()
    }
}

struct FoundationModelsHTTPServer: Sendable {
    let sessionStore = BridgeSessionStore()

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        router.get("/health") { _, _ in
            try encodeJSONResponse(self.healthPayload())
        }

        router.post("/sessions") { request, _ in
            let payload: SessionCreateRequest = try await decodeBody(request, as: SessionCreateRequest.self)
            let sessionID = try await self.sessionStore.create(payload)
            return try encodeJSONResponse(SessionCreateResponse(session_id: sessionID))
        }

        router.delete("/sessions/:sessionID") { request, _ in
            let sessionID = try self.extractSessionID(from: request.uri.path, suffix: "")
            await self.sessionStore.delete(sessionID)
            return try encodeJSONResponse(["deleted": true])
        }

        router.post("/sessions/:sessionID/cancel") { request, _ in
            let sessionID = try self.extractSessionID(from: request.uri.path, suffix: "/cancel")
            guard let session = await self.sessionStore.get(sessionID) else {
                return try errorResponse(status: .notFound, message: BridgeError.missingSession(sessionID).localizedDescription)
            }
            let cancelled = await session.cancelActiveResponse()
            return try encodeJSONResponse(["cancelled": cancelled])
        }

        router.post("/sessions/:sessionID/tool-results/:callID") { request, _ in
            let (sessionID, callID) = try self.extractSessionIDAndCallID(from: request.uri.path)
            guard let session = await self.sessionStore.get(sessionID) else {
                return try errorResponse(status: .notFound, message: BridgeError.missingSession(sessionID).localizedDescription)
            }
            let payload: ToolResultSubmitRequest = try await decodeBody(request, as: ToolResultSubmitRequest.self)
            let accepted = await session.submitToolResult(callID: callID, result: payload.result)
            if !accepted {
                return try errorResponse(status: .notFound, message: "Tool call not found: \(callID)")
            }
            return try encodeJSONResponse(["accepted": true])
        }

        router.post("/sessions/:sessionID/responses") { request, _ in
            let sessionID = try self.extractSessionID(from: request.uri.path, suffix: "/responses")
            guard let session = await self.sessionStore.get(sessionID) else {
                return try errorResponse(status: .notFound, message: BridgeError.missingSession(sessionID).localizedDescription)
            }

            let payload: SessionResponseRequest = try await decodeBody(request, as: SessionResponseRequest.self)
            let isStreaming = payload.stream ?? true
            if isStreaming {
                return try await self.handleStreamingRequest(session: session, payload: payload)
            }
            return try await self.handleBufferedRequest(session: session, payload: payload)
        }

        return router
    }

    private func handleBufferedRequest(
        session: BridgeSessionActor,
        payload: SessionResponseRequest
    ) async throws -> Response {
        let prompt = try self.promptFromPayload(payload)
        let fullText = try await session.respond(prompt: prompt)
        return try encodeJSONResponse(SessionResponseBody(text: fullText))
    }

    private func handleStreamingRequest(
        session: BridgeSessionActor,
        payload: SessionResponseRequest
    ) async throws -> Response {
        let prompt = try self.promptFromPayload(payload)
        let stream = try await session.stream(prompt: prompt)

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive",
            ],
            body: .init(asyncSequence: stream)
        )
    }

    private func promptFromPayload(_ payload: SessionResponseRequest) throws -> String {
        try self.ensureFoundationModelsAvailable()
        let prompt = payload.prompt ?? payload.messages.last(where: { $0.role == "user" })?.content ?? ""
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BridgeError.invalidRequest("response prompt must not be empty")
        }
        return prompt
    }

    private func ensureFoundationModelsAvailable() throws {
        guard Self.foundationModelsAvailable else {
            throw BridgeError.unsupportedPlatform
        }

        #if canImport(FoundationModels)
        let availability = SystemLanguageModel.default.availability
        guard case .available = availability else {
            throw BridgeError.modelUnavailable(self.availabilityDescription(availability))
        }
        #endif
    }

    private func healthPayload() -> HealthResponse {
        guard Self.foundationModelsAvailable else {
            return HealthResponse(status: "unavailable", availability: "unsupported", detail: BridgeError.unsupportedPlatform.localizedDescription)
        }

        #if canImport(FoundationModels)
        let availability = SystemLanguageModel.default.availability
        switch availability {
        case .available:
            return HealthResponse(status: "ok", availability: "available", detail: nil)
        default:
            return HealthResponse(status: "unavailable", availability: self.availabilityLabel(availability), detail: self.availabilityDescription(availability))
        }
        #else
        return HealthResponse(status: "unavailable", availability: "unsupported", detail: BridgeError.unsupportedPlatform.localizedDescription)
        #endif
    }

    private func availabilityLabel(_ availability: SystemLanguageModel.Availability) -> String {
        switch availability {
        case .available:
            return "available"
        case .unavailable(.appleIntelligenceNotEnabled):
            return "apple_intelligence_not_enabled"
        case .unavailable(.modelNotReady):
            return "model_not_ready"
        case .unavailable(.deviceNotEligible):
            return "device_not_eligible"
        case .unavailable:
            return "unavailable"
        @unknown default:
            return "unknown"
        }
    }

    private func availabilityDescription(_ availability: SystemLanguageModel.Availability) -> String {
        switch availability {
        case .available:
            return "Foundation Models is available."
        case .unavailable(.appleIntelligenceNotEnabled):
            return "Apple Intelligence must be enabled in System Settings."
        case .unavailable(.modelNotReady):
            return "The on-device Foundation model is not ready yet."
        case .unavailable(.deviceNotEligible):
            return "This device does not support Apple Intelligence."
        case .unavailable:
            return "Foundation Models is unavailable."
        @unknown default:
            return "Foundation Models availability is unknown."
        }
    }

    private func extractSessionID(from path: String, suffix: String) throws -> String {
        let prefix = "/sessions/"
        guard path.hasPrefix(prefix) else {
            throw BridgeError.invalidRequest("invalid session route: \(path)")
        }
        let trimmed = String(path.dropFirst(prefix.count))
        let sessionID = suffix.isEmpty ? trimmed : String(trimmed.dropLast(suffix.count))
        let normalized = sessionID.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !normalized.isEmpty else {
            throw BridgeError.invalidRequest("missing session id in path \(path)")
        }
        return normalized
    }

    private func extractSessionIDAndCallID(from path: String) throws -> (String, String) {
        let components = path.split(separator: "/")
        guard components.count == 4,
              components[0] == "sessions",
              components[2] == "tool-results"
        else {
            throw BridgeError.invalidRequest("invalid tool result route: \(path)")
        }

        let sessionID = String(components[1])
        let callID = String(components[3])
        guard !sessionID.isEmpty, !callID.isEmpty else {
            throw BridgeError.invalidRequest("missing session id or call id in path \(path)")
        }
        return (sessionID, callID)
    }

    private static var foundationModelsAvailable: Bool {
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) {
            return true
        }
        return false
        #else
        return false
        #endif
    }
}

@main
struct FoundationModelsBridgeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "foundation-models-bridge",
        abstract: "Local HTTP bridge for Apple Foundation Models"
    )

    @Option(name: .long, help: "Hostname to bind")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Port to bind")
    var port: Int = 8008

    func run() async throws {
        let server = FoundationModelsHTTPServer()
        let router = server.buildRouter()
        let app = Application(router: router, configuration: .init(address: .hostname(host, port: port)))
        try await app.run()
    }
}

private func decodeBody<T: Decodable>(_ request: Request, as type: T.Type) async throws -> T {
    let body = try await request.body.collect(upTo: 10 * 1024 * 1024)
    let data = Data(buffer: body)
    return try JSONDecoder().decode(T.self, from: data)
}

private func encodeJSONResponse<T: Encodable>(_ value: T) throws -> Response {
    let data = try JSONEncoder().encode(value)
    var buffer = ByteBufferAllocator().buffer(capacity: data.count)
    buffer.writeBytes(data)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: buffer)
    )
}

private func errorResponse(status: HTTPResponse.Status, message: String) throws -> Response {
    try encodeJSONResponse(["error": message]).withStatus(status)
}

private func encodeSSEBuffer<T: Encodable>(_ value: T) -> ByteBuffer {
    let data = (try? JSONEncoder().encode(value)) ?? Data()
    let jsonString = String(data: data, encoding: .utf8) ?? "{}"
    let line = "data: \(jsonString)\n\n"
    var buffer = ByteBufferAllocator().buffer(capacity: line.utf8.count)
    buffer.writeString(line)
    return buffer
}

private extension Response {
    func withStatus(_ status: HTTPResponse.Status) -> Response {
        Response(status: status, headers: self.headers, body: self.body)
    }
}
