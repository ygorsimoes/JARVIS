import ArgumentParser
import Foundation
import FoundationModelsBridgeCore
import Hummingbird
#if canImport(FoundationModels)
import FoundationModels
#endif

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
                    name: name,
                    callID: callID,
                    args: arguments
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
                    name: name,
                    callID: callID,
                    result: result
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
                continuation.yield(encodeSSEBuffer(SSEEvent(type: "response_chunk", text: delta)))
            }

            continuation.yield(encodeSSEBuffer(SSEEvent(type: "response_end")))
        } catch is CancellationError {
            continuation.yield(
                encodeSSEBuffer(
                    SSEEvent(
                        type: "error",
                        message: BridgeError.cancelled(sessionID).localizedDescription
                    )
                )
            )
        } catch {
            continuation.yield(encodeSSEBuffer(SSEEvent(type: "error", message: error.localizedDescription)))
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
            let sessionID = try FoundationModelsBridgeSupport.extractSessionID(from: request.uri.path, suffix: "")
            await self.sessionStore.delete(sessionID)
            return try encodeJSONResponse(["deleted": true])
        }

        router.post("/sessions/:sessionID/cancel") { request, _ in
            let sessionID = try FoundationModelsBridgeSupport.extractSessionID(from: request.uri.path, suffix: "/cancel")
            guard let session = await self.sessionStore.get(sessionID) else {
                return try errorResponse(status: .notFound, message: BridgeError.missingSession(sessionID).localizedDescription)
            }
            let cancelled = await session.cancelActiveResponse()
            return try encodeJSONResponse(["cancelled": cancelled])
        }

        router.post("/sessions/:sessionID/tool-results/:callID") { request, _ in
            let (sessionID, callID) = try FoundationModelsBridgeSupport.extractSessionIDAndCallID(from: request.uri.path)
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
            let sessionID = try FoundationModelsBridgeSupport.extractSessionID(from: request.uri.path, suffix: "/responses")
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
        try FoundationModelsBridgeSupport.validateAvailabilityState(self.currentAvailabilityState())
        let prompt = try FoundationModelsBridgeSupport.prompt(from: payload)
        let fullText = try await session.respond(prompt: prompt)
        return try encodeJSONResponse(SessionResponseBody(text: fullText))
    }

    private func handleStreamingRequest(
        session: BridgeSessionActor,
        payload: SessionResponseRequest
    ) async throws -> Response {
        try FoundationModelsBridgeSupport.validateAvailabilityState(self.currentAvailabilityState())
        let prompt = try FoundationModelsBridgeSupport.prompt(from: payload)
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

    private func healthPayload() -> HealthResponse {
        FoundationModelsBridgeSupport.healthPayload(for: self.currentAvailabilityState())
    }

    private func currentAvailabilityState() -> BridgeAvailabilityState {
        #if canImport(FoundationModels)
        let availability = SystemLanguageModel.default.availability
        switch availability {
        case .available:
            return .available
        case .unavailable(.appleIntelligenceNotEnabled):
            return .appleIntelligenceNotEnabled
        case .unavailable(.modelNotReady):
            return .modelNotReady
        case .unavailable(.deviceNotEligible):
            return .deviceNotEligible
        case .unavailable:
            return .unavailable
        @unknown default:
            return .unknown
        }
        #else
        return .unsupported
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
    let data = try FoundationModelsBridgeSupport.encodeJSONData(value)
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
    let line = FoundationModelsBridgeSupport.encodeSSELine(value)
    var buffer = ByteBufferAllocator().buffer(capacity: line.utf8.count)
    buffer.writeString(line)
    return buffer
}

private extension Response {
    func withStatus(_ status: HTTPResponse.Status) -> Response {
        Response(status: status, headers: self.headers, body: self.body)
    }
}
