import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

public struct BridgeMessagePayload: Codable, Sendable, Equatable {
    public let role: String
    public let content: String
    public let metadata: [String: JSONValue]?

    public init(role: String, content: String, metadata: [String: JSONValue]? = nil) {
        self.role = role
        self.content = content
        self.metadata = metadata
    }
}

public enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    public init(from decoder: Decoder) throws {
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

    public func encode(to encoder: Encoder) throws {
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

    public var stringValue: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }

    public var objectValue: [String: JSONValue]? {
        guard case let .object(value) = self else { return nil }
        return value
    }

    public var stringArrayValue: [String]? {
        guard case let .array(values) = self else { return nil }
        return values.compactMap { $0.stringValue }
    }

    public var serializedString: String {
        guard let data = try? JSONEncoder().encode(self),
              let json = String(data: data, encoding: .utf8)
        else {
            return "null"
        }
        return json
    }

    public var promptText: String {
        if case let .string(value) = self {
            return value
        }
        return self.serializedString
    }

    #if canImport(FoundationModels)
    public static func fromGeneratedContent(_ content: GeneratedContent) throws -> JSONValue {
        let data = Data(content.jsonString.utf8)
        let object = try JSONSerialization.jsonObject(with: data)
        return try fromJSONObject(object)
    }
    #endif

    public static func fromJSONObject(_ object: Any) throws -> JSONValue {
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

public struct ToolDefinitionPayload: Codable, Sendable, Equatable {
    public let name: String
    public let description: String?
    public let input_schema: JSONValue?

    public init(name: String, description: String? = nil, input_schema: JSONValue? = nil) {
        self.name = name
        self.description = description
        self.input_schema = input_schema
    }
}

public struct SessionCreateRequest: Codable, Sendable, Equatable {
    public let session_id: String?
    public let instructions: String?
    public let tools: [ToolDefinitionPayload]?

    public init(session_id: String? = nil, instructions: String? = nil, tools: [ToolDefinitionPayload]? = nil) {
        self.session_id = session_id
        self.instructions = instructions
        self.tools = tools
    }
}

public struct SessionCreateResponse: Codable, Sendable, Equatable {
    public let session_id: String

    public init(session_id: String) {
        self.session_id = session_id
    }
}

public struct SessionResponseRequest: Codable, Sendable, Equatable {
    public let prompt: String?
    public let messages: [BridgeMessagePayload]
    public let stream: Bool?

    public init(prompt: String? = nil, messages: [BridgeMessagePayload], stream: Bool? = nil) {
        self.prompt = prompt
        self.messages = messages
        self.stream = stream
    }
}

public struct SessionResponseBody: Codable, Sendable, Equatable {
    public let text: String

    public init(text: String) {
        self.text = text
    }
}

public struct ToolResultSubmitRequest: Codable, Sendable, Equatable {
    public let result: JSONValue

    public init(result: JSONValue) {
        self.result = result
    }
}

public struct HealthResponse: Codable, Sendable, Equatable {
    public let status: String
    public let availability: String
    public let detail: String?

    public init(status: String, availability: String, detail: String? = nil) {
        self.status = status
        self.availability = availability
        self.detail = detail
    }
}

public struct SSEEvent: Codable, Sendable, Equatable {
    public let type: String
    public let text: String?
    public let name: String?
    public let callID: String?
    public let args: JSONValue?
    public let result: JSONValue?
    public let message: String?

    public enum CodingKeys: String, CodingKey {
        case type
        case text
        case name
        case callID = "call_id"
        case args
        case result
        case message
    }

    public init(
        type: String,
        text: String? = nil,
        name: String? = nil,
        callID: String? = nil,
        args: JSONValue? = nil,
        result: JSONValue? = nil,
        message: String? = nil
    ) {
        self.type = type
        self.text = text
        self.name = name
        self.callID = callID
        self.args = args
        self.result = result
        self.message = message
    }
}

public enum BridgeError: Error, LocalizedError, Equatable {
    case unsupportedPlatform
    case modelUnavailable(String)
    case missingSession(String)
    case invalidRequest(String)
    case sessionBusy(String)
    case cancelled(String)

    public var errorDescription: String? {
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

public enum BridgeAvailabilityState: Sendable, Equatable {
    case available
    case appleIntelligenceNotEnabled
    case modelNotReady
    case deviceNotEligible
    case unavailable
    case unsupported
    case unknown
}

public enum FoundationModelsBridgeSupport {
    public static func prompt(from payload: SessionResponseRequest) throws -> String {
        let prompt = payload.prompt ?? payload.messages.last(where: { $0.role == "user" })?.content ?? ""
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BridgeError.invalidRequest("response prompt must not be empty")
        }
        return prompt
    }

    public static func extractSessionID(from path: String, suffix: String) throws -> String {
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

    public static func extractSessionIDAndCallID(from path: String) throws -> (String, String) {
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

    public static func availabilityLabel(for state: BridgeAvailabilityState) -> String {
        switch state {
        case .available:
            return "available"
        case .appleIntelligenceNotEnabled:
            return "apple_intelligence_not_enabled"
        case .modelNotReady:
            return "model_not_ready"
        case .deviceNotEligible:
            return "device_not_eligible"
        case .unavailable:
            return "unavailable"
        case .unsupported:
            return "unsupported"
        case .unknown:
            return "unknown"
        }
    }

    public static func availabilityDescription(for state: BridgeAvailabilityState) -> String {
        switch state {
        case .available:
            return "Foundation Models is available."
        case .appleIntelligenceNotEnabled:
            return "Apple Intelligence must be enabled in System Settings."
        case .modelNotReady:
            return "The on-device Foundation model is not ready yet."
        case .deviceNotEligible:
            return "This device does not support Apple Intelligence."
        case .unavailable:
            return "Foundation Models is unavailable."
        case .unsupported:
            return BridgeError.unsupportedPlatform.localizedDescription
        case .unknown:
            return "Foundation Models availability is unknown."
        }
    }

    public static func validateAvailabilityState(_ state: BridgeAvailabilityState) throws {
        switch state {
        case .available:
            return
        case .unsupported:
            throw BridgeError.unsupportedPlatform
        default:
            throw BridgeError.modelUnavailable(availabilityDescription(for: state))
        }
    }

    public static func healthPayload(for state: BridgeAvailabilityState) -> HealthResponse {
        if state == .available {
            return HealthResponse(status: "ok", availability: availabilityLabel(for: state), detail: nil)
        }
        return HealthResponse(
            status: "unavailable",
            availability: availabilityLabel(for: state),
            detail: availabilityDescription(for: state)
        )
    }

    public static func encodeJSONData<T: Encodable>(_ value: T) throws -> Data {
        try JSONEncoder().encode(value)
    }

    public static func encodeSSELine<T: Encodable>(_ value: T) -> String {
        let data = (try? JSONEncoder().encode(value)) ?? Data()
        let jsonString = String(data: data, encoding: .utf8) ?? "{}"
        return "data: \(jsonString)\n\n"
    }
}
