import Foundation
import FoundationModelsBridgeCore
import Testing

struct FoundationModelsBridgeCoreTests {
    @Test
    func jsonValueRoundTripsThroughJSON() throws {
        let original = JSONValue.object([
            "name": .string("jarvis"),
            "enabled": .bool(true),
            "count": .integer(2),
            "items": .array([.string("a"), .number(1.5)]),
        ])

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(JSONValue.self, from: data)

        #expect(decoded == original)
    }

    @Test
    func extractsPromptFromExplicitPromptOrLastUserMessage() throws {
        let explicit = try FoundationModelsBridgeSupport.prompt(
            from: SessionResponseRequest(
                prompt: "Use o prompt explicito",
                messages: [BridgeMessagePayload(role: "user", content: "Ignorar")]
            )
        )
        #expect(explicit == "Use o prompt explicito")

        let fallback = try FoundationModelsBridgeSupport.prompt(
            from: SessionResponseRequest(
                messages: [
                    BridgeMessagePayload(role: "assistant", content: "oi"),
                    BridgeMessagePayload(role: "user", content: "ultimo pedido"),
                ]
            )
        )
        #expect(fallback == "ultimo pedido")
    }

    @Test
    func rejectsEmptyPromptPayload() {
        do {
            _ = try FoundationModelsBridgeSupport.prompt(
                from: SessionResponseRequest(messages: [BridgeMessagePayload(role: "assistant", content: "sem prompt")])
            )
            Issue.record("Expected invalid request for empty prompt")
        } catch let error as BridgeError {
            #expect(error == .invalidRequest("response prompt must not be empty"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func extractsSessionIdentifiersFromRoutes() throws {
        let sessionID = try FoundationModelsBridgeSupport.extractSessionID(
            from: "/sessions/session-123/responses",
            suffix: "/responses"
        )
        let ids = try FoundationModelsBridgeSupport.extractSessionIDAndCallID(
            from: "/sessions/session-123/tool-results/call-456"
        )

        #expect(sessionID == "session-123")
        #expect(ids.0 == "session-123")
        #expect(ids.1 == "call-456")
    }

    @Test
    func reportsAvailabilityAndValidationState() throws {
        let unavailable = FoundationModelsBridgeSupport.healthPayload(for: .unsupported)
        let available = FoundationModelsBridgeSupport.healthPayload(for: .available)

        #expect(unavailable.status == "unavailable")
        #expect(unavailable.availability == "unsupported")
        #expect(available.status == "ok")
        #expect(available.availability == "available")

        do {
            try FoundationModelsBridgeSupport.validateAvailabilityState(.modelNotReady)
            Issue.record("Expected unavailable state to throw")
        } catch let error as BridgeError {
            #expect(error == .modelUnavailable("The on-device Foundation model is not ready yet."))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func encodesSSELinesWithDataPrefix() throws {
        let line = FoundationModelsBridgeSupport.encodeSSELine(SSEEvent(type: "response_chunk", text: "ola"))

        #expect(line.hasPrefix("data: "))
        #expect(line.hasSuffix("\n\n"))

        let payload = String(line.dropFirst("data: ".count).dropLast(2))
        let event = try JSONDecoder().decode(SSEEvent.self, from: Data(payload.utf8))
        #expect(event == SSEEvent(type: "response_chunk", text: "ola"))
    }
}
