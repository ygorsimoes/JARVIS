import Foundation
import FoundationModelsBridgeCore
import Testing

@Suite("FoundationModelsBridgeSupport")
struct FoundationModelsBridgeSupportAdditionalTests {
    @Test(arguments: [
        BridgeAvailabilityState.available,
        .appleIntelligenceNotEnabled,
        .modelNotReady,
        .deviceNotEligible,
        .unavailable,
        .unsupported,
        .unknown,
    ])
    func availabilityStateAlwaysProducesStableLabelsAndDescriptions(
        state: BridgeAvailabilityState
    ) {
        let label = FoundationModelsBridgeSupport.availabilityLabel(for: state)
        let description = FoundationModelsBridgeSupport.availabilityDescription(for: state)

        #expect(!label.isEmpty)
        #expect(!description.isEmpty)
    }

    @Test(arguments: [
        "/sessions/session-123/responses",
        "/sessions/session-123/cancel",
        "/sessions/session-123",
    ])
    func extractsSessionIdentifiersFromMultipleRoutes(path: String) throws {
        let suffix = path == "/sessions/session-123" ? "" : "/" + path.split(separator: "/").last!
        let sessionID = try FoundationModelsBridgeSupport.extractSessionID(
            from: path,
            suffix: String(suffix)
        )

        #expect(sessionID == "session-123")
    }

    @Test(arguments: [
        "/invalid-route",
        "/sessions/",
        "/sessions//responses",
    ])
    func rejectsInvalidSessionRoutes(path: String) {
        do {
            _ = try FoundationModelsBridgeSupport.extractSessionID(
                from: path,
                suffix: "/responses"
            )
            Issue.record("Expected invalid route to throw")
        } catch let error as BridgeError {
            if path == "/invalid-route" {
                #expect(error == .invalidRequest("invalid session route: /invalid-route"))
            } else {
                #expect(error.localizedDescription.contains("missing session id"))
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test(arguments: [
        "/sessions/session-123/tool-results",
        "/sessions//tool-results/call-456",
        "/bad/session-123/tool-results/call-456",
    ])
    func rejectsInvalidToolResultRoutes(path: String) {
        do {
            _ = try FoundationModelsBridgeSupport.extractSessionIDAndCallID(from: path)
            Issue.record("Expected invalid tool-result route to throw")
        } catch let error as BridgeError {
            let localizedDescription = error.localizedDescription
            #expect(
                localizedDescription.contains("invalid tool result route")
                    || localizedDescription.contains("missing session id or call id")
            )
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func jsonValueFromJSONObjectCoversCommonContainerTypes() throws {
        let payload: [String: Any] = [
            "name": "jarvis",
            "enabled": true,
            "count": 2,
            "ratio": 1.5,
            "items": [1, "two", NSNull()],
            "nested": ["scope": "global"],
        ]

        let value = try JSONValue.fromJSONObject(payload)

        #expect(value == .object([
            "name": .string("jarvis"),
            "enabled": .bool(true),
            "count": .integer(2),
            "ratio": .number(1.5),
            "items": .array([.integer(1), .string("two"), .null]),
            "nested": .object(["scope": .string("global")]),
        ]))
    }

    @Test
    func promptTextSerializesStructuredValues() {
        let value = JSONValue.object([
            "time": .string("12:34"),
            "timezone": .string("UTC"),
        ])

        let prompt = value.promptText

        #expect(prompt.contains("12:34"))
        #expect(prompt.contains("timezone"))
    }

    @Test
    func rejectsUnsupportedJSONObjectValues() {
        do {
            _ = try JSONValue.fromJSONObject(Date())
            Issue.record("Expected unsupported JSON object to throw")
        } catch let error as BridgeError {
            #expect(error == .invalidRequest("unsupported JSON value in tool payload"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
