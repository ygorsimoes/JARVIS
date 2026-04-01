#if canImport(FoundationModels)
import Foundation
import FoundationModelsBridgeCore
import NIOCore
import Testing
@testable import FoundationModelsBridge

@Suite("FoundationModelsBridgeActors")
struct FoundationModelsBridgeActorTests {
    @Test
    func toolInvocationCoordinatorEmitsToolCallAndToolResultEvents() async throws {
        let coordinator = ToolInvocationCoordinator(sessionID: "session-123")
        let (stream, continuation) = AsyncStream.makeStream(of: ByteBuffer.self)
        await coordinator.beginStreaming(continuation)

        let task = Task {
            try await coordinator.requestToolCall(
                name: "system.get_time",
                arguments: .object([:])
            )
        }

        var iterator = stream.makeAsyncIterator()
        guard let firstBuffer = await iterator.next() else {
            Issue.record("Expected tool_call event")
            continuation.finish()
            return
        }

        let firstEvent = try decodeSSEEvent(firstBuffer)
        guard let callID = firstEvent.callID else {
            Issue.record("Expected call_id in tool_call event")
            continuation.finish()
            return
        }

        #expect(firstEvent.type == "tool_call")
        #expect(firstEvent.name == "system.get_time")

        let accepted = await coordinator.submitToolResult(
            callID: callID,
            result: .string("Agora sao 12:34")
        )
        #expect(accepted)

        let promptText = try await task.value
        #expect(promptText == "Agora sao 12:34")

        guard let secondBuffer = await iterator.next() else {
            Issue.record("Expected tool_result event")
            continuation.finish()
            return
        }

        let secondEvent = try decodeSSEEvent(secondBuffer)
        #expect(secondEvent.type == "tool_result")
        #expect(secondEvent.callID == callID)
        #expect(secondEvent.result == .string("Agora sao 12:34"))

        continuation.finish()
        await coordinator.finishStreaming()
    }

    @Test
    func cancellingPendingToolCallsFailsOutstandingRequests() async throws {
        let coordinator = ToolInvocationCoordinator(sessionID: "session-456")
        let (stream, continuation) = AsyncStream.makeStream(of: ByteBuffer.self)
        await coordinator.beginStreaming(continuation)

        let task = Task {
            try await coordinator.requestToolCall(
                name: "system.get_time",
                arguments: .object([:])
            )
        }

        var iterator = stream.makeAsyncIterator()
        guard let firstBuffer = await iterator.next() else {
            Issue.record("Expected tool_call event before cancellation")
            continuation.finish()
            return
        }

        let firstEvent = try decodeSSEEvent(firstBuffer)
        #expect(firstEvent.type == "tool_call")

        await coordinator.cancelPendingToolCalls()

        do {
            _ = try await task.value
            Issue.record("Expected cancellation to throw")
        } catch let error as BridgeError {
            #expect(error == .cancelled("session-456"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        continuation.finish()
        await coordinator.finishStreaming()
    }

    @Test
    func submitToolResultReturnsFalseForUnknownCallIDs() async {
        let coordinator = ToolInvocationCoordinator(sessionID: "session-789")

        let accepted = await coordinator.submitToolResult(
            callID: "missing-call",
            result: .string("ignored")
        )

        #expect(!accepted)
    }
}

private func decodeSSEEvent(_ buffer: ByteBuffer) throws -> SSEEvent {
    var copy = buffer
    guard let line = copy.readString(length: copy.readableBytes) else {
        throw BridgeError.invalidRequest("expected SSE buffer contents")
    }

    let prefix = "data: "
    let payload = line.hasPrefix(prefix)
        ? String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        : line.trimmingCharacters(in: .whitespacesAndNewlines)
    return try JSONDecoder().decode(SSEEvent.self, from: Data(payload.utf8))
}
#endif
