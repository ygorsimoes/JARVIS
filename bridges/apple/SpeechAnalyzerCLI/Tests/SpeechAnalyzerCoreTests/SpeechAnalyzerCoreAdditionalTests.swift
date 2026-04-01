import Foundation
import SpeechAnalyzerCore
import Testing

@Suite("SpeechAnalyzerCoreAdditional")
struct SpeechAnalyzerCoreAdditionalTests {
    @Test(arguments: [
        (CLIError.unsupportedFormat, "unsupported_format"),
        (CLIError.unsupportedOS, "unsupported_os"),
        (CLIError.speechTranscriberUnavailable, "speech_transcriber_unavailable"),
        (CLIError.noCompatibleAudioFormat, "no_compatible_audio_format"),
        (CLIError.microphonePermissionDenied, "microphone_permission_denied"),
    ])
    func cliErrorDescriptionsRemainStable(
        sample: (CLIError, String)
    ) {
        let (error, expectedDescription) = sample
        #expect(error.localizedDescription == expectedDescription)
    }

    @Test
    func parseOptionsSupportsLocaleAndFormatInMockMode() throws {
        let options = try parseOptions(
            arguments: [
                "--mock-transcript", "Ola mundo",
                "--locale", "pt-BR",
                "--format", "ndjson",
            ]
        )

        #expect(options.mockTranscript == "Ola mundo")
        #expect(options.locale == "pt-BR")
        #expect(options.format == "ndjson")
        #expect(options.live == false)
        #expect(options.vadOnly == false)
    }

    @Test
    func parseOptionsSupportsVadOnlyMode() throws {
        let options = try parseOptions(
            arguments: [
                "--live",
                "--vad-only",
                "--format", "ndjson",
            ]
        )

        #expect(options.live)
        #expect(options.vadOnly)
        #expect(options.format == "ndjson")
    }

    @Test(arguments: [
        ["--bogus"],
        ["--mock-transcript"],
        ["--format"],
    ])
    func parseOptionsRejectsInvalidArgumentShapes(arguments: [String]) {
        do {
            _ = try parseOptions(arguments: arguments)
            Issue.record("Expected invalid arguments to throw")
        } catch let error as CLIError {
            #expect(error == .invalidArguments)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func mockSequenceSkipsPartialForEmptyTranscript() {
        let events = mockSequence(transcript: "")

        #expect(events.map(\.type) == [
            "speech_started",
            "speech_detector_result",
            "speech_detector_result",
            "speech_ended",
            "final_transcript",
        ])
    }

    @Test
    func speechAnalyzerEventUsesSnakeCaseCodingKeys() throws {
        let event = SpeechAnalyzerEvent(
            type: "ready",
            text: "Ola",
            isFinal: true,
            confidence: 0.91,
            speechDetected: true,
            locale: "pt-BR",
            sampleRate: 16_000,
            channelCount: 1,
            message: "ok"
        )

        let data = try JSONEncoder().encode(event)
        let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(payload?["is_final"] as? Bool == true)
        #expect(payload?["speech_detected"] as? Bool == true)
        #expect(payload?["sample_rate"] as? Double == 16_000)
        #expect(payload?["channel_count"] as? Int == 1)
    }
}
