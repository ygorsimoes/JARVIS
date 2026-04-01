import SpeechAnalyzerCore
import Testing

struct SpeechAnalyzerCoreTests {
    @Test
    func parseOptionsForLiveMode() throws {
        let options = try parseOptions(arguments: ["--live", "--locale", "en-US", "--format", "ndjson"])

        #expect(options == CLIOptions(live: true, locale: "en-US", format: "ndjson", mockTranscript: nil))
    }

    @Test
    func parseOptionsForMockTranscript() throws {
        let options = try parseOptions(arguments: ["--mock-transcript", "Que horas sao agora?"])

        #expect(options.mockTranscript == "Que horas sao agora?")
        #expect(options.live == false)
    }

    @Test
    func parseOptionsRejectsInvalidArguments() {
        do {
            _ = try parseOptions(arguments: ["--locale"])
            Issue.record("Expected invalid arguments error")
        } catch let error as CLIError {
            #expect(error == .invalidArguments)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test
    func mockSequenceProducesExpectedEventOrder() {
        let events = mockSequence(transcript: "Que horas sao agora")

        #expect(events.map(\.type) == [
            "speech_started",
            "speech_detector_result",
            "partial_transcript",
            "speech_detector_result",
            "speech_ended",
            "final_transcript",
        ])
        #expect(events[2].text == "Que horas sao")
        #expect(events[5].text == "Que horas sao agora")
    }

    @Test
    func usageTextMentionsBothSupportedModes() {
        #expect(usageText.contains("--live"))
        #expect(usageText.contains("--mock-transcript"))
    }
}
