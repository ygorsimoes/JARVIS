import Foundation

public struct CLIOptions: Sendable, Equatable {
    public var live: Bool
    public var vadOnly: Bool
    public var locale: String
    public var format: String
    public var mockTranscript: String?

    public init(
        live: Bool = false,
        vadOnly: Bool = false,
        locale: String = "pt-BR",
        format: String = "ndjson",
        mockTranscript: String? = nil
    ) {
        self.live = live
        self.vadOnly = vadOnly
        self.locale = locale
        self.format = format
        self.mockTranscript = mockTranscript
    }
}

public enum CLIError: Error, LocalizedError, Equatable {
    case invalidArguments
    case unsupportedFormat
    case unsupportedOS
    case speechTranscriberUnavailable
    case unsupportedLocale(String)
    case noCompatibleAudioFormat
    case microphonePermissionDenied

    public var errorDescription: String? {
        switch self {
        case .invalidArguments:
            return "invalid_arguments"
        case .unsupportedFormat:
            return "unsupported_format"
        case .unsupportedOS:
            return "unsupported_os"
        case .speechTranscriberUnavailable:
            return "speech_transcriber_unavailable"
        case let .unsupportedLocale(locale):
            return "unsupported_locale:\(locale)"
        case .noCompatibleAudioFormat:
            return "no_compatible_audio_format"
        case .microphonePermissionDenied:
            return "microphone_permission_denied"
        }
    }
}

public struct SpeechAnalyzerEvent: Codable, Sendable, Equatable {
    public let type: String
    public let text: String?
    public let isFinal: Bool?
    public let confidence: Double?
    public let speechDetected: Bool?
    public let locale: String?
    public let sampleRate: Double?
    public let channelCount: Int?
    public let message: String?

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case isFinal = "is_final"
        case confidence
        case speechDetected = "speech_detected"
        case locale
        case sampleRate = "sample_rate"
        case channelCount = "channel_count"
        case message
    }

    public init(
        type: String,
        text: String? = nil,
        isFinal: Bool? = nil,
        confidence: Double? = nil,
        speechDetected: Bool? = nil,
        locale: String? = nil,
        sampleRate: Double? = nil,
        channelCount: Int? = nil,
        message: String? = nil
    ) {
        self.type = type
        self.text = text
        self.isFinal = isFinal
        self.confidence = confidence
        self.speechDetected = speechDetected
        self.locale = locale
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.message = message
    }
}

public func parseOptions(arguments: [String]) throws -> CLIOptions {
    var options = CLIOptions()
    var iterator = arguments.makeIterator()

    while let argument = iterator.next() {
        switch argument {
        case "--live":
            options.live = true
        case "--vad-only":
            options.vadOnly = true
        case "--locale":
            guard let locale = iterator.next() else { throw CLIError.invalidArguments }
            options.locale = locale
        case "--format":
            guard let format = iterator.next() else { throw CLIError.invalidArguments }
            options.format = format
        case "--mock-transcript":
            guard let transcript = iterator.next() else { throw CLIError.invalidArguments }
            options.mockTranscript = transcript
        default:
            throw CLIError.invalidArguments
        }
    }

    return options
}

public func mockSequence(transcript: String) -> [SpeechAnalyzerEvent] {
    let partial = transcript.split(separator: " ").prefix(3).joined(separator: " ")
    var events: [SpeechAnalyzerEvent] = [
        SpeechAnalyzerEvent(type: "speech_started"),
        SpeechAnalyzerEvent(type: "speech_detector_result", speechDetected: true),
    ]

    if !partial.isEmpty {
        events.append(
            SpeechAnalyzerEvent(
                type: "partial_transcript",
                text: partial,
                confidence: 0.80
            )
        )
    }

    events.append(contentsOf: [
        SpeechAnalyzerEvent(type: "speech_detector_result", speechDetected: false),
        SpeechAnalyzerEvent(type: "speech_ended"),
        SpeechAnalyzerEvent(type: "final_transcript", text: transcript, confidence: 0.98),
    ])
    return events
}

public let usageText = """
Usage: speechanalyzer-cli --live --locale pt-BR --format ndjson
       speechanalyzer-cli --live --vad-only --format ndjson
       speechanalyzer-cli --mock-transcript \"Que horas sao agora?\"
"""
