import Foundation
@preconcurrency import AVFoundation
import Speech
import SpeechAnalyzerCore
import Darwin
import Logging


private func makeSpeechAnalyzerLogger() -> Logger {
    var logger = Logger(label: "jarvis.bridge.speech_analyzer")
    logger[metadataKey: "bridge"] = "speech_analyzer"
    return logger
}

actor EventEmitter {
    func emit(_ event: SpeechAnalyzerEvent) throws {
        let data = try JSONEncoder().encode(event)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
        fflush(stdout)
    }
}

@available(macOS 26.0, *)
final class MicrophoneCapture: @unchecked Sendable {
    let audioEngine: AVAudioEngine
    let converter: AVAudioConverter
    let inputContinuation: AsyncStream<AnalyzerInput>.Continuation
    let targetFormat: AVAudioFormat

    init(
        targetFormat: AVAudioFormat,
        inputContinuation: AsyncStream<AnalyzerInput>.Continuation
    ) throws {
        self.targetFormat = targetFormat
        self.inputContinuation = inputContinuation
        self.audioEngine = AVAudioEngine()

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 else {
            throw CLIError.microphonePermissionDenied
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw CLIError.noCompatibleAudioFormat
        }
        self.converter = converter

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nil) { [self] buffer, _ in
            handleBuffer(buffer)
        }
    }

    func start() throws {
        do {
            try audioEngine.start()
        } catch {
            throw CLIError.microphonePermissionDenied
        }
    }

    func stop() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        inputContinuation.finish()
    }

    private func handleBuffer(_ buffer: AVAudioPCMBuffer) {
        let frameCapacity = AVAudioFrameCount(
            ceil(Double(buffer.frameLength) * targetFormat.sampleRate / converter.inputFormat.sampleRate)
        )
        guard frameCapacity > 0 else { return }
        guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCapacity) else {
            return
        }

        var error: NSError?
        nonisolated(unsafe) var consumed = false
        nonisolated(unsafe) let sourceBuffer = buffer

        converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        if error == nil, convertedBuffer.frameLength > 0 {
            inputContinuation.yield(AnalyzerInput(buffer: convertedBuffer))
        }
    }
}

@main
struct SpeechAnalyzerCLI {
    static func main() async throws {
        LoggingSystem.bootstrap(StreamLogHandler.standardError)
        let emitter = EventEmitter()
        var logger = makeSpeechAnalyzerLogger()
        logger.info("Starting speech analyzer bridge")

        do {
            let options = try parseOptions(arguments: Array(CommandLine.arguments.dropFirst()))
            logger[metadataKey: "locale"] = "\(options.locale)"
            guard options.format == "ndjson" else {
                throw CLIError.unsupportedFormat
            }

            if let transcript = options.mockTranscript {
                try await emitEvents(mockSequence(transcript: transcript), emitter: emitter)
                return
            }

            if options.live {
                try await runLive(options: options, emitter: emitter)
                return
            }

            throw CLIError.invalidArguments
        } catch {
            logger.error("Speech analyzer bridge failed", metadata: ["error": "\(error.localizedDescription)"])
            try? await emitter.emit(
                SpeechAnalyzerEvent(
                    type: "error",
                    message: error.localizedDescription
                )
            )
            if case CLIError.invalidArguments = error {
                FileHandle.standardError.write(Data("\(usageText)\n".utf8))
            }
            Foundation.exit(EXIT_FAILURE)
        }
    }

    static func emitEvents(_ events: [SpeechAnalyzerEvent], emitter: EventEmitter) async throws {
        for event in events {
            try await emitter.emit(event)
        }
    }

    static func runLive(options: CLIOptions, emitter: EventEmitter) async throws {
        var logger = makeSpeechAnalyzerLogger()
        logger[metadataKey: "locale"] = "\(options.locale)"

        guard #available(macOS 26.0, *) else {
            throw CLIError.unsupportedOS
        }

        guard SpeechTranscriber.isAvailable else {
            throw CLIError.speechTranscriberUnavailable
        }

        let locale = Locale(identifier: options.locale)
        let supportedLocales = await SpeechTranscriber.supportedLocales
        guard supportedLocales.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) else {
            throw CLIError.unsupportedLocale(locale.identifier(.bcp47))
        }

        let detector = SpeechDetector(
            detectionOptions: .init(sensitivityLevel: .medium),
            reportResults: true
        )
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults, .fastResults],
            attributeOptions: []
        )
        let modules: [any SpeechModule] = [detector, transcriber]

        try await ensureAssets(locale: locale, modules: modules)

        guard let targetFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: modules) else {
            throw CLIError.noCompatibleAudioFormat
        }

        let analyzer = SpeechAnalyzer(modules: modules)
        let (inputSequence, inputContinuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let capture = try MicrophoneCapture(targetFormat: targetFormat, inputContinuation: inputContinuation)

        try capture.start()
        try await analyzer.start(inputSequence: inputSequence)
        try await emitter.emit(
            SpeechAnalyzerEvent(
                type: "ready",
                locale: locale.identifier(.bcp47),
                sampleRate: targetFormat.sampleRate,
                channelCount: Int(targetFormat.channelCount)
            )
        )
        logger.info(
            "Speech analyzer ready",
            metadata: [
                "sample_rate": "\(targetFormat.sampleRate)",
                "channel_count": "\(targetFormat.channelCount)",
            ]
        )

        async let transcriberTask: Void = monitorTranscriber(transcriber, emitter: emitter)
        async let detectorTask: Void = monitorDetector(detector, emitter: emitter)

        do {
            _ = try await (transcriberTask, detectorTask)
        } catch let cancellation as CancellationError {
            capture.stop()
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
            throw cancellation
        } catch {
            capture.stop()
            try? await analyzer.finalizeAndFinishThroughEndOfInput()
            throw error
        }
    }

    @available(macOS 26.0, *)
    static func ensureAssets(locale: Locale, modules: [any SpeechModule]) async throws {
        let installedLocales = await SpeechTranscriber.installedLocales
        let isInstalled = installedLocales.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) })
        if !isInstalled {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
                try await request.downloadAndInstall()
            }
        }
    }

    @available(macOS 26.0, *)
    static func monitorTranscriber(
        _ transcriber: SpeechTranscriber,
        emitter: EventEmitter
    ) async throws {
        var lastPartial = ""
        for try await result in transcriber.results {
            let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }

            if result.isFinal {
                try await emitter.emit(
                    SpeechAnalyzerEvent(
                        type: "final_transcript",
                        text: text,
                        isFinal: true
                    )
                )
                lastPartial = ""
            } else if text != lastPartial {
                try await emitter.emit(
                    SpeechAnalyzerEvent(
                        type: "partial_transcript",
                        text: text,
                        isFinal: false
                    )
                )
                lastPartial = text
            }
        }
    }

    @available(macOS 26.0, *)
    static func monitorDetector(
        _ detector: SpeechDetector,
        emitter: EventEmitter
    ) async throws {
        var lastSpeechState: Bool?
        for try await result in detector.results {
            try await emitter.emit(
                SpeechAnalyzerEvent(
                    type: "speech_detector_result",
                    speechDetected: result.speechDetected
                )
            )

            if lastSpeechState != result.speechDetected {
                lastSpeechState = result.speechDetected
                try await emitter.emit(
                    SpeechAnalyzerEvent(
                        type: result.speechDetected ? "speech_started" : "speech_ended"
                    )
                )
            }
        }
    }
}
