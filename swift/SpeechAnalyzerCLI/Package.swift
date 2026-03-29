// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SpeechAnalyzerCLI",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "speechanalyzer-cli", targets: ["SpeechAnalyzerCLI"]),
    ],
    targets: [
        .executableTarget(
            name: "SpeechAnalyzerCLI",
            path: "Sources"
        ),
    ]
)
