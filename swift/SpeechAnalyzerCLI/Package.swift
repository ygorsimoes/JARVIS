// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SpeechAnalyzerCLI",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "speechanalyzer-cli", targets: ["SpeechAnalyzerCLI"]),
    ],
    targets: [
        .target(
            name: "SpeechAnalyzerCore",
            path: "Sources/SpeechAnalyzerCore"
        ),
        .executableTarget(
            name: "SpeechAnalyzerCLI",
            dependencies: ["SpeechAnalyzerCore"],
            path: "Sources/SpeechAnalyzerCLI"
        ),
        .testTarget(
            name: "SpeechAnalyzerCoreTests",
            dependencies: ["SpeechAnalyzerCore"],
            path: "Tests/SpeechAnalyzerCoreTests"
        ),
    ]
)
