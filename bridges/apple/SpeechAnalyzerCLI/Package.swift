// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SpeechAnalyzerCLI",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "speechanalyzer-cli", targets: ["SpeechAnalyzerCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log.git", from: "1.6.0"),
        .package(
            url: "https://github.com/swiftlang/swift-testing.git",
            branch: "release/6.2"
        ),
    ],
    targets: [
        .target(
            name: "SpeechAnalyzerCore",
            path: "Sources/SpeechAnalyzerCore"
        ),
        .executableTarget(
            name: "SpeechAnalyzerCLI",
            dependencies: [
                "SpeechAnalyzerCore",
                .product(name: "Logging", package: "swift-log"),
            ],
            path: "Sources/SpeechAnalyzerCLI"
        ),
        .testTarget(
            name: "SpeechAnalyzerCoreTests",
            dependencies: [
                "SpeechAnalyzerCore",
                .product(name: "Testing", package: "swift-testing"),
            ],
            path: "Tests/SpeechAnalyzerCoreTests"
        ),
    ]
)
