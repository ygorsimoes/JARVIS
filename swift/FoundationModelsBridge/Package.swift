// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "FoundationModelsBridge",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .executable(name: "foundation-models-bridge", targets: ["FoundationModelsBridge"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.19.0"),
    ],
    targets: [
        .executableTarget(
            name: "FoundationModelsBridge",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Hummingbird", package: "hummingbird"),
            ],
            path: "Sources"
        ),
    ]
)
