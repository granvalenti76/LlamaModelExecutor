// swift-tools-version: 6.4

import PackageDescription

let package = Package(
    name: "LlamaModelExecutor",

    platforms: [
        .iOS("27.0"), .macOS("27.0"), .visionOS("27.0"), .watchOS("27.0"),
    ],

    products: [
        .library(
            name: "LlamaModelExecutor",
            targets: ["LlamaModelExecutor"]
        ),
    ],


    targets: [
        .target(
            name: "LlamaModelExecutor",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency=complete"),
            ],
        ),
        .executableTarget(
            name: "LlamaTest",
            dependencies: ["LlamaModelExecutor"],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency=complete"),
            ],
        ),
        .testTarget(
            name: "LlamaModelExecutorTests",
            dependencies: ["LlamaModelExecutor"],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency=complete"),
            ],
        ),

    ],
    swiftLanguageModes: [.v6]
)
