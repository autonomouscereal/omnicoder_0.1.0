// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "OmniCoderDecodeConsole",
    platforms: [ .macOS(.v13) ],
    products: [ .executable(name: "OmniCoderDecodeConsole", targets: ["App"]) ],
    targets: [
        .executableTarget(
            name: "App",
            path: "Sources",
            resources: [.copy("Resources/omnicoder_decode_step.mlmodel")]
        )
    ]
)


