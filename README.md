# LlamaModelExecutor

A Swift library that bridges Apple's `FoundationModels` framework (macOS 27+) with a remote [llama.cpp](https://github.com/ggerganov/llama.cpp) server.

Implements the `LanguageModelExecutor` protocol so you can use llama.cpp models through FoundationModels' familiar `LanguageModelSession` API — the same API used by Apple's on-device and Private Cloud Compute models.

## Requirements

- macOS 27.0+ (FoundationModels is required)
- A running [llama.cpp server](https://github.com/ggerganov/llama.cpp) with OpenAI-compatible API enabled (`llama-server`)

## Installation

### Swift Package Manager

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/your-username/LlamaModelExecutor", from: "1.0.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["LlamaModelExecutor"]
    ),
]
```

## Usage

```swift
import FoundationModels
import LlamaModelExecutor

let config = LlamaConfiguration(
    modelName: "gemma-4-12b-it-Q4_K_M.gguf"
)
let model = LlamaModel(configuration: config)
let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "What is the capital of France?")
print(response.content)
```

### Streaming

```swift
for try await chunk in session.streamResponse(to: "Tell me a story") {
    print(chunk, terminator: "")
}
```

### Custom Configuration

```swift
let config = LlamaConfiguration(
    modelName: "llama-3.2-3b.gguf",
    temperature: 0.3,
    maxTokens: 4096,
    baseURL: URL(string: "http://192.168.1.100:8080/v1")!
)
```

## Architecture

```
LanguageModelSession       ← FoundationModels (system framework)
    └── LanguageModel
        └── LlamaModel     ← this library
            └── LlamaExecutor
                └── HTTPTransport (protocol)
                    ├── URLSessionTransport  ← production
                    └── MockTransport         ← tests

llama.cpp server           ← remote process (OpenAI-compatible API)
```

- `HTTPTransport` is injectable for unit testing. See `Tests/LlamaModelExecutorTests/` for examples with `MockTransport`.
- Model capabilities are declared honestly — no tool calling, vision, or reasoning advertised by default.

## License

MIT — see [LICENSE](LICENSE).
