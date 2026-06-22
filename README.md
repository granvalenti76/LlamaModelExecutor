# LlamaModelExecutor
EXPERIMENTAL<p>
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
    .package(url: "https://github.com/granvalenti76/LlamaModelExecutor", from: "1.0.0"),
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

## Tool Calling

llama.cpp supports OpenAI-compatible tool/function calling. The library maps FoundationModels'
`Transcript.ToolDefinition` and `Transcript.ToolCall`/`ToolOutput` entries to the wire format:

```swift
let request = LanguageModelExecutorGenerationRequest(
    transcript: transcript,
    enabledTools: [
        Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the current weather for a city",
            parameters: GenerationSchema(
                type: String.self,
                description: "city name",
                properties: []
            )
        )
    ],
    generationOptions: GenerationOptions(toolCallingMode: .allowed),
    ...
)
```

When the model responds with a tool call, `StreamChunk.ToolCallDelta` events are forwarded
through the channel as `LanguageModelExecutorGenerationChannel.ToolCalls` events.

## Architecture

```
LanguageModelSession       ← FoundationModels (system framework)
    └── LanguageModel
        └── LlamaModel     ← this library
            └── LlamaExecutor
                ├── RequestBuilder     → HTTP body
                ├── SSEStreamParser    ← SSE lines → StreamChunk
                ├── TokenTracker       ← token accounting
                ├── ChannelForwarder   → channel events
                └── HTTPTransport (protocol)
                    ├── URLSessionTransport  ← production
                    └── MockTransport         ← tests

llama.cpp server           ← remote process (OpenAI-compatible API)
```

- `HTTPTransport` is injectable for unit testing. See `Tests/LlamaModelExecutorTests/` for examples with `MockTransport`.
- Tool calling, streaming, and metadata forwarding are supported out of the box.

## License

MIT — see [LICENSE](LICENSE).
