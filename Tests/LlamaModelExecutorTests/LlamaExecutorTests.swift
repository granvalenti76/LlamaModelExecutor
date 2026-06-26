//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Testing
import Foundation
import FoundationModels
@testable import LlamaModelExecutor

// MARK: - Mock transport

struct MockTransport: HTTPTransport {
    let response: URLResponse
    let data: Data
    let lines: [String]

    init(
        statusCode: Int = 200,
        data: Data = Data(),
        lines: [String] = []
    ) {
        self.response = HTTPURLResponse(
            url: URL(string: "http://127.0.0.1:8080/v1/chat/completions")!,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: nil
        )!
        self.data = data
        self.lines = lines
    }

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        (data, response)
    }

    func lines(for request: URLRequest) async throws -> (AsyncThrowingStream<String, Error>, URLResponse) {
        let stream = AsyncThrowingStream<String, Error> { continuation in
            for line in lines {
                continuation.yield(line)
            }
            continuation.finish()
        }
        return (stream, response)
    }
}

// MARK: - Mock JSON builders

/// Builds JSON strings that mirror the actual llama.cpp server format.
///
/// Real server observations:
/// - Final chunk has `timings` at top level, **no** `usage` object
/// - Reasoning models emit `reasoning_content` deltas before `content`
/// - `timings` includes `predicted_per_second`, `predicted_per_token_ms`
/// - Tool call chunks follow OpenAI streaming delta format
enum MockJSON {
    static func textDelta(_ text: String) -> String {
        "{\"choices\":[{\"delta\":{\"content\":\"\(text)\"}}]}"
    }

    static func reasoningDelta(_ text: String) -> String {
        "{\"choices\":[{\"delta\":{\"reasoning_content\":\"\(text)\"}}]}"
    }

    static let roleAnnouncement = "{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}"

    /// Final chunk as sent by llama.cpp: no `usage`, just `timings`.
    static func finish(promptTokens: Int = 10, completionTokens: Int = 20) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":50.0}}"
    }

    /// Final chunk with full timing info (llama.cpp format, no `usage`).
    static func finishWithTimingsOnly(promptTokens: Int = 10, completionTokens: Int = 20) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":42.5,\"predicted_per_token_ms\":23.5}}"
    }

    /// Final chunk with both `usage` and `timings` (OpenAI-compatible servers).
    static func finishWithUsage(promptTokens: Int = 10, completionTokens: Int = 20) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":\(promptTokens),\"completion_tokens\":\(completionTokens)},\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":50.0}}"
    }

    /// Final chunk with `completion_tokens_details.reasoning_tokens` (rare, some servers support it).
    static func finishWithUsageAndReasoning(promptTokens: Int = 10, completionTokens: Int = 20, reasoningTokens: Int = 5) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":\(promptTokens),\"completion_tokens\":\(completionTokens),\"completion_tokens_details\":{\"reasoning_tokens\":\(reasoningTokens)}},\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":50.0}}"
    }

    /// Final chunk with `finish_reason: "tool_calls"` (llama.cpp format, no usage).
    static func finishWithToolCalls(promptTokens: Int = 10, completionTokens: Int = 20) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":50.0}}"
    }

    // MARK: - Tool call deltas

    /// First chunk for a tool call: announces id, name, and begins arguments.
    /// - Parameters:
    ///   - id: The tool call identifier.
    ///   - name: The function name.
    ///   - arguments: Initial arguments fragment (may be empty string).
    static func toolCallBegin(id: String = "call_abc123", name: String = "get_weather", arguments: String = "") -> String {
        """
{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"\(id)\",\"type\":\"function\",\"function\":{\"name\":\"\(name)\",\"arguments\":\"\(arguments)\"}}]}}]}
""".trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Subsequent chunk for a tool call: appends incremental arguments.
    /// - Parameters:
    ///   - index: The tool call index.
    ///   - arguments: Incremental arguments fragment.
    static func toolCallAppend(index: Int = 0, arguments: String) -> String {
        let escaped = arguments
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":\(index),\"function\":{\"arguments\":\"\(escaped)\"}}]}}]}"
    }

    static let done = "[DONE]"
}

extension String {
    var sseData: String { "data: \(self)" }
}

// MARK: - Error handling tests (no channel consumption needed)

@Suite("HTTP error handling")
struct HTTPErrorTests {

    let config = LlamaConfiguration(modelName: "test")
    let model = LlamaModel(configuration: LlamaConfiguration(modelName: "test"))

    func makeRequest() -> LanguageModelExecutorGenerationRequest {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        return LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
    }

    @Test("HTTP 404 throws httpError")
    func http404() async throws {
        let transport = MockTransport(statusCode: 404)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        await #expect(throws: LlamaError.httpError(statusCode: 404)) {
            try await executor.respond(
                to: makeRequest(),
                model: model,
                streamingInto: LanguageModelExecutorGenerationChannel()
            )
        }
    }
}

// MARK: - StreamChunk JSON decoding (pure unit tests, no channel)

@Suite("StreamChunk JSON decoding")
struct StreamChunkDecodingTests {

    @Test("decodes timings: predicted_per_second and predicted_per_token_ms")
    func timingsFull() throws {
        let json = MockJSON.finishWithTimingsOnly(promptTokens: 7, completionTokens: 15)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.timings != nil)
        #expect(chunk.timings?.predicted_per_second == 42.5)
        #expect(chunk.timings?.predicted_per_token_ms == 23.5)
        #expect(chunk.usage == nil)  // llama.cpp format: no usage
    }

    @Test("decodes usage when present (OpenAI format)")
    func usagePresent() throws {
        let json = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.usage != nil)
        #expect(chunk.usage?.prompt_tokens == 10)
        #expect(chunk.usage?.completion_tokens == 20)
        #expect(chunk.usage?.completion_tokens_details == nil)
        #expect(chunk.timings != nil)
    }

    @Test("decodes completion_tokens_details.reasoning_tokens when present")
    func completionTokensDetails() throws {
        let json = MockJSON.finishWithUsageAndReasoning(promptTokens: 10, completionTokens: 20, reasoningTokens: 5)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.usage != nil)
        #expect(chunk.usage?.completion_tokens_details != nil)
        #expect(chunk.usage?.completion_tokens_details?.reasoning_tokens == 5)
    }

    @Test("completion_tokens_details defaults to nil when absent")
    func completionTokensDetailsNil() throws {
        let json = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.usage != nil)
        #expect(chunk.usage?.completion_tokens_details == nil)
    }

    @Test("timings optional fields default to nil")
    func timingsOptionalsDefaultNil() throws {
        let json = MockJSON.finish(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.timings != nil)
        #expect(chunk.timings?.prompt_n == 10)
        #expect(chunk.timings?.predicted_n == 20)
        #expect(chunk.timings?.predicted_per_second == 50.0)
        #expect(chunk.timings?.predicted_per_token_ms == nil)
        #expect(chunk.usage == nil)  // llama.cpp format
    }

    @Test("decodes tool_calls in delta")
    func toolCallDelta() throws {
        let json = MockJSON.toolCallBegin(id: "call_1", name: "get_weather", arguments: "{}")
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        let toolCalls = chunk.choices?.first?.delta?.tool_calls
        #expect(toolCalls != nil)
        #expect(toolCalls?.count == 1)
        #expect(toolCalls?.first?.id == "call_1")
        #expect(toolCalls?.first?.function?.name == "get_weather")
        #expect(toolCalls?.first?.function?.arguments == "{}")
        #expect(toolCalls?.first?.type == "function")
    }

    @Test("decodes incremental tool call append")
    func toolCallAppendDelta() throws {
        let json = MockJSON.toolCallAppend(index: 0, arguments: "{\\\"loc")
        let data = json.data(using: .utf8)!
        let decoder = JSONDecoder()
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        let toolCalls = chunk.choices?.first?.delta?.tool_calls
        #expect(toolCalls != nil)
        #expect(toolCalls?.count == 1)
        #expect(toolCalls?.first?.index == 0)
        #expect(toolCalls?.first?.id == nil)  // not present on append
        #expect(toolCalls?.first?.function?.name == nil)  // not present on append
        #expect(toolCalls?.first?.function?.arguments == "{\\\"loc")
    }
}

// MARK: - Integration tests (require concurrent channel consumer)
//
// `LanguageModelExecutorGenerationChannel` must be consumed concurrently with
// `respond()` or it will block. Each test uses `async let` to run the executor
// while draining the channel, mirroring how `LanguageModelSession` uses the
// channel internally.

@Suite("LlamaExecutor integration")
struct LlamaExecutorIntegrationTests {

    let config = LlamaConfiguration(modelName: "test")
    let model = LlamaModel(configuration: LlamaConfiguration(modelName: "test"))

    func makeRequest() -> LanguageModelExecutorGenerationRequest {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        return LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
    }

    // MARK: - Smoke tests (completes without error)

    @Test("basic stream (llama.cpp format: timings only)")
    func validStream() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finish(promptTokens: 5, completionTokens: 3).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let _ = executor.respond(to: makeRequest(), model: model, streamingInto: channel)
        for try await _ in channel { }
    }

    @Test("empty stream (immediate [DONE])")
    func emptyStream() async throws {
        let lines = [MockJSON.done.sseData]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let _ = executor.respond(to: makeRequest(), model: model, streamingInto: channel)
        for try await _ in channel { }
    }

    // MARK: - Reasoning token accumulation tests

    @Test("accumulates reasoning tokens from reasoning_content deltas")
    func reasoningTokensAccumulated() async throws {
        // Reasoning deltas totaling 15 characters -> reasoningTokens should be 15
        let lines = [
            MockJSON.reasoningDelta("Think step ").sseData,
            MockJSON.reasoningDelta("by step...").sseData,
            MockJSON.textDelta("Answer: 42").sseData,
            MockJSON.finish(promptTokens: 10, completionTokens: 20).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usage: LanguageModelExecutorGenerationChannel.Usage?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateUsage(let u) = response.action else { continue }
            usage = u
        }

        try await respond
        #expect(usage != nil)
        #expect(usage?.output.reasoningTokenCount == 15)  // "Think step " (11) + "by step..." (4)
        #expect(usage?.output.totalTokenCount == 20)
        #expect(usage?.input.totalTokenCount == 10)
    }

    @Test("reasoning tokens default to 0 when no reasoning deltas")
    func reasoningTokensDefaultZero() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finish(promptTokens: 10, completionTokens: 20).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usage: LanguageModelExecutorGenerationChannel.Usage?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateUsage(let u) = response.action else { continue }
            usage = u
        }

        try await respond
        #expect(usage != nil)
        #expect(usage?.output.reasoningTokenCount == 0)
    }

    @Test("prefers server-reported reasoning tokens over accumulated count")
    func reasoningTokensPreferredFromServer() async throws {
        // Accumulated: "Think" (5 chars), but server says 10 -> should use 10.
        let lines = [
            MockJSON.reasoningDelta("Think").sseData,
            MockJSON.textDelta("Answer").sseData,
            MockJSON.finishWithUsageAndReasoning(promptTokens: 10, completionTokens: 20, reasoningTokens: 10).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usage: LanguageModelExecutorGenerationChannel.Usage?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateUsage(let u) = response.action else { continue }
            usage = u
        }

        try await respond
        #expect(usage != nil)
        #expect(usage?.output.reasoningTokenCount == 10)  // server-reported overrides accumulated
    }

    // MARK: - Token counts from timings (llama.cpp format)

    @Test("token counts from timings when usage absent (llama.cpp format)")
    func usageFromTimings() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finishWithTimingsOnly(promptTokens: 7, completionTokens: 15).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usage: LanguageModelExecutorGenerationChannel.Usage?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateUsage(let u) = response.action else { continue }
            usage = u
        }

        try await respond
        #expect(usage != nil)
        #expect(usage?.input.totalTokenCount == 7)
        #expect(usage?.output.totalTokenCount == 15)
    }

    // MARK: - Metadata event verification

    @Test("metadata: predicted_per_second from timings")
    func metadataPredictedPerSecond() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finish(promptTokens: 10, completionTokens: 20).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var metadata: [String: any Sendable & Codable & Equatable]?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateMetadata(let m) = response.action else { continue }
            metadata = m.values
        }

        try await respond
        #expect(metadata != nil)
        #expect(metadata?["predicted_per_second"] as? Double == 50.0)
    }

    @Test("metadata: predicted_per_token_ms when present")
    func metadataPredictedPerTokenMs() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finishWithTimingsOnly(promptTokens: 7, completionTokens: 15).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var metadata: [String: any Sendable & Codable & Equatable]?
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateMetadata(let m) = response.action else { continue }
            metadata = m.values
        }

        try await respond
        #expect(metadata != nil)
        #expect(metadata?["predicted_per_second"] as? Double == 42.5)
        #expect(metadata?["predicted_per_token_ms"] as? Double == 23.5)
    }

    @Test("no metadata event when timings have no extra fields")
    func metadataNoneWhenSparse() async throws {
        let sparseFinish = "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":5,\"predicted_n\":3}}"
        let lines = [
            MockJSON.textDelta("Hi").sseData,
            sparseFinish.sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var foundMetadata = false
        for try await event in channel {
            guard let response = event as? LanguageModelExecutorGenerationChannel.Response else { continue }
            guard case .updateMetadata = response.action else { continue }
            foundMetadata = true
        }

        try await respond
        #expect(!foundMetadata)
    }
}

// MARK: - Tool calling tests

@Suite("LlamaExecutor tool calling")
struct LlamaExecutorToolCallingTests {

    let config = LlamaConfiguration(modelName: "test")
    let model = LlamaModel(configuration: LlamaConfiguration(modelName: "test"))

    func makeRequest() -> LanguageModelExecutorGenerationRequest {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "What is the weather in Rome?"))])
        )
        return LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
    }

    // MARK: - Tool call streaming

    @Test("streams incremental tool call arguments as ToolCalls events")
    func incrementalToolCall() async throws {
        // Simulate a tool call stream with incremental arguments.
        // The arguments arrive in fragments: "", then "{\"loc", then "ation\": \"Rome\"}"
        let lines = [
            MockJSON.toolCallBegin(id: "call_123", name: "get_weather", arguments: "").sseData,
            MockJSON.toolCallAppend(index: 0, arguments: "{\\\"loc").sseData,
            MockJSON.toolCallAppend(index: 0, arguments: "ation\\\": \\\"Rome\\\"}").sseData,
            MockJSON.finishWithToolCalls(promptTokens: 15, completionTokens: 10).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var toolCallFragments: [String] = []
        for try await event in channel {
            guard let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls else { continue }
            guard case .toolCall(let tc) = toolCalls.action else { continue }
            guard case .appendArguments(let fragment) = tc.action else { continue }
            toolCallFragments.append(fragment.content)
        }

        try await respond
        #expect(toolCallFragments.count == 3)
        #expect(toolCallFragments[0] == "")
        #expect(toolCallFragments[1] == "{\\\"loc")
        #expect(toolCallFragments[2] == "ation\\\": \\\"Rome\\\"}")
    }

    @Test("emits usage after tool call finish")
    func usageAfterToolCall() async throws {
        let lines = [
            MockJSON.toolCallBegin(id: "call_456", name: "get_weather", arguments: "{}").sseData,
            MockJSON.finishWithToolCalls(promptTokens: 20, completionTokens: 15).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usage: LanguageModelExecutorGenerationChannel.Usage?
        for try await event in channel {
            if let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls {
                guard case .updateUsage(let u) = toolCalls.action else { continue }
                usage = u
            }
            if let response = event as? LanguageModelExecutorGenerationChannel.Response {
                guard case .updateUsage(let u) = response.action else { continue }
                usage = u
            }
        }

        try await respond
        #expect(usage != nil)
        #expect(usage?.input.totalTokenCount == 20)
        #expect(usage?.output.totalTokenCount == 15)
    }

    @Test("parses tool call delta with single-chunk JSON arguments")
    func toolCallSingleChunkArgs() async throws {
        let lines = [
            MockJSON.toolCallBegin(
                id: "call_789",
                name: "search_documents",
                arguments: "{\\\"query\\\":\\\"annual report\\\",\\\"limit\\\":5}"
            ).sseData,
            MockJSON.finishWithToolCalls(promptTokens: 30, completionTokens: 25).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var fragments: [String] = []
        for try await event in channel {
            guard let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls else { continue }
            guard case .toolCall(let tc) = toolCalls.action else { continue }
            guard case .appendArguments(let fragment) = tc.action else { continue }
            fragments.append(fragment.content)
        }

        try await respond
        #expect(fragments.count == 1)
        #expect(fragments[0] == "{\\\"query\\\":\\\"annual report\\\",\\\"limit\\\":5}")
    }

    @Test("tool call stream also emits usage on response channel")
    func usageOnResponseAfterToolCall() async throws {
        let lines = [
            MockJSON.toolCallBegin(id: "call_x1", name: "get_weather", arguments: "{}").sseData,
            MockJSON.finish(promptTokens: 10, completionTokens: 5).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        let channel = LanguageModelExecutorGenerationChannel()

        async let respond: Void = executor.respond(to: makeRequest(), model: model, streamingInto: channel)

        var usageCount = 0
        for try await event in channel {
            if let response = event as? LanguageModelExecutorGenerationChannel.Response {
                guard case .updateUsage = response.action else { continue }
                usageCount += 1
            }
            if let toolCalls = event as? LanguageModelExecutorGenerationChannel.ToolCalls {
                guard case .updateUsage = toolCalls.action else { continue }
                usageCount += 1
            }
        }

        try await respond
        // Should receive one usage event (either via Response or ToolCalls channel)
        #expect(usageCount == 1)
    }
}
