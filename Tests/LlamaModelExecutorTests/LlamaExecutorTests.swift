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

    // MARK: - Tool-call deltas (OpenAI SSE format)

    /// First delta for a tool call: carries id, type, function.name, and an empty arguments string.
    static func toolCallStart(index: Int = 0, id: String = "call_1", name: String = "get_weather") -> String {
        """
        {"choices":[{"delta":{"tool_calls":[{"index":\(index),"id":"\(id)","type":"function","function":{"name":"\(name)","arguments":""}}]}}]}
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Subsequent delta: carries only function.arguments (streaming JSON fragment).
    static func toolCallArguments(index: Int = 0, arguments: String) -> String {
        """
        {"choices":[{"delta":{"tool_calls":[{"index":\(index),"function":{"arguments":"\(arguments)"}}]}}]}
        """.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Tool-call finish chunk: finish_reason is "tool_calls".
    static let toolCallFinish = """
    {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"timings":{"prompt_n":10,"predicted_n":5,"predicted_per_second":50.0}}
    """.trimmingCharacters(in: .whitespacesAndNewlines)

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

// MARK: - SSEStreamParser unit tests

@Suite("SSEStreamParser")
struct SSEStreamParserTests {

    let decoder = JSONDecoder()

    /// Helper: creates a stream from raw SSE lines (with or without "data: " prefix).
    private func makeStream(_ lines: [String]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            for line in lines {
                continuation.yield(line)
            }
            continuation.finish()
        }
    }

    @Test("yields decoded chunks from valid SSE lines")
    func validSSELines() async throws {
        let raw = [
            "data: \(MockJSON.textDelta("Hello"))",
            "data: \(MockJSON.textDelta(" World"))",
            "data: \(MockJSON.done)",
        ]
        let parser = SSEStreamParser()
        var chunks: [StreamChunk] = []

        for try await chunk in parser.parse(makeStream(raw)) {
            chunks.append(chunk)
        }

        #expect(chunks.count == 2)
        #expect(chunks[0].choices?.first?.delta?.content == "Hello")
        #expect(chunks[1].choices?.first?.delta?.content == " World")
    }

    @Test("skips empty lines and non-data lines")
    func skipsNoise() async throws {
        let raw = [
            "",
            "data: \(MockJSON.textDelta("A"))",
            ": heartbeat",
            "data: \(MockJSON.textDelta("B"))",
            "random text",
            "data: \(MockJSON.done)",
        ]
        let parser = SSEStreamParser()
        var chunks: [StreamChunk] = []

        for try await chunk in parser.parse(makeStream(raw)) {
            chunks.append(chunk)
        }

        #expect(chunks.count == 2)
        #expect(chunks[0].choices?.first?.delta?.content == "A")
        #expect(chunks[1].choices?.first?.delta?.content == "B")
    }

    @Test("terminates early on [DONE] and ignores subsequent lines")
    func doneTerminates() async throws {
        let raw = [
            "data: \(MockJSON.textDelta("First"))",
            "data: \(MockJSON.done)",
            "data: \(MockJSON.textDelta("Ignored"))",  // after [DONE]
        ]
        let parser = SSEStreamParser()
        var chunks: [StreamChunk] = []

        for try await chunk in parser.parse(makeStream(raw)) {
            chunks.append(chunk)
        }

        #expect(chunks.count == 1)
        #expect(chunks[0].choices?.first?.delta?.content == "First")
    }

    @Test("throws after exceeding malformed chunk threshold")
    func malformedThreshold() async throws {
        let raw = [
            "data: not-json",
            "data: also-not-json",
            "data: still-not-json",
            "data: nope",
            "data: invalid",
        ]
        let parser = SSEStreamParser(maxMalformedChunks: 3)

        await #expect(throws: LlamaError.streamError("3 consecutive malformed SSE chunks — aborting")) {
            for try await _ in parser.parse(makeStream(raw)) { }
        }
    }

    @Test("resets malformed counter after a valid chunk")
    func malformedCounterResets() async throws {
        let raw = [
            "data: bad-json",
            "data: \(MockJSON.textDelta("Valid"))",
            "data: bad-again",
            "data: also-bad",
            "data: still-bad",
        ]
        let parser = SSEStreamParser(maxMalformedChunks: 3)
        var chunks: [StreamChunk] = []

        // The counter resets after the valid chunk, so we need 3 consecutive
        // malformed chunks after that to trigger the throw.
        // "bad-again", "also-bad", "still-bad" = 3 → throw.
        await #expect(throws: LlamaError.streamError("3 consecutive malformed SSE chunks — aborting")) {
            for try await chunk in parser.parse(makeStream(raw)) {
                chunks.append(chunk)
            }
        }

        #expect(chunks.count == 1)
    }

    @Test("empty stream yields no chunks")
    func emptyStream() async throws {
        let parser = SSEStreamParser()
        var count = 0

        for try await _ in parser.parse(makeStream([])) {
            count += 1
        }

        #expect(count == 0)
    }
}

// MARK: - TokenTracker unit tests

@Suite("TokenTracker")
struct TokenTrackerTests {

    @Test("starts with zero counts")
    func initialValues() {
        var tracker = TokenTracker()
        let counts = tracker.finalize()

        #expect(counts.promptTokens == 0)
        #expect(counts.completionTokens == 0)
        #expect(counts.reasoningTokens == 0)
        #expect(counts.timingMetadata.isEmpty)
    }

    @Test("usage overrides prompt and completion tokens")
    func usageUpdatesTokens() {
        let json = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        #expect(counts.promptTokens == 10)
        #expect(counts.completionTokens == 20)
    }

    @Test("timings updates tokens when usage absent")
    func timingsFallback() {
        let json = MockJSON.finishWithTimingsOnly(promptTokens: 7, completionTokens: 15)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        #expect(counts.promptTokens == 7)
        #expect(counts.completionTokens == 15)
    }

    @Test("usage takes priority over timings in same chunk")
    func usagePriorityInSameChunk() {
        // finishWithUsage has both usage and timings.
        let json = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        // usage has prompt_tokens=10, timings has prompt_n=10 (same value here)
        // The key is that usage was read, not timings.
        #expect(counts.promptTokens == 10)
        #expect(counts.completionTokens == 20)
    }

    @Test("usage overrides previous timings when it arrives later")
    func usageOverridesPreviousTimings() {
        let timingsJson = MockJSON.finishWithTimingsOnly(promptTokens: 5, completionTokens: 8)
        let timingsData = timingsJson.data(using: .utf8)!
        let timingsChunk = try! JSONDecoder().decode(StreamChunk.self, from: timingsData)

        let usageJson = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let usageData = usageJson.data(using: .utf8)!
        let usageChunk = try! JSONDecoder().decode(StreamChunk.self, from: usageData)

        var tracker = TokenTracker()
        tracker.update(from: timingsChunk)  // first: timings
        tracker.update(from: usageChunk)    // then: usage overrides

        let counts = tracker.finalize()
        #expect(counts.promptTokens == 10)   // from usage
        #expect(counts.completionTokens == 20) // from usage
    }

    @Test("reasoning: character accumulation fallback")
    func reasoningAccumulated() {
        var tracker = TokenTracker()
        tracker.accountReasoning(delta: "Think")
        tracker.accountReasoning(delta: " step by step")

        let counts = tracker.finalize()
        #expect(counts.reasoningTokens == 18)  // "Think" (5) + " step by step" (13) = 18
    }

    @Test("reasoning: server-reported takes priority over accumulated")
    func reasoningServerPriority() {
        let json = MockJSON.finishWithUsageAndReasoning(promptTokens: 10, completionTokens: 20, reasoningTokens: 8)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.accountReasoning(delta: "Some long reasoning text that would be many tokens")  // 49 chars
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        #expect(counts.reasoningTokens == 8)  // server-reported, not 49
    }

    @Test("metadata extracted from timings")
    func timingMetadata() {
        let json = MockJSON.finishWithTimingsOnly(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        #expect(counts.timingMetadata["predicted_per_second"] as? Double == 42.5)
        #expect(counts.timingMetadata["predicted_per_token_ms"] as? Double == 23.5)
    }

    @Test("metadata only contains predicted_per_second when timings lack per_token_ms")
    func metadataOnlyPerSecond() throws {
        // MockJSON.finish() includes predicted_per_second:50.0 but no predicted_per_token_ms.
        let json = MockJSON.finish(promptTokens: 5, completionTokens: 3)
        let data = json.data(using: .utf8)!
        let chunk = try JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let counts = tracker.finalize()
        #expect(counts.timingMetadata.count == 1)
        #expect(counts.timingMetadata["predicted_per_second"] as? Double == 50.0)
        #expect(counts.timingMetadata["predicted_per_token_ms"] == nil)
    }

    @Test("finalize is idempotent")
    func finalizeIdempotent() {
        let json = MockJSON.finishWithUsage(promptTokens: 10, completionTokens: 20)
        let data = json.data(using: .utf8)!
        let chunk = try! JSONDecoder().decode(StreamChunk.self, from: data)

        var tracker = TokenTracker()
        tracker.update(from: chunk)

        let first = tracker.finalize()
        let second = tracker.finalize()

        #expect(first.promptTokens == second.promptTokens)
        #expect(first.completionTokens == second.completionTokens)
        #expect(first.reasoningTokens == second.reasoningTokens)
    }
}

// MARK: - Tool-call delta decoding tests

@Suite("StreamChunk tool-call decoding")
struct StreamChunkToolCallDecodingTests {

    let decoder = JSONDecoder()

    @Test("decodes tool_calls delta with id, type, function.name")
    func toolCallStart() throws {
        let json = MockJSON.toolCallStart(index: 0, id: "call_42", name: "get_weather")
        let data = json.data(using: .utf8)!
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        let delta = try #require(chunk.choices?.first?.delta)
        let calls = try #require(delta.tool_calls)
        #expect(calls.count == 1)
        #expect(calls[0].index == 0)
        #expect(calls[0].id == "call_42")
        #expect(calls[0].type == "function")
        let fn = try #require(calls[0].function)
        #expect(fn.name == "get_weather")
        #expect(fn.arguments == "")
    }

    @Test("decodes tool_calls delta with only arguments")
    func toolCallArguments() throws {
        // Use a plain text fragment that does not require JSON-in-JSON escaping.
        let json = MockJSON.toolCallArguments(index: 0, arguments: "latitude")
        let data = json.data(using: .utf8)!
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        let delta = try #require(chunk.choices?.first?.delta)
        let calls = try #require(delta.tool_calls)
        #expect(calls.count == 1)
        #expect(calls[0].index == 0)
        #expect(calls[0].id == nil)           // only on first delta
        #expect(calls[0].function?.name == nil)  // only on first delta
        #expect(calls[0].function?.arguments == "latitude")
    }

    @Test("decodes finish_reason tool_calls")
    func toolCallFinish() throws {
        let json = MockJSON.toolCallFinish
        let data = json.data(using: .utf8)!
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        #expect(chunk.choices?.first?.finish_reason == "tool_calls")
        #expect(chunk.timings?.predicted_per_second == 50.0)
    }

    @Test("tool_calls defaults to nil when absent")
    func toolCallsNilWhenAbsent() throws {
        let json = MockJSON.textDelta("Hello")
        let data = json.data(using: .utf8)!
        let chunk = try decoder.decode(StreamChunk.self, from: data)

        let delta = try #require(chunk.choices?.first?.delta)
        #expect(delta.tool_calls == nil)
        #expect(delta.content == "Hello")
    }
}

// MARK: - RequestBuilder tool-call tests

@Suite("RequestBuilder tool handling")
struct RequestBuilderToolTests {

    let config = LlamaConfiguration(modelName: "test")
    let baseURL = URL(string: "http://127.0.0.1:8080/v1")!

    @Test("build does not include tools when enabledToolDefinitions is empty")
    func noToolsWhenEmpty() throws {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let built = try RequestBuilder.build(
            from: request,
            modelName: "test-model",
            temperature: 0.7,
            maxTokens: 100,
            baseURL: baseURL
        )

        let body = try JSONSerialization.jsonObject(with: built.urlRequest.httpBody!) as! [String: Any]
        #expect(body["tools"] == nil)
        #expect(body["tool_choice"] == nil)
    }

    @Test("build includes tools when enabledToolDefinitions is provided")
    func toolsPresent() throws {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        let schema = GenerationSchema(
            type: String.self,
            description: "A city name",
            properties: []
        )
        let toolDef = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get the weather for a city",
            parameters: schema
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [toolDef],
            generationOptions: GenerationOptions(),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let built = try RequestBuilder.build(
            from: request,
            modelName: "test-model",
            temperature: 0.7,
            maxTokens: 100,
            baseURL: baseURL
        )

        let body = try JSONSerialization.jsonObject(with: built.urlRequest.httpBody!) as! [String: Any]
        let tools = try #require(body["tools"] as? [[String: Any]])
        #expect(tools.count == 1)
        #expect(tools[0]["type"] as? String == "function")
        let fn = try #require(tools[0]["function"] as? [String: Any])
        #expect(fn["name"] as? String == "get_weather")
        #expect(fn["description"] as? String == "Get the weather for a city")
        #expect(fn["parameters"] is [String: Any])
    }

    @Test("tool_choice is none when toolCallingMode is disallowed")
    func toolChoiceDisallowed() throws {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        let schema = GenerationSchema(type: String.self, description: "", properties: [])
        let toolDef = Transcript.ToolDefinition(
            name: "test", description: "", parameters: schema
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [toolDef],
            generationOptions: GenerationOptions(
                samplingMode: nil,
                temperature: nil,
                maximumResponseTokens: nil,
                toolCallingMode: .disallowed
            ),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let built = try RequestBuilder.build(
            from: request,
            modelName: "test",
            temperature: 0.7,
            maxTokens: 100,
            baseURL: baseURL
        )

        let body = try JSONSerialization.jsonObject(with: built.urlRequest.httpBody!) as! [String: Any]
        #expect(body["tool_choice"] as? String == "none")
    }

    @Test("tool_choice is required when toolCallingMode is required")
    func toolChoiceRequired() throws {
        let entry = Transcript.Entry.prompt(
            Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: "Hi"))])
        )
        let schema = GenerationSchema(type: String.self, description: "", properties: [])
        let toolDef = Transcript.ToolDefinition(
            name: "test", description: "", parameters: schema
        )
        let request = LanguageModelExecutorGenerationRequest(
            id: UUID(),
            transcript: Transcript(entries: [entry]),
            enabledTools: [toolDef],
            generationOptions: GenerationOptions(
                samplingMode: nil,
                temperature: nil,
                maximumResponseTokens: nil,
                toolCallingMode: .required
            ),
            contextOptions: ContextOptions(),
            metadata: [:]
        )
        let built = try RequestBuilder.build(
            from: request,
            modelName: "test",
            temperature: 0.7,
            maxTokens: 100,
            baseURL: baseURL
        )

        let body = try JSONSerialization.jsonObject(with: built.urlRequest.httpBody!) as! [String: Any]
        #expect(body["tool_choice"] as? String == "required")
    }
}
