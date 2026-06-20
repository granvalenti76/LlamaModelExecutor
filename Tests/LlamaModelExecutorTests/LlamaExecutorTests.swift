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
        // Reasoning deltas totaling 15 characters → reasoningTokens should be 15
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
        // Accumulated: "Think" (5 chars), but server says 10 → should use 10.
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
