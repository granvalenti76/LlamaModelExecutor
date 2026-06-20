
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

enum MockJSON {
    static func textDelta(_ text: String) -> String {
        "{\"choices\":[{\"delta\":{\"content\":\"\(text)\"}}]}"
    }

    static func reasoningDelta(_ text: String) -> String {
        "{\"choices\":[{\"delta\":{\"reasoning_content\":\"\(text)\"}}]}"
    }

    static let roleAnnouncement = "{\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}"

    static func finish(promptTokens: Int = 10, completionTokens: Int = 20) -> String {
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":\(promptTokens),\"completion_tokens\":\(completionTokens)},\"timings\":{\"prompt_n\":\(promptTokens),\"predicted_n\":\(completionTokens),\"predicted_per_second\":50.0}}"
    }

    static let done = "[DONE]"
}

extension String {
    var sseData: String { "data: \(self)" }
}

@Suite("MockTransport")
struct MockTransportTests {

    let config = LlamaConfiguration(modelName: "test")

    @Test("create executor")
    func createExecutor() throws {
        let transport = MockTransport()
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        #expect(true)
    }
}

@Suite("LlamaExecutor HTTP errors")
struct LlamaExecutorHTTPErrorTests {

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

    @Test("HTTP 404")
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

    @Test("valid stream")
    func validStream() async throws {
        let lines = [
            MockJSON.textDelta("Hello").sseData,
            MockJSON.finish(promptTokens: 5, completionTokens: 3).sseData,
            MockJSON.done.sseData,
        ]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        try await executor.respond(
            to: makeRequest(),
            model: model,
            streamingInto: LanguageModelExecutorGenerationChannel()
        )
    }

    @Test("empty stream")
    func emptyStream() async throws {
        let lines = [MockJSON.done.sseData]
        let transport = MockTransport(lines: lines)
        let executor = try LlamaExecutor(configuration: config, transport: transport)
        try await executor.respond(
            to: makeRequest(),
            model: model,
            streamingInto: LanguageModelExecutorGenerationChannel()
        )
    }
}
