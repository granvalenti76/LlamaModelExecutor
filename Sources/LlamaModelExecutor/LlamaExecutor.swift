//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation
import FoundationModels

/// The executor that drives inference on a remote llama.cpp server
/// through its OpenAI-compatible HTTP API.
///
/// This executor is the glue between FoundationModels' ``LanguageModelExecutor``
/// protocol and a running llama.cpp server instance.
public struct LlamaExecutor: LanguageModelExecutor {

    public typealias Model = LlamaModel
    public typealias Configuration = LlamaConfiguration

    private let configuration: Configuration
    private let transport: HTTPTransport
    private let decoder: JSONDecoder

    /// Creates an executor with a custom transport.
    /// - Parameter configuration: Connection and generation parameters.
    /// - Parameter transport: The HTTP transport to use. Inject a fake in tests.
    public init(configuration: Configuration, transport: HTTPTransport) throws {
        self.configuration = configuration
        self.transport = transport
        self.decoder = JSONDecoder()
    }

    /// Creates an executor backed by the default ``URLSessionTransport``.
    /// Conformance requirement for ``LanguageModelExecutor``.
    public init(configuration: Configuration) throws {
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = 300
        sessionConfig.timeoutIntervalForResource = 600
        let session = URLSession(configuration: sessionConfig)
        self.configuration = configuration
        self.transport = URLSessionTransport(session: session)
        self.decoder = JSONDecoder()
    }

    // MARK: - prewarm

    /// No-op for remote server.
    public func prewarm(model: LlamaModel, transcript: Transcript) {
        // Nothing to prewarm for a remote server.
    }

    // MARK: - respond

    /// Streams a response from llama-server, translating SSE deltas into channel events.
    /// - Parameter request: The generation request from FoundationModels.
    /// - Parameter model: The model that initiated the request.
    /// - Parameter channel: Event channel for streaming deltas and usage.
    nonisolated(nonsending)
    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: LlamaModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // 1. Build the HTTP request via RequestBuilder
        let built = try RequestBuilder.build(
            from: request,
            modelName: configuration.modelName,
            temperature: configuration.temperature,
            maxTokens: configuration.maxTokens,
            baseURL: configuration.baseURL
        )

        // 2. Perform streaming request
        let (lines, response) = try await transport.lines(for: built.urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LlamaError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw LlamaError.httpError(statusCode: httpResponse.statusCode)
        }

        // 3. Parse SSE stream
        var promptTokens = 0
        var completionTokens = 0
        var malformedChunkCount = 0
        let maxMalformedChunks = 5

        for try await line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)

            guard !trimmed.isEmpty else { continue }
            guard trimmed.hasPrefix("data: ") else { continue }
            let payload = String(trimmed.dropFirst(6))

            guard payload != "[DONE]" else { break }
            guard let jsonData = payload.data(using: .utf8) else { continue }

            do {
                let chunk = try decoder.decode(StreamChunk.self, from: jsonData)

                if let choice = chunk.choices?.first {
                    // Reasoning content
                    if let reasoning = choice.delta?.reasoning_content, !reasoning.isEmpty {
                        let action = LanguageModelExecutorGenerationChannel.Reasoning.Action.appendText(
                            reasoning,
                            segmentID: nil,
                            tokenCount: reasoning.count
                        )
                        await channel.send(
                            LanguageModelExecutorGenerationChannel.Reasoning.reasoning(
                                entryID: request.id.uuidString,
                                action: action
                            )
                        )
                    }

                    // Response text
                    if let content = choice.delta?.content, !content.isEmpty {
                        let action = LanguageModelExecutorGenerationChannel.Response.Action.appendText(
                            content,
                            segmentID: nil,
                            tokenCount: content.count
                        )
                        await channel.send(
                            LanguageModelExecutorGenerationChannel.Response.response(
                                entryID: request.id.uuidString,
                                action: action
                            )
                        )
                    }

                    // Token counts
                    if let usage = chunk.usage {
                        promptTokens = usage.prompt_tokens ?? 0
                        completionTokens = usage.completion_tokens ?? 0
                    } else if let timings = chunk.timings {
                        promptTokens = timings.prompt_n ?? 0
                        completionTokens = timings.predicted_n ?? 0
                    }
                }
            } catch {
                malformedChunkCount += 1
                if malformedChunkCount >= maxMalformedChunks {
                    throw LlamaError.streamError(
                        "\(malformedChunkCount) consecutive malformed SSE chunks — aborting"
                    )
                }
                continue
            }
        }

        // 4. Send final usage update
        await channel.send(
            LanguageModelExecutorGenerationChannel.Response.response(
                entryID: request.id.uuidString,
                action: .updateUsage(
                    input: .init(totalTokenCount: promptTokens, cachedTokenCount: 0),
                    output: .init(totalTokenCount: completionTokens, reasoningTokenCount: 0)
                )
            )
        )
    }
}
