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

    /// Creates an executor with a custom transport.
    /// - Parameter configuration: Connection and generation parameters.
    /// - Parameter transport: The HTTP transport to use. Inject a fake in tests.
    public init(configuration: Configuration, transport: HTTPTransport) throws {
        self.configuration = configuration
        self.transport = transport
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
        // 1. Resolve effective temperature from sampling mode.
        //    Greedy sampling is approximated by forcing temperature to 0.
        let effectiveTemperature: Double
        if request.generationOptions.samplingMode == .greedy {
            effectiveTemperature = 0
        } else {
            effectiveTemperature = configuration.temperature
        }

        // 2. Build the HTTP request via RequestBuilder.
        let built = try RequestBuilder.build(
            from: request,
            modelName: configuration.modelName,
            temperature: effectiveTemperature,
            maxTokens: configuration.maxTokens,
            baseURL: configuration.baseURL
        )

        // 3. Execute the streaming request.
        let (lines, response) = try await transport.lines(for: built.urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LlamaError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            throw LlamaError.httpError(statusCode: httpResponse.statusCode)
        }

        // 4. Parse the SSE stream, tracking tokens and forwarding events.
        let parser = SSEStreamParser()
        var tracker = TokenTracker()
        let forwarder = ChannelForwarder(channel: channel, entryID: request.id.uuidString)

        // Track active tool calls by stream index so we can correlate
        // arguments fragments with their id/name (sent only on the first delta).
        var activeToolCalls: [Int: (id: String, name: String)] = [:]

        for try await chunk in parser.parse(lines) {
            tracker.update(from: chunk)

            if let choice = chunk.choices?.first {
                // Reasoning-content delta.
                if let reasoning = choice.delta?.reasoning_content, !reasoning.isEmpty {
                    tracker.accountReasoning(delta: reasoning)
                    await forwarder.sendReasoning(text: reasoning, tokenCount: reasoning.count)
                }

                // Response-text delta.
                if let content = choice.delta?.content, !content.isEmpty {
                    await forwarder.sendResponse(text: content, tokenCount: content.count)
                }

                // Tool-call delta — the model is requesting a tool invocation.
                if let toolCallDeltas = choice.delta?.tool_calls {
                    for delta in toolCallDeltas {
                        // Correlate id/name from the first delta or reuse tracked values.
                        let callID = delta.id ?? activeToolCalls[delta.index]?.id ?? ""
                        let callName = delta.function?.name ?? activeToolCalls[delta.index]?.name ?? ""

                        if activeToolCalls[delta.index] == nil {
                            activeToolCalls[delta.index] = (id: callID, name: callName)
                        }

                        if let args = delta.function?.arguments, !args.isEmpty {
                            await forwarder.sendToolCall(
                                id: callID,
                                name: callName,
                                fragment: args,
                                tokenCount: args.count
                            )
                        }
                    }
                }
            }
        }

        // 5. Forward timing metadata and final usage.
        let counts = tracker.finalize()
        if !counts.timingMetadata.isEmpty {
            await forwarder.sendMetadata(counts.timingMetadata)
        }
        await forwarder.sendFinalUsage(
            promptTokens: counts.promptTokens,
            completionTokens: counts.completionTokens,
            reasoningTokens: counts.reasoningTokens
        )
    }
}
