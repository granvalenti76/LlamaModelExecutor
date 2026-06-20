//
//  LlamaExecutor.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
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

    public func prewarm(model: LlamaModel, transcript: Transcript) {
        // Nothing to prewarm for a remote server.
    }

    // MARK: - respond

    nonisolated(nonsending)
    public func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: LlamaModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // 1. Convert transcript entries into OpenAI-style messages
        let messages = try convertTranscriptEntries(request.transcript)

        // 2. Build the request body
        let temperature = request.generationOptions.temperature ?? configuration.temperature
        let maxTokens = request.generationOptions.maximumResponseTokens ?? configuration.maxTokens
        let modelName = configuration.modelName

        var body: [String: Any] = [
            "model": modelName,
            "messages": messages,
            "stream": true,
            "temperature": temperature,
            "max_tokens": maxTokens,
        ]

        var urlRequest = URLRequest(url: configuration.baseURL.appendingPathComponent("chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        // 3. Perform streaming request
        let (lines, response) = try await transport.lines(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LlamaError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw LlamaError.httpError(statusCode: httpResponse.statusCode)
        }

        // 4. Parse SSE stream
        var promptTokens = 0
        var completionTokens = 0
        var tokensPerSecond: Double?
        var malformedChunkCount = 0
        let maxMalformedChunks = 5

        for try await line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)

            // Skip empty lines
            guard !trimmed.isEmpty else { continue }

            // Only process "data:" lines
            guard trimmed.hasPrefix("data: ") else { continue }
            let payload = String(trimmed.dropFirst(6))

            // The stream is done
            guard payload != "[DONE]" else { break }

            guard let jsonData = payload.data(using: .utf8) else { continue }

            do {
                let chunk = try decoder.decode(StreamChunk.self, from: jsonData)

                if let choice = chunk.choices?.first {
                    // Send reasoning text as reasoning events
                    if let reasoning = choice.delta?.reasoning_content, !reasoning.isEmpty {
                        let action = LanguageModelExecutorGenerationChannel.Reasoning.Action.appendText(
                            reasoning,
                            segmentID: nil,
                            tokenCount: reasoning.count
                        )
                        let event = LanguageModelExecutorGenerationChannel.Reasoning.reasoning(
                            entryID: request.id.uuidString,
                            action: action
                        )
                        await channel.send(event)
                    }

                    // Send content text as response events (skip nulls from role announcement)
                    if let content = choice.delta?.content, !content.isEmpty {
                        let action = LanguageModelExecutorGenerationChannel.Response.Action.appendText(
                            content,
                            segmentID: nil,
                            tokenCount: content.count
                        )
                        let event = LanguageModelExecutorGenerationChannel.Response.response(
                            entryID: request.id.uuidString,
                            action: action
                        )
                        await channel.send(event)
                    }

                    // Capture usage from the standard field or from llama.cpp timings
                    if let usage = chunk.usage {
                        promptTokens = usage.prompt_tokens ?? 0
                        completionTokens = usage.completion_tokens ?? 0
                    } else if let timings = chunk.timings {
                        promptTokens = timings.prompt_n ?? 0
                        completionTokens = timings.predicted_n ?? 0
                        if let predictedPerSecond = timings.predicted_per_second {
                            tokensPerSecond = predictedPerSecond
                        }
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

        // 5. Send final usage update
        let usageInput = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: promptTokens,
            cachedTokenCount: 0
        )
        let usageOutput = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: completionTokens,
            reasoningTokenCount: 0
        )
        let usageAction = LanguageModelExecutorGenerationChannel.Response.Action.updateUsage(
            input: usageInput,
            output: usageOutput
        )
        let usageEvent = LanguageModelExecutorGenerationChannel.Response.response(
            entryID: request.id.uuidString,
            action: usageAction
        )
        await channel.send(usageEvent)
    }

    // MARK: - Private helpers

    /// Convert FoundationModels transcript entries into OpenAI chat message dictionaries.
    private func convertTranscriptEntries(_ transcript: Transcript) throws -> [[String: Any]] {
        var messages: [[String: Any]] = []

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                let text = extractText(from: instructions.segments)
                if !text.isEmpty {
                    messages.append(["role": "system", "content": text])
                }

            case .prompt(let prompt):
                let text = extractText(from: prompt.segments)
                messages.append(["role": "user", "content": text])

            case .response(let response):
                let text = extractText(from: response.segments)
                messages.append(["role": "assistant", "content": text])

            case .toolCalls:
                break
            case .toolOutput:
                break
            case .reasoning:
                break
            @unknown default:
                break
            }
        }

        return messages
    }

    /// Extract plain text from an array of transcript segments.
    private func extractText(from segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let textSegment):
                return textSegment.content
            case .structure(let structuredSegment):
                return String(describing: structuredSegment.content)
            case .attachment:
                return nil
            case .custom:
                return nil
            @unknown default:
                return nil
            }
        }.joined()
    }
}

/// Keys used for metadata sent through ``LanguageModelExecutorGenerationChannel``.
enum LlamaMetadata {
    /// Tokens per second reported by llama.cpp server timings (decoding phase only).
    static let tokensPerSecond = "llama_tokens_per_second"
}
