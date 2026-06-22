//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation
import FoundationModels

/// Thin abstraction over ``LanguageModelExecutorGenerationChannel`` that
/// eliminates the repetitive boilerplate of constructing event objects with
/// the same `entryID` on every call.
///
/// Usage
/// =====
/// ```swift
/// let forwarder = ChannelForwarder(channel: channel, entryID: request.id.uuidString)
///
/// await forwarder.sendResponse(text: "Hello", tokenCount: 5)
/// await forwarder.sendReasoning(text: "thinking…", tokenCount: 12)
/// await forwarder.sendMetadata(["predicted_per_second": 42.5])
/// await forwarder.sendFinalUsage(prompt: 10, completion: 20, reasoning: 3)
/// ```
package struct ChannelForwarder: Sendable {

    private let channel: LanguageModelExecutorGenerationChannel
    private let entryID: String

    /// Creates a forwarder bound to a specific channel and entry.
    /// - Parameter channel: The generation channel to send events to.
    /// - Parameter entryID: The `id.uuidString` of the generation request.
    package init(
        channel: LanguageModelExecutorGenerationChannel,
        entryID: String
    ) {
        self.channel = channel
        self.entryID = entryID
    }

    // MARK: - Reasoning delta

    /// Forwards a reasoning‑content delta.
    /// - Parameters:
    ///   - text: The reasoning text fragment.
    ///   - tokenCount: Estimated token count (character count fallback).
    package func sendReasoning(text: String, tokenCount: Int) async {
        let action = LanguageModelExecutorGenerationChannel.Reasoning.Action.appendText(
            text,
            segmentID: nil,
            tokenCount: tokenCount
        )
        await channel.send(
            LanguageModelExecutorGenerationChannel.Reasoning.reasoning(
                entryID: entryID,
                action: action
            )
        )
    }

    // MARK: - Response delta

    /// Forwards a response‑text delta.
    /// - Parameters:
    ///   - text: The generated text fragment.
    ///   - tokenCount: Estimated token count (character count fallback).
    package func sendResponse(text: String, tokenCount: Int) async {
        let action = LanguageModelExecutorGenerationChannel.Response.Action.appendText(
            text,
            segmentID: nil,
            tokenCount: tokenCount
        )
        await channel.send(
            LanguageModelExecutorGenerationChannel.Response.response(
                entryID: entryID,
                action: action
            )
        )
    }

    // MARK: - Metadata

    /// Forwards timing metadata (e.g. `predicted_per_second`).
    /// - Parameter metadata: Key‑value pairs to attach to the response.
    package func sendMetadata(
        _ metadata: [String: any Sendable & Codable & Equatable]
    ) async {
        await channel.send(
            LanguageModelExecutorGenerationChannel.Response.response(
                entryID: entryID,
                action: .updateMetadata(metadata)
            )
        )
    }

    // MARK: - Final usage

    /// Forwards the final token‑usage summary.
    /// - Parameters:
    ///   - promptTokens: Number of prompt tokens consumed.
    ///   - completionTokens: Number of completion tokens generated.
    ///   - reasoningTokens: Number of tokens used for reasoning.
    package func sendFinalUsage(
        promptTokens: Int,
        completionTokens: Int,
        reasoningTokens: Int
    ) async {
        let input = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: promptTokens,
            cachedTokenCount: 0
        )
        let output = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: completionTokens,
            reasoningTokenCount: reasoningTokens
        )
        await channel.send(
            LanguageModelExecutorGenerationChannel.Response.response(
                entryID: entryID,
                action: .updateUsage(input: input, output: output)
            )
        )
    }
}
