//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation

/// Centralises token-count accounting across the three sources llama.cpp
/// uses to report token counts during streaming:
///
/// 1. **`usage`** (OpenAI-compatible payload, usually on the final chunk)
/// 2. **`timings`** (llama.cpp-specific payload, also on the final chunk)
/// 3. **Character-based estimation** (for reasoning tokens when the server
///    does not report them via `completion_tokens_details.reasoning_tokens`)
///
/// Priority order
/// ==============
/// - Prompt / completion tokens: the **last** source that appears wins
///   (`usage` → `timings` if `usage` is absent in that chunk).
/// - Reasoning tokens: server-reported (`usage.completion_tokens_details`)
///   takes precedence over the accumulated character estimate.
///
/// Usage
/// =====
/// ```swift
/// var tracker = TokenTracker()
///
/// for try await chunk in stream {
///     tracker.update(from: chunk)
///     if let reasoning = chunk.firstChoice?.delta?.reasoning_content {
///         tracker.accountReasoning(delta: reasoning)
///     }
/// }
///
/// let counts = tracker.finalize()
/// // counts.promptTokens, counts.completionTokens, counts.reasoningTokens
/// ```
package struct TokenTracker: Sendable {

    // MARK: - State

    private var promptTokens: Int = 0
    private var completionTokens: Int = 0

    /// Accumulated character count from `reasoning_content` deltas.
    /// Used as fallback when the server does not report `usage`.
    private var reasoningAccumulated: Int = 0

    /// Server-reported reasoning tokens, set when a chunk carries
    /// `usage.completion_tokens_details.reasoning_tokens`.
    private var serverReasoningTokens: Int?

    /// The last‑seen timing metadata. Forwarded as response metadata at the end.
    private var lastTimings: StreamChunk.Timings?

    // MARK: - Update

    /// Incorporates token counts from a decoded ``StreamChunk``.
    ///
    /// - `usage` always updates prompt and completion tokens.
    /// - `timings` updates prompt and completion tokens **only** when `usage`
    ///   is absent in the same chunk (llama.cpp convention).
    /// - Timing metadata (`predicted_per_second`, etc.) is captured regardless
    ///   for later forwarding.
    package mutating func update(from chunk: StreamChunk) {
        // 1. Standard OpenAI usage payload — highest priority.
        if let usage = chunk.usage {
            promptTokens = usage.prompt_tokens ?? 0
            completionTokens = usage.completion_tokens ?? 0
            if let serverReasoning = usage.completion_tokens_details?.reasoning_tokens {
                serverReasoningTokens = serverReasoning
            }
        }

        // 2. llama.cpp timings — only used when the chunk carries no usage.
        if let timings = chunk.timings, chunk.usage == nil {
            promptTokens = timings.prompt_n ?? 0
            completionTokens = timings.predicted_n ?? 0
        }

        // 3. Capture timing metadata regardless of which source was used.
        if let timings = chunk.timings {
            lastTimings = timings
        }
    }

    /// Accounts for a reasoning‑content delta by accumulating its character count.
    ///
    /// This is a fallback estimate used when the server does _not_ report
    /// `completion_tokens_details.reasoning_tokens` in the final `usage` payload.
    package mutating func accountReasoning(delta: String) {
        reasoningAccumulated += delta.count
    }

    // MARK: - Finalize

    /// Produces the final ``TokenCounts`` after the stream has been fully consumed.
    ///
    /// - The reasoning token count prefers the server‑reported value (if present)
    ///   over the character‑based estimate.
    /// - Timing metadata is extracted from the last captured timings.
    package mutating func finalize() -> TokenCounts {
        let resolvedReasoning = serverReasoningTokens ?? reasoningAccumulated

        var metadata: [String: any Sendable & Codable & Equatable] = [:]
        if let timings = lastTimings {
            if let perSecond = timings.predicted_per_second {
                metadata["predicted_per_second"] = perSecond
            }
            if let perTokenMs = timings.predicted_per_token_ms {
                metadata["predicted_per_token_ms"] = perTokenMs
            }
        }

        return TokenCounts(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            reasoningTokens: resolvedReasoning,
            timingMetadata: metadata
        )
    }
}

// MARK: - TokenCounts

/// Immutable snapshot of the final token counts and timing metadata produced
/// by ``TokenTracker/finalize()``.
package struct TokenCounts: Sendable, Equatable {
    /// Number of tokens in the prompt.
    package let promptTokens: Int
    /// Number of tokens generated in the completion.
    package let completionTokens: Int
    /// Number of tokens used for reasoning / chain‑of‑thought.
    package let reasoningTokens: Int
    /// Timing metadata extracted from the last `timings` payload
    /// (e.g. `predicted_per_second`, `predicted_per_token_ms`).
    package let timingMetadata: [String: any Sendable & Codable & Equatable]

    package init(
        promptTokens: Int,
        completionTokens: Int,
        reasoningTokens: Int,
        timingMetadata: [String: any Sendable & Codable & Equatable]
    ) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.reasoningTokens = reasoningTokens
        self.timingMetadata = timingMetadata
    }

    package static func == (lhs: TokenCounts, rhs: TokenCounts) -> Bool {
        lhs.promptTokens == rhs.promptTokens
            && lhs.completionTokens == rhs.completionTokens
            && lhs.reasoningTokens == rhs.reasoningTokens
            && lhs.timingMetadata.keys == rhs.timingMetadata.keys
    }
}
