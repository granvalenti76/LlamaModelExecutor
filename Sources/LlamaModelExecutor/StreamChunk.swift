//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation

/// A streaming chunk from llama.cpp SSE endpoint.
///
/// Every `data:` line in the stream (except `[DONE]`) is decoded into this shape.
/// The same model is reused for every chunk — fields that are absent on the wire
/// simply decode as `nil`.
package struct StreamChunk: Decodable, Sendable {
    /// A single streaming choice.
    package struct Choice: Decodable, Sendable {
        /// The delta payload carried by this choice.
        package struct Delta: Decodable, Sendable {
            /// Role announcement (present only on the very first chunk).
            public let role: String?
            /// Content delta emitted during normal text generation.
            public let content: String?
            /// Reasoning-content delta emitted by reasoning-capable models (e.g. Gemma).
            public let reasoning_content: String?
            /// Tool call deltas, each describing one function call being streamed.
            /// Present only when the model decides to invoke a tool.
            public let tool_calls: [ToolCallDelta]?
        }
        public let delta: Delta?
        public let finish_reason: String?
    }

    /// A single tool call delta within an SSE chunk.
    ///
    /// On the first chunk for a given `index` the server sends `id` and
    /// `function.name`. Subsequent chunks for the same `index` only contain
    /// `function.arguments` which is streamed incrementally.
    package struct ToolCallDelta: Decodable, Sendable {
        /// The index of this tool call among multiple parallel calls.
        public let index: Int?
        /// The unique identifier for this tool call (present only on the first chunk).
        public let id: String?
        /// The type of tool call; always "function" for OpenAI-compatible APIs.
        public let type: String?
        /// The function details.
        public let function: FunctionDelta?

        package struct FunctionDelta: Decodable, Sendable {
            /// The name of the function being called (present only on the first chunk).
            public let name: String?
            /// A JSON fragment of the function arguments, streamed incrementally.
            public let arguments: String?
        }
    }

    /// Standard OpenAI usage payload (usually present on the last streaming chunk,
    /// or on the non-streaming response).
    package struct Usage: Decodable, Sendable {
        /// Number of tokens in the prompt.
        public let prompt_tokens: Int?
        /// Number of tokens generated in the completion.
        public let completion_tokens: Int?
        /// Breakdown of completion tokens reported by the model.
        /// Populated by models that distinguish reasoning from output tokens (e.g. Gemma).
        public let completion_tokens_details: CompletionTokensDetails?

        /// Breakdown of completion token counts.
        package struct CompletionTokensDetails: Decodable, Sendable {
            /// Number of tokens used for reasoning / chain-of-thought.
            public let reasoning_tokens: Int?
        }
    }

    /// llama.cpp puts detailed timing info here instead of the standard `usage` field
    /// during streaming. These values are forwarded as response metadata.
    package struct Timings: Decodable, Sendable {
        /// Number of prompt tokens processed.
        package let prompt_n: Int?
        /// Number of tokens predicted so far.
        package let predicted_n: Int?
        /// Prediction throughput in tokens per second.
        package let predicted_per_second: Double?
        /// Average time per predicted token in milliseconds.
        package let predicted_per_token_ms: Double?
    }

    public let choices: [Choice]?
    public let usage: Usage?
    public let timings: Timings?
}
