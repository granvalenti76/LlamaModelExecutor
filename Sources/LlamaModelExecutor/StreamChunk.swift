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
            /// Tool-call deltas emitted when the model invokes a tool.
            /// Each element is a fragment of a tool call that gets streamed
            /// across multiple chunks (id/name on the first, arguments in subsequent).
            public let tool_calls: [ToolCallDelta]?
        }
        public let delta: Delta?
        /// The reason the model stopped generating.
        /// When `"tool_calls"`, the model is requesting a tool invocation.
        public let finish_reason: String?
    }

    /// A fragment of a streaming tool-call delta in OpenAI SSE format.
    ///
    /// The first chunk for a tool call carries `id` and `function.name`;
    /// subsequent chunks append `function.arguments` as a JSON fragment.
    package struct ToolCallDelta: Decodable, Sendable {
        /// The index of this tool call (used to correlate fragments across chunks).
        public let index: Int
        /// The unique identifier for this tool call (present only on the first delta).
        public let id: String?
        /// The type of tool (always `"function"` for OpenAI-compatible APIs).
        public let type: String?
        /// The function details: name on first delta, arguments as streaming JSON.
        public let function: FunctionDelta?

        /// Function-specific fields in a tool-call delta.
        package struct FunctionDelta: Decodable, Sendable {
            /// The name of the function to call (present only on the first delta).
            public let name: String?
            /// A streaming fragment of the JSON arguments.
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
