
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
        }
        public let delta: Delta?
        public let finish_reason: String?
    }

    /// Standard OpenAI usage payload (usually present on the last streaming chunk,
    /// or on the non-streaming response).
    package struct Usage: Decodable, Sendable {
        public let prompt_tokens: Int?
        public let completion_tokens: Int?
    }

    /// llama.cpp puts detailed timing info here instead of the standard `usage` field
    /// during streaming.
    package struct Timings: Decodable, Sendable {
        /// Number of prompt tokens processed.
        package let prompt_n: Int?
        /// Number of tokens predicted so far.
        package let predicted_n: Int?
    }

    public let choices: [Choice]?
    public let usage: Usage?
    public let timings: Timings?
}
