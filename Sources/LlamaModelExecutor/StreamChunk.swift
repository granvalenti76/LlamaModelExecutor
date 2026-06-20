//
//  StreamChunk.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation

/// A streaming chunk from llama.cpp SSE endpoint.
///
/// Every `data:` line in the stream (except `[DONE]`) is decoded into this shape.
/// The same model is reused for every chunk — fields that are absent on the wire
/// simply decode as `nil`.
public struct StreamChunk: Decodable, Sendable {
    /// A single streaming choice.
    public struct Choice: Decodable, Sendable {
        /// The delta payload carried by this choice.
        public struct Delta: Decodable, Sendable {
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
    public struct Usage: Decodable, Sendable {
        public let prompt_tokens: Int?
        public let completion_tokens: Int?
    }

    /// llama.cpp puts detailed timing info here instead of the standard `usage` field
    /// during streaming.
    public struct Timings: Decodable, Sendable {
        /// Number of prompt tokens processed.
        public let prompt_n: Int?
        /// Number of tokens predicted so far.
        public let predicted_n: Int?
        /// Tokens per second during the decoding phase (server-reported).
        public let predicted_per_second: Double?
    }

    public let choices: [Choice]?
    public let usage: Usage?
    public let timings: Timings?
}
