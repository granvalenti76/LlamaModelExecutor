
//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation

/// Configuration for the LlamaModelExecutor.
///
/// Controls the connection to a llama.cpp server and generation parameters.
public struct LlamaConfiguration: Hashable, Sendable, Codable {
    /// The model identifier sent to llama-server in the request body.
    public let modelName: String

    /// Sampling temperature. Higher values produce more random outputs.
    public let temperature: Double

    /// Maximum number of tokens to generate per response.
    public let maxTokens: Int

    /// Base URL of llama-server's OpenAI-compatible API endpoint.
    /// Defaults to `http://127.0.0.1:8080/v1`.
    public let baseURL: URL

    /// Default base URL pointing to a local llama-server instance.
    public static let defaultURL = URL(string: "http://127.0.0.1:8080/v1")!

    /// Creates a new configuration.
    ///
    /// - Parameters:
    ///   - modelName: The model identifier sent to llama-server.
    ///   - temperature: Sampling temperature (default: 0.7).
    ///   - maxTokens: Maximum tokens to generate (default: 32000).
    ///   - baseURL: Base URL of llama-server's API (default: localhost:8080/v1).
    public init(
        modelName: String,
        temperature: Double = 0.7,
        maxTokens: Int = 32000,
        baseURL: URL = defaultURL
    ) {
        self.modelName = modelName
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.baseURL = baseURL
    }
}
