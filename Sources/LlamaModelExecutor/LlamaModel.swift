
//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation
import FoundationModels

/// A LanguageModel that wraps a remote llama.cpp server.
///
/// This model communicates with llama-server via its OpenAI-compatible HTTP API.
/// It supports text generation, streaming, and tool/function calling.
public struct LlamaModel: LanguageModel, Sendable {

    // MARK: - LanguageModel conformance

    /// Capabilities advertised by this model.
    public var capabilities: LanguageModelCapabilities

    /// The configuration used to create this model, surfaced as required by ``LanguageModel``.
    public var executorConfiguration: LlamaConfiguration {
        configuration
    }

    /// The executor type that drives inference for this model.
    public typealias Executor = LlamaExecutor

    // MARK: - Internal state

    private let configuration: LlamaConfiguration

    // MARK: - Initializer

    /// Creates a model that communicates with a llama.cpp server.
    /// - Parameter configuration: Connection and generation parameters.
    public init(configuration: LlamaConfiguration) {
        self.configuration = configuration
        self.capabilities = LanguageModelCapabilities([.toolCalling])
    }
}
