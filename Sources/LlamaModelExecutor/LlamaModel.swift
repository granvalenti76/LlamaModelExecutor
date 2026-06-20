//
//  LlamaModel.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation
import FoundationModels

/// A LanguageModel that wraps a remote llama.cpp server.
///
/// This model communicates with llama-server via its OpenAI-compatible HTTP API.
/// It supports basic text generation without tool calling, vision, or reasoning.
public struct LlamaModel: LanguageModel, Sendable {

    // MARK: - LanguageModel conformance

    public var capabilities: LanguageModelCapabilities

    public var executorConfiguration: LlamaConfiguration {
        configuration
    }

    /// The executor type that drives inference for this model.
    public typealias Executor = LlamaExecutor

    // MARK: - Internal state

    private let configuration: LlamaConfiguration

    // MARK: - Initializer

    public init(configuration: LlamaConfiguration) {
        self.configuration = configuration
        self.capabilities = LanguageModelCapabilities(capabilities: [])
    }
}
