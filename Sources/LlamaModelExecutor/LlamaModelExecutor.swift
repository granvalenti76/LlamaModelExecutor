//
//  LlamaModelExecutor.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation
import FoundationModels

// Re-export the main types for convenient access.
@_exported import FoundationModels

/// LlamaModelExecutor provides a FoundationModels-compatible executor
/// that drives inference through a remote llama.cpp server via its
/// OpenAI-compatible HTTP API.
///
/// Usage:
/// ```swift
/// let config = LlamaConfiguration(modelName: "gemma-4-12b-it-Q4_K_M.gguf")
/// let model = LlamaModel(configuration: config)
/// let session = LanguageModelSession(model: model)
///
/// let response = try await session.respond(to: "Hello!")
/// print(response.content)
/// ```
public enum LlamaModelExecutor {

}
