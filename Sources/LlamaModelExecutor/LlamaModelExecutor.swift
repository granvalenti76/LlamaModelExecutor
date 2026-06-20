// The Swift Programming Language
// https://docs.swift.org/swift-book
import Foundation
import FoundationModels


/// An executor responsible for managing and running inference via llama-server
/// The executor conform to LanguageModelExecutor Protocol and uses a 'LlamaConfiguration'
/// To define its operational parameters.
/// It handles the lifecycle of the model, including prewarming and processing generation requestes.

struct LlamaModelExecutor: LanguageModelExecutor {
    typealias Configuration = LlamaConfiguration
    private let configuration: Configuration
    
    init(configuration: Configuration) throws {
        self.prewarm(model: configuration.model, transcript: configuration.transcript)
    }
}
