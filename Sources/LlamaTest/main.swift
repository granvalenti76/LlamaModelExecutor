//
//  main.swift
//  LlamaTest
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation
import FoundationModels
import LlamaModelExecutor

@main
enum Main {
    static func main() async throws {
        let config = LlamaConfiguration(
            modelName: "gemma-4-12b-it-Q4_K_M.gguf",
            temperature: 0.0  // deterministic for testing
        )
        let model = LlamaModel(configuration: config)
        let session = LanguageModelSession(model: model)

        print("⏳ Sending request...")
        let response = try await session.respond(to: "Ciao! Rispondi solo con una parola.")
        print("✅ Response: \(response.content)")
        print("   Usage: \(response.usage)")
    }
}
