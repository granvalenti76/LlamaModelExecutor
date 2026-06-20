
//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
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
