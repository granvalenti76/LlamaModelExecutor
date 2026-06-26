//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation
import FoundationModels
import LlamaModelExecutor

// MARK: - Example tool: get_weather

/// Arguments for the get_weather tool.
/// The @Generable macro automatically provides the JSON schema and
/// conversion from generated content.
@Generable
struct GetWeatherArguments {
    /// The city and country to get weather for (e.g. "Rome, Italy")
    var location: String
}

/// A tool that returns the current weather for a location.
/// This is a mock — it returns a canned response.
struct GetWeatherTool: Tool {
    typealias Arguments = GetWeatherArguments
    typealias Output = String

    var name: String { "get_weather" }
    var description: String { "Get the current weather for a location" }
    var includesSchemaInInstructions: Bool { false }

    func call(arguments: GetWeatherArguments) async throws -> String {
        "The weather in \(arguments.location) is sunny, 22°C with light breeze."
    }
}

// MARK: - Example tool: search_documents

@Generable
struct SearchDocumentsArguments {
    /// The search query
    var query: String
    /// Maximum number of results to return
    var limit: Int
}

struct SearchDocumentsTool: Tool {
    typealias Arguments = SearchDocumentsArguments
    typealias Output = String

    var name: String { "search_documents" }
    var description: String { "Search internal documents for information" }
    var includesSchemaInInstructions: Bool { false }

    func call(arguments: SearchDocumentsArguments) async throws -> String {
        "[Mock] Found \(arguments.limit) results for \"\(arguments.query)\": annual_report_2025.pdf, summary_q4.pdf"
    }
}

// MARK: - Main

@main
enum Main {
    static func main() async throws {
        let config = LlamaConfiguration(
            modelName: "gemma-4-12b-it-Q4_K_M.gguf",
            temperature: 0.0
        )
        let model = LlamaModel(configuration: config)

        print("""
        ╔══════════════════════════════════════════════════╗
        ║   LlamaModelExecutor — Tool Calling Demo        ║
        ╚══════════════════════════════════════════════════╝
        """)

        // ── 1) Text-only request ──
        let session1 = LanguageModelSession(model: model)
        print("\n⏳ [Text only] Sending...")
        let textResponse = try await session1.respond(to: "Ciao! Rispondi solo con una parola.")
        print("✅ [Text only] Response: \(textResponse.content)")
        print("   Usage: \(textResponse.usage)")

        // ── 2) Request with tools ──
        let session2 = LanguageModelSession(
            model: model,
            tools: [GetWeatherTool(), SearchDocumentsTool()],
            instructions: "You are a helpful assistant with access to tools."
        )
        print("\n⏳ [With tools] Sending 'What is the weather in Rome?'...")
        let toolResponse = try await session2.respond(
            to: "What is the weather in Rome, Italy?"
        )
        print("✅ [With tools] Response: \(toolResponse.content)")
        print("   Usage: \(toolResponse.usage)")

        // ── 3) Forced tool call (tool_choice = .required) ──
        let session3 = LanguageModelSession(model: model, tools: [GetWeatherTool()])
        print("\n⏳ [Required tool] Forcing get_weather...")
        let forcedResponse = try await session3.respond(
            to: "How is the weather?",
            options: GenerationOptions(toolCallingMode: .required)
        )
        print("✅ [Required tool] Response: \(forcedResponse.content)")
        print("   Usage: \(forcedResponse.usage)")
    }
}
