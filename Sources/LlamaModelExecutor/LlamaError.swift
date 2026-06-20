//
//  LlamaError.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation

/// Errors that can occur during communication with llama-server.
enum LlamaError: Error, LocalizedError, Equatable {
    case invalidResponse
    case httpError(statusCode: Int)
    case streamError(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from llama-server"
        case .httpError(let code):
            return "llama-server returned HTTP \(code)"
        case .streamError(let message):
            return "Stream error: \(message)"
        }
    }
}
