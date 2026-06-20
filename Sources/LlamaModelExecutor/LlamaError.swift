
//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
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
