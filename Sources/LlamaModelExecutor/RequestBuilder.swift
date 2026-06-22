//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation
import FoundationModels

/// Pure translation: FoundationModels generation request → OpenAI-compatible
/// HTTP request body for llama.cpp server.
///
/// Every transcript entry type and generation option is mapped here so
/// ``LlamaExecutor`` only has to send the request and parse SSE.
package enum RequestBuilder {

    /// The result of building a request.
    package struct Built {
        /// The URL request ready to be sent via ``HTTPTransport``.
        package var urlRequest: URLRequest
    }

    /// Builds a streaming chat completions request from a FoundationModels request.
    ///
    /// - Parameter request: The generation request from the system.
    /// - Parameter modelName: The server-side model identifier.
    /// - Parameter temperature: Default temperature when the request doesn't specify one.
    /// - Parameter maxTokens: Default max tokens when the request doesn't specify one.
    /// - Parameter baseURL: The server's base URL.
    /// - Throws: ``LlamaError`` if the transcript cannot be serialized.
    /// - Returns: A ``Built`` request ready to send.
    package static func build(
        from request: LanguageModelExecutorGenerationRequest,
        modelName: String,
        temperature: Double,
        maxTokens: Int,
        baseURL: URL
    ) throws -> Built {
        let messages = try convertTranscriptEntries(request.transcript)
        let resolvedTemperature = request.generationOptions.temperature ?? temperature
        let resolvedMaxTokens = request.generationOptions.maximumResponseTokens ?? maxTokens

        var body: [String: Any] = [
            "model": modelName,
            "messages": messages,
            "stream": true,
            "temperature": resolvedTemperature,
            "max_tokens": resolvedMaxTokens,
        ]

        // Include tool definitions and tool_choice when tools are registered.
        if !request.enabledToolDefinitions.isEmpty {
            body["tools"] = try convertToolDefinitions(request.enabledToolDefinitions)

            if let mode = request.generationOptions.toolCallingMode {
                switch mode.kind {
                case .required:
                    body["tool_choice"] = "required"
                case .disallowed:
                    body["tool_choice"] = "none"
                case .allowed:
                    break  // let the model decide (default behaviour)
                @unknown default:
                    break
                }
            }
        }

        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        return Built(urlRequest: urlRequest)
    }

    // MARK: - Transcript conversion

    /// Convert FoundationModels transcript entries into OpenAI chat message dictionaries.
    private static func convertTranscriptEntries(_ transcript: Transcript) throws -> [[String: Any]] {
        var messages: [[String: Any]] = []

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                let text = extractText(from: instructions.segments)
                if !text.isEmpty {
                    messages.append(["role": "system", "content": text])
                }

            case .prompt(let prompt):
                let text = extractText(from: prompt.segments)
                messages.append(["role": "user", "content": text])

            case .response(let response):
                let text = extractText(from: response.segments)
                messages.append(["role": "assistant", "content": text])

            case .toolCalls(let toolCalls):
                let toolCallDicts: [[String: Any]] = toolCalls.map { call in
                    // GeneratedContent is not Encodable, but its description
                    // yields valid JSON for previously-emitted tool calls.
                    let argsJSON = String(describing: call.arguments)
                    return [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.toolName,
                            "arguments": argsJSON,
                        ],
                    ]
                }
                messages.append([
                    "role": "assistant",
                    "tool_calls": toolCallDicts,
                ])

            case .toolOutput(let output):
                let text = extractText(from: output.segments)
                messages.append([
                    "role": "tool",
                    "tool_call_id": output.id,
                    "content": text,
                ])
            case .reasoning:
                break
            @unknown default:
                break
            }
        }

        return messages
    }

    /// Convert FoundationModels tool definitions into the OpenAI `tools` array format.
    private static func convertToolDefinitions(
        _ tools: [Transcript.ToolDefinition]
    ) throws -> [[String: Any]] {
        let encoder = JSONEncoder()
        return try tools.map { tool in
            let paramsData = try encoder.encode(tool.parameters)
            let paramsJSON = try JSONSerialization.jsonObject(with: paramsData)
                as? [String: Any] ?? [:]
            return [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": paramsJSON,
                ],
            ]
        }
    }

    /// Extract plain text from an array of transcript segments.
    private static func extractText(from segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let textSegment):
                return textSegment.content
            case .structure(let structuredSegment):
                return String(describing: structuredSegment.content)
            case .attachment:
                return nil
            case .custom:
                return nil
            @unknown default:
                return nil
            }
        }.joined()
    }
}
