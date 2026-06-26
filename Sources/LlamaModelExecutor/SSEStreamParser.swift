//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation

/// Parses an SSE (Server-Sent Events) stream of lines from llama.cpp into
/// decoded ``StreamChunk`` values, handling malformed chunks with a configurable
/// tolerance threshold.
///
/// Usage
/// =====
/// ```swift
/// let (lines, _) = try await transport.lines(for: urlRequest)
/// let parser = SSEStreamParser(maxMalformedChunks: 5)
/// for try await chunk in parser.parse(lines) {
///     // Use chunk.choices, chunk.usage, chunk.timings…
/// }
/// ```
///
/// The parser automatically:
/// - Skips empty lines and non-`data:` lines.
/// - Strips the `"data: "` prefix from SSE events.
/// - Finishes the stream when it encounters `[DONE]`.
/// - Tracks consecutive malformed JSON chunks and throws ``LlamaError/streamError(_:)``
///   when the threshold is exceeded.
package struct SSEStreamParser {

    private let maxMalformedChunks: Int
    private let decoder: JSONDecoder

    /// Creates a parser with the given tolerance for malformed chunks.
    /// - Parameter maxMalformedChunks: Maximum consecutive malformed chunks before throwing.
    ///                                 Defaults to `5`.
    package init(maxMalformedChunks: Int = 5) {
        self.maxMalformedChunks = maxMalformedChunks
        self.decoder = JSONDecoder()
    }

    /// Parses raw SSE lines into a stream of decoded ``StreamChunk`` values.
    ///
    /// The returned stream:
    /// - Yields each valid ``StreamChunk`` from the SSE data events.
    /// - Finishes normally when the input ends or `[DONE]` is received.
    /// - Throws ``LlamaError/streamError(_:)`` if too many consecutive chunks
    ///   fail to decode.
    ///
    /// - Parameter lines: The raw SSE line stream produced by ``HTTPTransport/lines(for:)``.
    /// - Returns: An async stream of decoded ``StreamChunk`` values.
    package func parse(
        _ lines: AsyncThrowingStream<String, Error>
    ) -> AsyncThrowingStream<StreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var malformedCount = 0

                do {
                    for try await rawLine in lines {
                        let trimmed = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)

                        // Skip empty lines.
                        guard !trimmed.isEmpty else { continue }

                        // Only process lines that start with the SSE data prefix.
                        guard trimmed.hasPrefix("data: ") else { continue }

                        // Strip "data: " to get the JSON payload.
                        let payload = String(trimmed.dropFirst(6))

                        // The [DONE] sentinel signals the end of the stream.
                        guard payload != "[DONE]" else {
                            continuation.finish()
                            return
                        }

                        // Attempt to decode as a StreamChunk.
                        guard let jsonData = payload.data(using: .utf8) else {
                            malformedCount += 1
                            if malformedCount >= maxMalformedChunks {
                                throw LlamaError.streamError(
                                    "\(malformedCount) consecutive malformed SSE chunks — aborting"
                                )
                            }
                            continue
                        }

                        do {
                            let chunk = try decoder.decode(StreamChunk.self, from: jsonData)
                            malformedCount = 0  // Reset on success.
                            continuation.yield(chunk)
                        } catch {
                            malformedCount += 1
                            if malformedCount >= maxMalformedChunks {
                                throw LlamaError.streamError(
                                    "\(malformedCount) consecutive malformed SSE chunks — aborting"
                                )
                            }
                            continue
                        }
                    }

                    // Input stream ended without encountering [DONE].
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }
}
