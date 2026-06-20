
//
//  LlamaModelExecutor
//
//  Copyright (c) 2026 Luca Travaglini. All rights reserved.
//  Licensed under MIT License. See LICENSE file for details.
//

import Foundation

/// The HTTP seam ``LlamaExecutor`` talks through. Production uses
/// ``URLSessionTransport``; tests inject a fake.
///
/// The streaming variant yields trimmed lines rather than raw bytes so both
/// the real implementation and a fake can speak the same vocabulary — SSE
/// (the wire format used by llama.cpp) is inherently line-oriented.
public protocol HTTPTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, URLResponse)
    func lines(
        for request: URLRequest
    ) async throws -> (AsyncThrowingStream<String, Error>, URLResponse)
}

/// `URLSession`-backed transport used in production.
public struct URLSessionTransport: HTTPTransport {
    private let session: URLSession

    /// Creates a transport backed by the given URL session.
    /// - Parameter session: The URL session to use (defaults to ``URLSession.shared``).
    public init(session: URLSession = .shared) {
        self.session = session
    }

    /// Non-streaming request. Returns the full response body.
    public func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        try await session.data(for: request)
    }

    /// Bridges `URLSession.AsyncBytes.lines` into an ``AsyncThrowingStream``
    /// so a fake can produce the same vocabulary without a live connection.
    public func lines(
        for request: URLRequest
    ) async throws -> (AsyncThrowingStream<String, Error>, URLResponse) {
        let (asyncBytes, response) = try await session.bytes(for: request)
        let stream = AsyncThrowingStream<String, Error> { continuation in
            let task = Task {
                do {
                    for try await line in asyncBytes.lines {
                        continuation.yield(line)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        return (stream, response)
    }
}
