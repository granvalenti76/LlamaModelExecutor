//
//  File.swift
//  LlamaModelExecutor
//
//  Created by luca travaglini on 20/06/2026.
//

import Foundation


/// A configuration struct that defines the parameters  for LlamaModelExecutor
/// This configuration conform to 'Hashable' ,'Sendable' , 'Codable'

struct LlamaConfiguration: Hashable,Sendable, Codable {
    let modelName: String
    let temperature: Double
    let maxTokens: Int
    let baseURL: URL
    
    static let defaultURL = URL(string:"http://127.0.0.1:8080/v1")!
   
    init(modelName: String, temperature: Double = 0.7 , maxTokens: Int = 32000, baseURL: URL = defaultURL) {
        self.modelName = modelName
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.baseURL = baseURL
    }
}
