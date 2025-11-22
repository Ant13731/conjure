//
//  Config.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-18.
//

struct LoginConfig {
    static let defaultIPAddress = "172.20.10.7"
//    static let defaultIPAddress = "100.95.197.55"
//    static let defaultIPAddress = "100.115.181.103"
    static let defaultPort = "5000"
    static let defaultConnectionMethod = "WebRTC"
}


struct DataConfig {
    static let queueSize = 1
    static let landmarkDepthPixelRadius = 2
    static let webRTCChannelLabel = "hand_landmarks"
    static let numHands = 1
    static let minDepth = 0.1
    static let maxDepth = 1.5
//    static let minHandDetectionConfidence =
}
