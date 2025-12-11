//
//  Config.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-18.
//
enum ConnectionMode: String, CaseIterable, Identifiable {
    case onDevice = "On-device ML"
    case streamWebRTC = "WebRTC"
    case streamTCP = "TCP Stream"
    case streamUDP = "UDP Stream"

    var id: String {self.rawValue}
}

struct LoginConfig {
    static let defaultIPAddress = "172.20.10.15"
//    static let defaultIPAddress = "100.95.197.55"
//    static let defaultIPAddress = "100.115.181.103"
    static let defaultPort = "5000"
    static let defaultConnectionMode = ConnectionMode.streamUDP
//    static let defaultConnectionMode = ConnectionMode.onDevice
}


struct DataConfig {
    static let queueSize = 1
    static let landmarkDepthPixelRadius = 2
    static let webRTCChannelLabel = "hand_landmarks"
    static let numHands = 1
    static let minDepth: Float = 0.1
    static let maxDepth: Float = 1.5
//    static let minHandDetectionConfidence =
}
