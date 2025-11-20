//
//  Schema.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-18.
//

import Foundation
import MediaPipeTasksVision
import AVFoundation

struct Landmark: Codable {
    let x: Float
    let y: Float
    let depth: Float?
    let mediapipeConfidence: Float
}

struct Frame: Codable {
    let id: UInt32
    let timestamp: UInt64
    // Key corresponds to mediapipe digit labels (1 for thumb tip, etc.)
    let landmarks: [UInt8: Landmark]
}


struct IntermediateCameraFrame {
    let rgb: CMSampleBuffer
    let depth: AVDepthData
    let ts: CMTime
}
struct IntermediateLandmarkFrame {
    let result: HandLandmarkerResult
    let ts: CMTime
}
