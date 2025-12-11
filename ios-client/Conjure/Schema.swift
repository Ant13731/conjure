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
    let z: Float?
    let visibility: Float?

     //0 for right hand, 1 for left
}

struct Frame: Codable {
    let handedness: String
    let gesture: String
    let handedness_confidence: Float
    let gesture_confidence: Float
//    let id: UInt32
    let timestamp: Int
    // Key corresponds to mediapipe digit labels (1 for thumb tip, etc.)
    let landmarks: [Landmark]
}


struct IntermediateCameraFrame {
    let rgb: MPImage
    let depth: CVPixelBuffer
    let ts: Int
}
struct IntermediateLandmarkFrame {
    let result: GestureRecognizerResult
    let ts: Int
}
