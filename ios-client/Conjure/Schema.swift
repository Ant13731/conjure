//
//  Schema.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-18.
//

import AVFoundation
import Foundation
import MediaPipeTasksVision

/// Represents a single landmark/joint of a hand detected by mediapipe and post-processed with data from the TrueDepth sensor
struct Landmark: Codable {

    /// Normalized coordinate in [0.0, 1.0] representing horizontal position in the image
    let x: Float

    /// Normalized coordinate in [0.0, 1.0] representing vertical position in the image
    let y: Float

    /// Depth value relative to the camera's plane, taken from the TrueDepth sensor
    let z: Float?

    /// Depth value relative to the wrist according to mediapipe
    let relativeDepth: Float?

    /// Visibility of each joint may determine which depth to use
    let visible: Bool?
}

/// Represents a single detected hand with its landmarks and gesture
struct LandmarkedHand: Codable {
    /// Should be "Left" or "Right"
    let handedness: String
    let landmarks: [Landmark]
    let gesture: String
    let handedness_confidence: Float
    let gesture_confidence: Float
}

/// Final frame to send through the connection, containing all detected hands and their landmarks
struct LandmarkedFrame: Codable {
    let hands: [LandmarkedHand]
    let timestamp: Int
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
