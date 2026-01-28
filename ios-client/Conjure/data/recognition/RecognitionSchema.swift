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
    let handednessConfidence: Float
    let gestureConfidence: Float
}

enum UIDeviceOrientation_: Int, Codable {
    case unknown = 0
    case portrait = 1
    case portraitUpsideDown = 2
    case landscapeLeft = 3
    case landscapeRight = 4
    case faceUp = 5
    case faceDown = 6

    init(from orientation: UIDeviceOrientation) {
        switch orientation {
        case .unknown:
            self = .unknown
        case .portrait:
            self = .portrait
        case .portraitUpsideDown:
            self = .portraitUpsideDown
        case .landscapeLeft:
            self = .landscapeLeft
        case .landscapeRight:
            self = .landscapeRight
        case .faceUp:
            self = .faceUp
        case .faceDown:
            self = .faceDown
        @unknown default:
            self = .unknown
        }
    }
}

/// Final frame to send through the connection, containing all detected hands and their landmarks
struct LandmarkedFrame: Codable {
    let hands: [LandmarkedHand]
    let timestamp: Int
    let orientation: UIDeviceOrientation_
}

struct IntermediateCameraFrame {
    let rgb: CVImageBuffer
    let depth: CVPixelBuffer
    let ts: Int
    let orientation: UIDeviceOrientation_
}

struct IntermediateLandmarkFrame {
    let result: GestureRecognizerResult
    let ts: Int
}
