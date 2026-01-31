//
//  SkeletonAnnotationFusedFrameConsumer.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-26.
//

import Combine
import SwiftUI

/// Hand landmark indices for MediaPipe
enum HandLandmarkIndex: Int {
    case wrist = 0
    case thumbCMC = 1
    case thumbMCP = 2
    case thumbIP = 3
    case thumbTip = 4
    case indexMCP = 5
    case indexPIP = 6
    case indexDIP = 7
    case indexTip = 8
    case middleMCP = 9
    case middlePIP = 10
    case middleDIP = 11
    case middleTip = 12
    case ringMCP = 13
    case ringPIP = 14
    case ringDIP = 15
    case ringTip = 16
    case pinkyMCP = 17
    case pinkyPIP = 18
    case pinkyDIP = 19
    case pinkyTip = 20

    static var fingerTipIndices: Set<Int> = [4, 8, 12, 16, 20]

    /// Hand skeleton connections (index pairs to draw lines between)
    static var connections: [(Int, Int)] = [
        // Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        // Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        // Middle finger
        (9, 10), (10, 11), (11, 12),
        (5, 9),
        // Ring finger
        (13, 14), (14, 15), (15, 16),
        (9, 13),
        // Pinky
        (17, 18), (18, 19), (19, 20),
        (13, 17),
        // Palm
        (0, 17),
    ]
}

struct SkeletonJointData {
    let x: Float  // normalized 0-1
    let y: Float  // normalized 0-1
    let z: Float  // depth in meters
    let visible: Bool
    let isTip: Bool
}

class SkeletonOverlayFusedFrameConsumer: FusedFrameConsumer, ObservableObject {
    @Published var joints: [[SkeletonJointData]] = []  // Array of hands, each with array of joints
    @Published var frameSize: CGSize = .zero

    private var previousOrientation: UIDeviceOrientation_?

    func consumeFusedFrame(_ frame: LandmarkedFrame) async {
        var allJoints: [[SkeletonJointData]] = []

        for hand in frame.hands {
            var handJoints: [SkeletonJointData] = []

            for (index, landmark) in hand.landmarks.enumerated() {
                let isTip = HandLandmarkIndex.fingerTipIndices.contains(index)
                let z = landmark.z ?? 0
                let visible = landmark.visible ?? true

                let (rotatedX, rotatedY) = rotateCoordinates(
                    x: landmark.x,
                    y: landmark.y,
                    orientation: frame.orientation  // You already have this!
                )

                let joint = SkeletonJointData(
                    x: rotatedX,
                    // x: landmark.x,
                    y: rotatedY,
                    // y: landmark.y,
                    z: z,
                    visible: visible,
                    isTip: isTip
                )
                handJoints.append(joint)
            }

            allJoints.append(handJoints)
        }

        await MainActor.run {
            self.joints = allJoints
        }
    }

    func rotateCoordinates(x: Float, y: Float, orientation: UIDeviceOrientation_) -> (
        Float, Float
    ) {
        switch orientation {
        case .portrait:
            previousOrientation = .portrait
            return (y, x)
        case .landscapeLeft:
            previousOrientation = .landscapeLeft
            return (x, 1 - y)
        case .landscapeRight:
            previousOrientation = .landscapeRight
            return (1 - x, y)
        default:
            if previousOrientation != nil {
                return rotateCoordinates(x: x, y: y, orientation: previousOrientation!)
            }
            return (x, y)

        }
    }
}
