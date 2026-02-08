//
//  FusedFrameConsumer.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-02-08.
//

import ARKit
import AVFoundation
import Accelerate
import Combine
import SwiftUI

class CommunicationFusedFrameConsumer: FusedFrameConsumer {
    private let communicationManager: CommunicationManager

    init(communicationManager: CommunicationManager) {
        self.communicationManager = communicationManager
    }

    func consumeFusedFrame(_ frame: LandmarkedFrame) async {
        print(
            "CommunicationFusedFrameConsumer: Sending fused frame with gesture \(frame.hands.first?.gesture ?? "blank")"
        )
        if let errMsg = communicationManager.send(frame: frame) {
            print("CommunicationFusedFrameConsumer: Error sending frame: \(errMsg)")
        }
    }
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

                let joint = SkeletonJointData(
                    x: (1 - landmark.x),
                    y: landmark.y,
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
}
