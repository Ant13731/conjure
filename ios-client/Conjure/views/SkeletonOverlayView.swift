//
//  SkeletonOverlayView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-27.
//

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

struct SkeletonOverlayView: View {
    @ObservedObject var skeletonConsumer: SkeletonOverlayFusedFrameConsumer
    @EnvironmentObject var recognitionSettings: PersistentSettings<RecognitionSettings>

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Draw skeleton lines first (behind joints)
                if recognitionSettings.value.showSkeletonLines {
                    Canvas { context, size in
                        for hand in skeletonConsumer.joints {
                            drawSkeletonLines(
                                hand: hand,
                                geometry: geometry,
                                context: &context
                            )
                        }
                    }
                }

                // Draw joints and tips on top
                ForEach(0..<skeletonConsumer.joints.count, id: \.self) { handIndex in
                    ForEach(0..<skeletonConsumer.joints[handIndex].count, id: \.self) {
                        jointIndex in
                        let joint = skeletonConsumer.joints[handIndex][jointIndex]

                        // Skip invisible landmarks if setting is off
                        if !joint.visible && !recognitionSettings.value.showInvisibleLandmarks {
                            EmptyView()

                        } else {
                            let screenX = CGFloat(joint.x) * geometry.size.width
                            let screenY = CGFloat(joint.y) * geometry.size.height

                            // Determine if this is a fingertip
                            let isTip = joint.isTip

                            // Select appropriate color settings
                            let (nearColor, farColor, threshold, limit) =
                                isTip
                                ? (
                                    recognitionSettings.value.fingerTipColorNear,
                                    recognitionSettings.value.fingerTipColorFar,
                                    recognitionSettings.value.clickDepthThreshold,
                                    recognitionSettings.value.clickDepthLimit
                                )
                                : (
                                    recognitionSettings.value.jointColorNear,
                                    recognitionSettings.value.jointColorFar,
                                    recognitionSettings.value.moveDepthThreshold,
                                    recognitionSettings.value.moveDepthLimit
                                )

                            // Show fingertips or joints based on settings
                            if (isTip && recognitionSettings.value.showFingerTips)
                                || (!isTip && recognitionSettings.value.showJoints)
                            {
                                let interpolatedColor = Color_.interpolateColor(
                                    near: nearColor,
                                    far: farColor,
                                    depth: joint.z,
                                    threshold: threshold,
                                    limit: limit
                                )

                                Circle()
                                    .fill(interpolatedColor)
                                    .frame(
                                        width: CGFloat(recognitionSettings.value.jointRadius),
                                        height: CGFloat(recognitionSettings.value.jointRadius)
                                    )
                                    .opacity(joint.visible ? 1.0 : 0.5)
                                    .position(x: screenX, y: screenY)
                            }
                        }
                    }
                }
            }
            .allowsHitTesting(false)  // Let touches pass through to camera
        }
    }

    private func drawSkeletonLines(
        hand: [SkeletonJointData],
        geometry: GeometryProxy,
        context: inout GraphicsContext
    ) {
        let skeletonColor = recognitionSettings.value.skeletonLineColor.toUIColor()

        var path = Path()

        for (start, end) in HandLandmarkIndex.connections {
            guard start < hand.count && end < hand.count else { continue }

            let startJoint = hand[start]
            let endJoint = hand[end]

            // Skip if either joint is invisible and setting is off
            if (!startJoint.visible || !endJoint.visible)
                && !recognitionSettings.value.showInvisibleLandmarks
            {
                continue
            }

            let startX = CGFloat(startJoint.x) * geometry.size.width
            let startY = CGFloat(startJoint.y) * geometry.size.height
            let endX = CGFloat(endJoint.x) * geometry.size.width
            let endY = CGFloat(endJoint.y) * geometry.size.height

            path.move(to: CGPoint(x: startX, y: startY))
            path.addLine(to: CGPoint(x: endX, y: endY))
        }

        context.stroke(
            path,
            with: .color(skeletonColor),
            lineWidth: CGFloat(recognitionSettings.value.lineWidth)
        )
    }
}

#Preview {
    Color.black
}
