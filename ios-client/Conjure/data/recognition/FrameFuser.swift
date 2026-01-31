////
////  FrameFuser.swift
////  Conjure
////
////  Created by Anthony Hunt on 2025-11-19.
////

import AVFoundation
import MediaPipeTasksVision
import SwiftUI

protocol FusedFrameConsumer {
    func consumeFusedFrame(_ frame: LandmarkedFrame) async
}

actor FrameFuser {
    private var cameraBuffer: [Int: IntermediateCameraFrame] = [:]
    private var mpBuffer: [Int: IntermediateLandmarkFrame] = [:]
    private let maxBufferSize = 15

    // Receivers into this class
    func sendCameraFrame(_ frame: IntermediateCameraFrame) async {
        cameraBuffer[frame.ts] = frame
        await tryFuse(at: frame.ts)
        prune()
    }

    func sendMediaPipeFrame(_ frame: IntermediateLandmarkFrame) async {
        mpBuffer[frame.ts] = frame
        await tryFuse(at: frame.ts)
        prune()
    }

    // Recievers out of this class
    @MainActor
    private(set) var fusedFrameConsumers: [FusedFrameConsumer] = []
    @MainActor
    func addFusedFrameConsumer(_ consumer: FusedFrameConsumer) {
        fusedFrameConsumers.append(consumer)
    }
    @MainActor
    func removeFusedFrameConsumer(_ consumer: FusedFrameConsumer) {
        fusedFrameConsumers.removeAll { $0 as AnyObject === consumer as AnyObject }
    }
    @MainActor
    func clearFusedFrameConsumers() {
        fusedFrameConsumers.removeAll()
    }

    // Fusion logic
    private func tryFuse(at ts: Int) async {
        guard let intermediateCameraFrame = cameraBuffer[ts],
            let intermediateLandmarkFrame = mpBuffer[ts]
        else {
            print("Failed to find entries for mediapipe and camera at timestamp \(ts)")
            print("CamBuff:", cameraBuffer.keys)
            print("MPBuff_:", mpBuffer.keys)
            return
        }

        // Remove matched entries
        cameraBuffer[ts] = nil
        mpBuffer[ts] = nil

        let landmarkedFrame = await mergeIntermediateFrames(
            intermediateCameraFrame: intermediateCameraFrame,
            intermediateLandmarkFrame: intermediateLandmarkFrame,
            ts: ts)

        guard let landmarkedFrame else {
            print("Failed to merge intermediate frames at timestamp \(ts)")
            return
        }

        for consumer in await fusedFrameConsumers {
            print("Sending fused frame at ts \(ts) to consumer \(consumer)")
            Task { await consumer.consumeFusedFrame(landmarkedFrame) }
        }
    }

    private func mergeIntermediateFrames(
        intermediateCameraFrame: IntermediateCameraFrame,
        intermediateLandmarkFrame: IntermediateLandmarkFrame,
        ts: Int
    ) async -> LandmarkedFrame? {
        let (handednessList, gestureList, landmarksList) = await MainActor.run {
            (
                intermediateLandmarkFrame.result.handedness,
                intermediateLandmarkFrame.result.gestures,
                intermediateLandmarkFrame.result.landmarks
            )
        }

        if landmarksList.isEmpty || handednessList.isEmpty || gestureList.isEmpty {
            print("No hands detected")
            return nil
        }

        var landmarkedHands: [LandmarkedHand] = []
        for (handedness, (gesture, landmarks)) in zip(
            handednessList, zip(gestureList, landmarksList))
        {
            // Get handedness
            if handedness.isEmpty {
                print("Handedness is empty for a hand, skipping")
                continue
            }
            let handednessConfidence = handedness[0].score
            let handedness = handedness[0].categoryName ?? "unknown"

            // Get gesture
            if gesture.isEmpty {
                print("Gesture is empty for a hand, skipping")
                continue
            }
            let gestureName = gesture[0].categoryName ?? "unknown"
            let gestureConfidence = gesture[0].score

            // Get depth values for landmarks
            var landmarksWithDepth: [Landmark] = []
            for landmark in landmarks {
                let depth = depthAt(
                    x: landmark.x,
                    y: landmark.y,
                    from: await MainActor.run { intermediateCameraFrame.depth })
                landmarksWithDepth.append(
                    Landmark(
                        x: landmark.x, y: landmark.y, z: depth, relativeDepth: depth,
                        visible: landmark.visibility as? Bool))
            }

            landmarkedHands.append(
                LandmarkedHand(
                    handedness: handedness,
                    landmarks: landmarksWithDepth,
                    gesture: gestureName,
                    handednessConfidence: handednessConfidence,
                    gestureConfidence: gestureConfidence
                )
            )
        }
        return LandmarkedFrame(
            hands: landmarkedHands,
            timestamp: ts,
            orientation: intermediateCameraFrame.orientation
        )
    }

    static let INVALID_DEPTH: Float = 100

    func depthAt(x: Float, y: Float, from depthBuffer: CVPixelBuffer, areaSize: Int = 3) -> Float {
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        let format = CVPixelBufferGetPixelFormatType(depthBuffer)
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!

        // Grab the center of the pixel corresponding to normalized coords
        let xCenter = Int(round(x * Float(width - 1)))
        let yCenter = Int(round(y * Float(height - 1)))

        // Center pixel is out-of-bounds - Return a very far depth (100)
        guard xCenter >= 0, xCenter < width, yCenter >= 0, yCenter < height else {
            return FrameFuser.INVALID_DEPTH  // Matches expectation in python
        }

        // Define search region (in pixel size)
        // Anything outside this border is removed from consideration for depth values
        let half = areaSize / 2
        let xStart = max(xCenter - half, areaSize)
        let xEnd = min(xCenter + half + 1, width - areaSize)
        let yStart = max(yCenter - half, areaSize)
        let yEnd = min(yCenter + half + 1, height - areaSize)

        guard areaSize < width, areaSize < height else {
            print(
                "Area search size too large for depth buffer dimensions: \(areaSize) vs \(width)x\(height)"
            )
            return FrameFuser.INVALID_DEPTH
        }

        guard xStart < xEnd, yStart < yEnd else {
            print(
                "Area search size resulted in invalid search region: x[\(xStart), \(xEnd)), y[\(yStart), \(yEnd))"
            )
            return FrameFuser.INVALID_DEPTH
        }

        // When we lookup a pixel, it can be either a 16 bit float or a 32 bit float
        // In practice, this is usually just a 16 bit float, but different cameras may use different bits
        let getDepthValueWithFormat = depthAtWithFormat(
            base: base,
            bytesPerRow: bytesPerRow,
            format: format
        )

        var minDepth: Float = FrameFuser.INVALID_DEPTH
        for py in yStart..<yEnd {
            for px in xStart..<xEnd {
                minDepth = min(minDepth, getDepthValueWithFormat(px, py))
            }
        }

        return minDepth
    }

    private func depthAtWithFormat(
        base: UnsafeRawPointer,
        bytesPerRow: Int,
        format: OSType,
    ) -> ((Int, Int) -> Float) {
        switch format {
        case kCVPixelFormatType_DepthFloat16:
            let ptr = base.assumingMemoryBound(to: UInt16.self)
            let rowStride = bytesPerRow / MemoryLayout<UInt16>.size
            return { px, py in Float(Float16(bitPattern: ptr[py * rowStride + px])) }

        case kCVPixelFormatType_DepthFloat32:
            let ptr = base.assumingMemoryBound(to: Float.self)
            let rowStride = bytesPerRow / MemoryLayout<Float>.size
            return { px, py in ptr[py * rowStride + px] }

        default:
            return { _, __ in FrameFuser.INVALID_DEPTH }
        }
    }

    private func prune() {
        // keep only tiny number of frames
        if cameraBuffer.count > maxBufferSize {
            if let oldest = cameraBuffer.keys.min() {
                cameraBuffer.removeValue(forKey: oldest)
            }
        }
        if mpBuffer.count > maxBufferSize {
            if let oldest = mpBuffer.keys.min() {
                mpBuffer.removeValue(forKey: oldest)
            }
        }
    }
}
