////
////  FrameFuser.swift
////  Conjure
////
////  Created by Anthony Hunt on 2025-11-19.
////



import AVFoundation
import MediaPipeTasksVision

actor FrameFuser {
    private var cameraBuffer: [Int: IntermediateCameraFrame] = [:]
    private var mpBuffer: [Int: IntermediateLandmarkFrame] = [:]
    private let maxBufferSize = 6

    unowned private var webRTCClient: WebRTCClient!

    init(_ webRTCClient: WebRTCClient) {
           self.webRTCClient = webRTCClient
       }


    func sendCameraFrame(_ frame: IntermediateCameraFrame) {
        cameraBuffer[frame.ts] = frame
        Task {await tryFuse(at: frame.ts)}
        prune()
    }

    func sendMediaPipeFrame(_ frame: IntermediateLandmarkFrame) {
        mpBuffer[frame.ts] = frame
        Task {await tryFuse(at: frame.ts)}
        prune()
    }

    private func tryFuse(at ts: Int) async {
        guard let c = cameraBuffer[ts],
              let m = mpBuffer[ts] else {
            print("Failed to find entries for mediapipe and camera at timestamp \(ts)")
            print("CamBuff:", cameraBuffer.keys)
            print("MPBuff_:", mpBuffer.keys)
            return }

        // Remove matched entries
        cameraBuffer[ts] = nil
        mpBuffer[ts] = nil


        let firstHand = await MainActor.run {m.result.handedness}
        if firstHand.isEmpty || firstHand[0].isEmpty {
            print("No hands detected")
            return
        }
        let confidence = firstHand[0][0].score
        let handedness = firstHand[0][0].categoryName == "Left"

        var landmarks: [UInt8: Landmark] = [:]
        let awaited_landmarks = await MainActor.run {m.result.landmarks[0]}
        for (i, landmark) in awaited_landmarks.enumerated() {
            let depth = depthAt(x: landmark.x, y: landmark.y, from: await MainActor.run {c.depth})
            landmarks[UInt8(i)] = Landmark(x: landmark.x, y: landmark.y, depth: depth)
        }

        let frame = Frame(handedness: handedness, mediapipeConfidence: confidence, timestamp: ts, landmarks:landmarks)

        Task {await webRTCClient.send(frame: frame)}

        print("Fuser reached this point", c, m)

        // Emit final fused frame

    }

    func depthAt(x: Float, y: Float, from depthBuffer: CVPixelBuffer) -> Float? {
        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)

        let px = Int(x * Float(depthWidth))
         let py = Int(y * Float(depthHeight))


        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        guard px >= 0 && px < width && py >= 0 && py < height else {
            return nil
        }

        let format = CVPixelBufferGetPixelFormatType(depthBuffer)
        let base = CVPixelBufferGetBaseAddress(depthBuffer)!

        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)

        switch format {
        case kCVPixelFormatType_DepthFloat16:
             let ptr = base.assumingMemoryBound(to: UInt16.self)
             let rowStride = bytesPerRow / MemoryLayout<UInt16>.size
             let f16bits = ptr[py * rowStride + px]

             // Convert UInt16 bit-pattern → Float16 → Float
             let f16 = Float16(bitPattern: f16bits)
             return Float(f16)

        case kCVPixelFormatType_DepthFloat32:
            let pointer = base.assumingMemoryBound(to: Float.self)
            let rowStart = py * (bytesPerRow / MemoryLayout<Float>.size)
            return pointer[rowStart + px]

        default:
            return nil
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

