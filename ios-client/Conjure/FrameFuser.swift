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
    
    unowned private var webRTCClient: WebRTCClient?

    init(_ webRTCClient: WebRTCClient? = nil) {
           self.webRTCClient = webRTCClient
       }
    
    
    func sendCameraFrame(_ frame: IntermediateCameraFrame) {
        cameraBuffer[frame.ts] = frame
        tryFuse(at: frame.ts)
        prune()
    }

    func sendMediaPipeFrame(_ frame: IntermediateLandmarkFrame) {
        mpBuffer[frame.ts] = frame
        tryFuse(at: frame.ts)
        prune()
    }

    private func tryFuse(at ts: Int) {
        guard let c = cameraBuffer[ts],
              let m = mpBuffer[ts] else {
            print("Failed to find entries for mediapipe and camera at timestamp \(ts)")
            print("CamBuff:", cameraBuffer.keys)
            print("MPBuff_:", mpBuffer.keys)
            return }

        // Remove matched entries
        cameraBuffer[ts] = nil
        mpBuffer[ts] = nil
        
        var landmarks: [UInt8: Landmark] = [:]
        let handedness = m.result.handedness[0]
        
        
        
        let frame = Frame(handedness: handedness, timestamp: ts, landmarks:landmarks)
        
        Task {await webRTCClient!.send(frame: frame)}
        
        print("Fuser reached this point", c, m)

        // Emit final fused frame
        
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

