////
////  FrameFuser.swift
////  Conjure
////
////  Created by Anthony Hunt on 2025-11-19.
////
//
//import AVFoundation
//
//actor FusionCenter {
//    private var cameraBuffer: [CMTime: IntermediateCameraFrame] = [:]
//    private var mpBuffer: [CMTime: IntermediateLandmarkFrame] = [:]
//    
//    private var mediapipeManager = MediaPipeManager()
//    
//
//    // A callback to deliver the fused output:
//    // This is reassigned by the caller
//    var onFusedFrame: ((IntermediateCameraFrame, IntermediateLandmarkFrame) -> Void)!
//    var sendCameraFrameToMediaPipe: ((IntermediateCameraFrame) -> Void)!
//    
//    
//    func cameraInput(_ frame: IntermediateCameraFrame) {
//        cameraBuffer[frame.ts] = frame
//        sendCameraFrameToMediaPipe(frame)
////        tryFuse(at: frame.ts)
//        
////        prune()
//    }
//
//    func mediapipeInput(_ frame: IntermediateLandmarkFrame) {
//        mpBuffer[frame.ts] = frame
//        tryFuse(at: frame.ts)
//        prune()
//    }
//
//    private func tryFuse(at ts: CMTime) {
//        guard let c = cameraBuffer[ts],
//              let m = mpBuffer[ts] else {
//            print("Failed to find entries for mediapipe and camera at timestamp \(ts)")
//            return }
//
//        // Remove matched entries
//        cameraBuffer[ts] = nil
//        mpBuffer[ts] = nil
//
//        // Emit final fused frame
//        onFusedFrame?(c, m)
//    }
//
//    private func prune() {
//        // keep only tiny number of frames
//        if cameraBuffer.count > 5 {
//            if let oldest = cameraBuffer.keys.min() {
//                cameraBuffer.removeValue(forKey: oldest)
//            }
//        }
//        if mpBuffer.count > 5 {
//            if let oldest = mpBuffer.keys.min() {
//                mpBuffer.removeValue(forKey: oldest)
//            }
//        }
//    }
//}
//
