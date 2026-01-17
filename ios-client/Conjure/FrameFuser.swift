//////
//////  FrameFuser.swift
//////  Conjure
//////
//////  Created by Anthony Hunt on 2025-11-19.
//////
//
//
//
//import AVFoundation
//import MediaPipeTasksVision
//
//actor FrameFuser {
//    private var cameraBuffer: [Int: IntermediateCameraFrame] = [:]
//    private var mpBuffer: [Int: IntermediateLandmarkFrame] = [:]
//    private let maxBufferSize = 6
//
//    unowned private var webRTCClient: WebRTCClient!
//
//    init(_ webRTCClient: WebRTCClient) {
//           self.webRTCClient = webRTCClient
//       }
//
//
//    func sendCameraFrame(_ frame: IntermediateCameraFrame) {
//        cameraBuffer[frame.ts] = frame
//        Task {await tryFuse(at: frame.ts)}
//        prune()
//    }
//
//    func sendMediaPipeFrame(_ frame: IntermediateLandmarkFrame) {
//        
//        mpBuffer[frame.ts] = frame
//        Task {await tryFuse(at: frame.ts)}
//        prune()
//    }
//
//    private func tryFuse(at ts: Int) async {
//        guard let c = cameraBuffer[ts],
//              let m = mpBuffer[ts] else {
//            print("Failed to find entries for mediapipe and camera at timestamp \(ts)")
//            print("CamBuff:", cameraBuffer.keys)
//            print("MPBuff_:", mpBuffer.keys)
//            return }
//
//        // Remove matched entries
//        cameraBuffer[ts] = nil
//        mpBuffer[ts] = nil
//
//
//        let firstHand = await MainActor.run {m.result.handedness}
//        if firstHand.isEmpty || firstHand[0].isEmpty {
//            print("No hands detected")
//            return
//        }
//        let handedness_confidence = firstHand[0][0].score
//        let handedness = firstHand[0][0].categoryName ?? "unknown"
//        
//        let gestures = await MainActor.run {m.result.gestures}
//        let gesture: String
//        let gesture_confidence: Float
//        if gestures.isEmpty || gestures[0].isEmpty {
//            print("No gesture detected")
//            gesture = "unknown"
//            gesture_confidence = 0
//        } else{
//            gesture = gestures[0][0].categoryName ?? "unknown"
//            gesture_confidence = gestures[0][0].score
//        }
//
//        var landmarks: [Landmark] = []
//        let awaited_landmarks = await MainActor.run {m.result.landmarks[0]}
//        for landmark in awaited_landmarks {
//            let depth = depthAt(x: landmark.x, y: landmark.y, from: await MainActor.run {c.depth})
//            landmarks.append(Landmark(x: landmark.x, y: landmark.y, z: depth, relativeDepth: depth, visible: landmark.visibility as? Bool))
//        }
//
////        let frame = LandmarkedFrame(handedness: handedness, gesture: gesture, handedness_confidence: handedness_confidence, gesture_confidence: gesture_confidence, timestamp: ts, landmarks:landmarks)
//        
//        print("Mediapipe frame retrieved with \(ts) at time  \(DispatchTime.now())")
//        
////        Task {await webRTCClient.send(frame: frame)}
//        print("Sent frame to server at \(ts) at time  \(DispatchTime.now())")
////        print("Fuser reached this point", c, m)
//
//        // Emit final fused frame
//
//    }
//    private func depthAtPixel(
//        px: Int,
//        py: Int,
//        width: Int,
//        bytesPerRow: Int,
//        base: UnsafeRawPointer,
//        format: OSType
//    ) -> Float {
//        
//        switch format {
//        case kCVPixelFormatType_DepthFloat16:
//            let ptr = base.assumingMemoryBound(to: UInt16.self)
//            let stride = bytesPerRow / MemoryLayout<UInt16>.size
//            let bits = ptr[py * stride + px]
//            let f16 = Float16(bitPattern: bits)
//            return Float(f16)
//            
//        case kCVPixelFormatType_DepthFloat32:
//            let ptr = base.assumingMemoryBound(to: Float.self)
//            let stride = bytesPerRow / MemoryLayout<Float>.size
//            return ptr[py * stride + px]
//            
//        default:
//            return 100
//        }
//    }
//
//    func depthAt(x: Float, y: Float, from depthBuffer: CVPixelBuffer, areaSize: Int = 3) -> Float {
//        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
//        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
//        
//        
//        let width = CVPixelBufferGetWidth(depthBuffer)
//        let height = CVPixelBufferGetHeight(depthBuffer)
//        
//        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
//        let format = CVPixelBufferGetPixelFormatType(depthBuffer)
//        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
//        
//        let xf = x * Float(width - 1)
//            let yf = y * Float(height - 1)
//            
//            let xCenter = Int(round(xf))
//            let yCenter = Int(round(yf))
//            
//            // Out-of-bounds → Python returns 100, Swift returns nil unless you want 100 too.
//            guard xCenter >= 0, xCenter < width,
//                  yCenter >= 0, yCenter < height else {
//                return 100   // match Python version exactly
//            }
//            
//            // Define grid window (same integer logic as Python)
//            let half = areaSize / 2
//            let xStart = max(xCenter - half, 0)
//            let xEnd   = min(xCenter + half + 1, width)
//            let yStart = max(yCenter - half, 0)
//            let yEnd   = min(yCenter + half + 1, height)
//            
//            // Closed container edges = no area → return center pixel
//            if xStart >= xEnd || yStart >= yEnd {
//                return depthAtPixel(px: xCenter, py: yCenter,
//                                    width: width, bytesPerRow: bytesPerRow,
//                                    base: base, format: format)
//            }
//            
//        var minDepth: Float = 100
//        
//        if format == kCVPixelFormatType_DepthFloat16 {
//                let ptr = base.assumingMemoryBound(to: UInt16.self)
//                let rowStride = bytesPerRow / MemoryLayout<UInt16>.size
//                
//                for py in yStart..<yEnd {
//                    let row = ptr + py * rowStride
//                    for px in xStart..<xEnd {
//                        let bits = row[px]
//                        let f16 = Float(Float16(bitPattern: bits))
//                        if f16 < minDepth { minDepth = f16 }
//                    }
//                }
//                
//            } else if format == kCVPixelFormatType_DepthFloat32 {
//                let ptr = base.assumingMemoryBound(to: Float.self)
//                let rowStride = bytesPerRow / MemoryLayout<Float>.size
//                
//                for py in yStart..<yEnd {
//                    let row = ptr + py * rowStride
//                    for px in xStart..<xEnd {
//                        let d = row[px]
//                        if d < minDepth { minDepth = d }
//                    }
//                }
//                
//            } else {
//                // Unsupported format → match Python default behavior?
//                return 100
//            }
//            
////            // --- Iterate through small block and find min depth
////            for py in yStart..<yEnd {
////                for px in xStart..<xEnd {
////                    if let d = depthAtPixel(
////                        px: px, py: py,
////                        width: width,
////                        bytesPerRow: bytesPerRow,
////                        base: base,
////                        format: format
////                    ) {
////                        if d < minDepth { minDepth = d }
////                    }
////                }
////            }
//            
//            return minDepth
////        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
////        let depthHeight = CVPixelBufferGetHeight(depthBuffer)
////        let px = Int(x * Float(depthWidth))
////        let py = Int(y * Float(depthHeight))
////
////        guard px >= 0 && px < width && py >= 0 && py < height else {
////            return 100
////        }
////
////        let format = CVPixelBufferGetPixelFormatType(depthBuffer)
////        let base = CVPixelBufferGetBaseAddress(depthBuffer)!
////
////        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
////
////        switch format {
////        case kCVPixelFormatType_DepthFloat16:
////             let ptr = base.assumingMemoryBound(to: UInt16.self)
////             let rowStride = bytesPerRow / MemoryLayout<UInt16>.size
////             let f16bits = ptr[py * rowStride + px]
////
////             // Convert UInt16 bit-pattern → Float16 → Float
////             let f16 = Float16(bitPattern: f16bits)
////             return Float(f16)
////
//////        case kCVPixelFormatType_DepthFloat32:
//////            let pointer = base.assumingMemoryBound(to: Float.self)
//////            let rowStart = py * (bytesPerRow / MemoryLayout<Float>.size)
//////            return pointer[rowStart + px]
////
////        default:
////            return 100
////        }
//    }
//
//    private func prune() {
//        // keep only tiny number of frames
//        if cameraBuffer.count > maxBufferSize {
//            if let oldest = cameraBuffer.keys.min() {
//                cameraBuffer.removeValue(forKey: oldest)
//            }
//        }
//        if mpBuffer.count > maxBufferSize {
//            if let oldest = mpBuffer.keys.min() {
//                mpBuffer.removeValue(forKey: oldest)
//            }
//        }
//    }
//}
//
