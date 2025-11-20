//
//  MediaPipeManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-19.
//

import Foundation
import MediaPipeTasksVision
import Combine



class MediaPipeManager: NSObject {
    private var handLandmarker: HandLandmarker!
    var cameraManagerCallback: ((HandLandmarkerResult) -> Void)!
    
    override init() {
        super.init()
        
//        let modelPath = Bundle.main.path(forResource: "hand_landmarker", ofType: "task")
        let modelPath = "hand_landmarker.task"
        
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
//        options.minHandDetectionConfidence = DataConfig.minHandDetectionConfidence
//        options.minHandPresenceConfidence = DataConfig.minHandPresenceConfidence
//        options.minHandTrackingConfidence = DataConfig.minHandTrackingConfidence
        options.numHands = DataConfig.numHands
        options.handLandmarkerLiveStreamDelegate = self

        handLandmarker = try! HandLandmarker(options: options)
        
    }

}

extension MediaPipeManager: HandLandmarkerLiveStreamDelegate {
    func handLandmarker(
        _ handLandmarker: HandLandmarker,
        didFinishDetection result: HandLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?) {
            if (error != nil) {
                print("Error when running mediapipe", error as Any)
                    return
            }
            if result == nil {
                print("No landmark result found")
                return
            }
            
            cameraManagerCallback(result!)
            
      }
}
