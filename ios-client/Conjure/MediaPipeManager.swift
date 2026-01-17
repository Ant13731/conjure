////
////  MediaPipeManager.swift
////  Conjure
////
////  Created by Anthony Hunt on 2025-11-19.
////
//
//import Foundation
//import MediaPipeTasksVision
//import Combine
//
//
//
//class MediaPipeManager: NSObject {
//    var handLandmark: GestureRecognizer!
//    unowned var frameFuser: FrameFuser!
//
//    init(frameFuser: FrameFuser) {
//        super.init()
//        self.frameFuser = frameFuser
//
////        let modelPath = Bundle.main.path(forResource: "hand_landmarker", ofType: "task")
////        let modelPath = "hand_landmarker.task"
//        let modelPath = "trained_mediapipe_gesture_recognizer.task"
//
//        let options = GestureRecognizerOptions()
//        options.baseOptions.modelAssetPath = modelPath
//        options.runningMode = .liveStream
////        options.minHandDetectionConfidence = DataConfig.minHandDetectionConfidence
////        options.minHandPresenceConfidence = DataConfig.minHandPresenceConfidence
////        options.minHandTrackingConfidence = DataConfig.minHandTrackingConfidence
////        options.numHands = DataConfig.numHands
//        options.gestureRecognizerLiveStreamDelegate = self
//
//        handLandmark = try! GestureRecognizer(options: options)
//
//    }
//
//}
//
//extension MediaPipeManager: GestureRecognizerLiveStreamDelegate {
//    func gestureRecognizer(
//        _ gestureRecognizer: GestureRecognizer,
//        didFinishGestureRecognition result: GestureRecognizerResult?,
//        timestampInMilliseconds: Int,
//        error: Error?) {
//            if (error != nil) {
//                print("Error when running mediapipe", error as Any)
//                    return
//            }
//            if result == nil {
//                print("No landmark result found")
//                return
//            }
//
////            let ts = CMTimeMake(value: CMTimeValue(timestampInMilliseconds), timescale: 1000)
//
//            let mediapipeFrame = IntermediateLandmarkFrame(result: result!, ts: timestampInMilliseconds)
//
//            Task {await frameFuser.sendMediaPipeFrame(mediapipeFrame)}
//
//      }
//}
