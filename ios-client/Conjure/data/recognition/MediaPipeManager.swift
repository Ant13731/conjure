//
//  MediaPipeManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-19.
//
import AVFoundation
import Combine
import Foundation
import MediaPipeTasksVision

class MediapipeManager: NSObject {
    let recognitionSettings: PersistentSettings<RecognitionSettings>
    var handLandmark: GestureRecognizer!
    unowned var frameFuser: FrameFuser?

    func addFrameFuser(_ frameFuser: FrameFuser) {
        self.frameFuser = frameFuser
    }

    init(recognitionSettings: PersistentSettings<RecognitionSettings>) {
        self.recognitionSettings = recognitionSettings
        super.init()

        // let modelPath = Bundle.main.path(forResource: "hand_landmarker", ofType: "task")
        // let modelPath = "hand_landmarker.task"
        let modelPath = "trained_mediapipe_gesture_recognizer.task"

        let options = GestureRecognizerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        // options.minHandDetectionConfidence = DataConfig.minHandDetectionConfidence
        // options.minHandPresenceConfidence = DataConfig.minHandPresenceConfidence
        // options.minHandTrackingConfidence = DataConfig.minHandTrackingConfidence
        // options.numHands = DataConfig.numHands
        options.gestureRecognizerLiveStreamDelegate = self

        handLandmark = try! GestureRecognizer(options: options)

    }

}

extension MediapipeManager: GestureRecognizerLiveStreamDelegate {

    func gestureRecognizer(
        _ gestureRecognizer: GestureRecognizer,
        didFinishGestureRecognition result: GestureRecognizerResult?,
        timestampInMilliseconds: Int,
        error: Error?
    ) {
        if error != nil {
            print("Error when running mediapipe", error as Any)
            return
        }
        if result == nil {
            print("No landmark result found")
            return
        }
        if frameFuser == nil {
            print("No frame fuser set in mediapipe manager")
            return
        }

        let mediapipeFrame = IntermediateLandmarkFrame(result: result!, ts: timestampInMilliseconds)
        Task { await frameFuser!.sendMediaPipeFrame(mediapipeFrame) }

    }
}
