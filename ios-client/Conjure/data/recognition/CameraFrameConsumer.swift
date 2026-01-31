//
//  CameraFrameConsumer.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-24.
//
import ARKit
import AVFoundation
import Combine
import MediaPipeTasksVision
import SwiftUI

class CameraFrameConsumer {
    func consumeFrame(
        rgbFrame: CMSampleBuffer,
        depthFrame: CVPixelBuffer,
        ts: Int,
        orientation: UIDeviceOrientation_
    ) async {
        // Override in subclasses
    }
}

class MediapipeFrameConsumer: CameraFrameConsumer {
    private let mediapipeManager: MediapipeManager

    init(mediapipeManager: MediapipeManager) {
        self.mediapipeManager = mediapipeManager
    }

    override func consumeFrame(
        rgbFrame: CMSampleBuffer,
        depthFrame: CVPixelBuffer,
        ts: Int,
        orientation: UIDeviceOrientation_
    ) async {
        guard let mpImage = try? MPImage(sampleBuffer: rgbFrame) else {
            print("MediapipeFrameConsumer: Failed to cast frame into MPImage")
            return
        }

        print("Sending frame to mediapipe with ts \(ts) at time \(DispatchTime.now())")
        do {
            try mediapipeManager.handLandmark.recognizeAsync(
                image: mpImage, timestampInMilliseconds: ts)
        } catch {
            print("Caught mediapipe error", error)
        }
    }
}

class RGBFrameConsumer: CameraFrameConsumer {
    let frameFuser: FrameFuser
    init(frameFuser: FrameFuser) {
        self.frameFuser = frameFuser
    }

    override func consumeFrame(
        rgbFrame: CMSampleBuffer,
        depthFrame: CVPixelBuffer,
        ts: Int,
        orientation: UIDeviceOrientation_
    ) async {
        guard let videoBuffer = CMSampleBufferGetImageBuffer(rgbFrame) else {
            print("Failed to get CVPixelBuffer from image frame")
            return
        }
        let cameraFrame = IntermediateCameraFrame(
            rgb: videoBuffer,
            depth: depthFrame,
            ts: ts,
            orientation: orientation
        )
        await frameFuser.sendCameraFrame(cameraFrame)
    }
}

// class USBFrameConsumer: CameraFrameConsumer {
//     override func consumeFrame(rgbFrame: CMSampleBuffer, depthFrame: CVPixelBuffer, ts: Int) {
//         guard let videoBuffer = CMSampleBufferGetImageBuffer(rgbFrame) else {
//             print("Failed to get CVPixelBuffer from image frame")
//             return
//         }
//         // --- Send over USB ---
//         usbSender!.send(videoBuffer: videoBuffer, depthBuffer: depthFrame)
//     }
// }
