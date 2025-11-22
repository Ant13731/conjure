//
//  CameraManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-18.
//

import AVFoundation
import SwiftUI
import Combine
import MediaPipeTasksVision

enum CameraSessionSetupError: Error{
    case notAuthorized
    case configurationFailed
    case failedToAddCamera
    case failedToAddDepthSensor
    case failedToAddDepthSensorCapture
}

class CameraManager: NSObject {
    
    private let captureSession = AVCaptureSession()
    private var videoOutput: AVCaptureVideoDataOutput!
    private var depthOutput: AVCaptureDepthDataOutput!
    private let synchronizerQueue = DispatchQueue(label: "camera.sync.queue")
    private var synchronizer: AVCaptureDataOutputSynchronizer!
    
    var isCameraAuthorized: Bool = false
    var isSessionSetup: Bool = false
    var isSessionRunning: Bool = false
    
    let mediapipeManager: MediaPipeManager?
    unowned var frameFuser: FrameFuser?
    
    let usbSender: USBSender?
    
    init(frameFuser: FrameFuser) {
        self.frameFuser = frameFuser
        self.mediapipeManager = MediaPipeManager(frameFuser: frameFuser)
        self.usbSender = nil
    }
    init(usbSender: USBSender) {
        self.usbSender = usbSender
        self.mediapipeManager = nil
        self.frameFuser = nil
    }
//    var webRTCClient: WebRTCClient!
//    
//    init(webRTCClient: WebRTCClient) {
//        self.webRTCClient = webRTCClient
//    }
    
    private func promptCameraPermissions() -> Bool {
        
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            
            // Determine if the user previously authorized camera access.
            isCameraAuthorized = status == .authorized
            
            // If the system hasn't determined the user's authorization status,
            // explicitly prompt them for approval.
            if status == .notDetermined {
                AVCaptureDevice.requestAccess(for: .video,
                                              completionHandler: {(result: Bool) -> Void in self.isCameraAuthorized=result})
            }
            
            return isCameraAuthorized
        
    }
    
    func setupSession() -> Result<Void, CameraSessionSetupError>{
        if isSessionRunning {
            stopSession()
        }
        
        isSessionSetup = false
        if promptCameraPermissions() == false { return .failure(.notAuthorized)}
        
        captureSession.beginConfiguration()
        captureSession.outputs.forEach{captureSession.removeOutput($0)}
        captureSession.sessionPreset = .vga640x480
        
//        // RGB camera
//        guard let frontCam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
//              let videoInput = try? AVCaptureDeviceInput(device: frontCam),
//              captureSession.canAddInput(videoInput) else { return .failure(.failedToAddCamera) }
//        captureSession.addInput(videoInput)
        
        
        // Depth camera
        guard let depthCam = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front),
              let depthInput = try? AVCaptureDeviceInput(device: depthCam),
              captureSession.canAddInput(depthInput) else { return .failure(.failedToAddDepthSensorCapture)}
        captureSession.addInput(depthInput)
        
        // RGB output
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoOutput.setSampleBufferDelegate(self, queue: synchronizerQueue)
        
        guard captureSession.canAddOutput(videoOutput) else {
            return .failure(.failedToAddCamera)
        }
        captureSession.addOutput(videoOutput)
        
        // Depth output
        depthOutput = AVCaptureDepthDataOutput()
        depthOutput.isFilteringEnabled = false// changing this to true may be faster than rolling our own
        depthOutput.setDelegate(self, callbackQueue: synchronizerQueue)
        guard captureSession.canAddOutput(depthOutput) else {
            return .failure(.failedToAddDepthSensor)
        }
        captureSession.addOutput(depthOutput)
        
        // Synchronize video + depth
        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer.setDelegate(self, queue: synchronizerQueue)
        
        captureSession.commitConfiguration()
        isSessionSetup = true
        return .success(())
    }
    
    
    
    func startSession() throws {
        
        struct SessionNotEnabledError: Error {let msg: String}
        if !isSessionSetup{
            throw SessionNotEnabledError(msg: "Must set up session before starting/stopping session")
        }
        captureSession.startRunning()
        isSessionRunning = true
    }
    
    func stopSession() {
        captureSession.stopRunning()
        isSessionRunning = false
    }
}

// MARK: - Delegates

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureDepthDataOutputDelegate, AVCaptureDataOutputSynchronizerDelegate {
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        guard let syncedVideo = synchronizedDataCollection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
              let syncedDepth = synchronizedDataCollection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData else {
            print("failed to sync video/depth data")
            return }
        
        guard !syncedVideo.sampleBufferWasDropped,
              !syncedDepth.depthDataWasDropped else {
            print("frame dropped")
            return }
        
        
        let lastDepthFrame = syncedDepth.depthData.depthDataMap
        let timestamp = CMSampleBufferGetPresentationTimeStamp(syncedVideo.sampleBuffer)
        let ts = Int(timestamp.seconds) * 1000
        
        if mediapipeManager != nil {
            guard let lastFrame = try? MPImage(sampleBuffer: syncedVideo.sampleBuffer) else {
                print("Failed to cast frame into MPImage")
                return
            }
            
            guard let _ = try? mediapipeManager!.handLandmark.detectAsync(image: lastFrame, timestampInMilliseconds: ts) else {
                //            print("Mediapipe discarded frame")
                return
            }
            
            let cameraFrame = IntermediateCameraFrame(rgb: lastFrame, depth: lastDepthFrame, ts: ts)
            
            Task {await frameFuser!.sendCameraFrame(cameraFrame)}
        } else {
            guard let videoBuffer = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer) else {
                print("Failed to get CVPixelBuffer from image frame")
                return
            }
//            CVPixelBufferLockBaseAddress(videoBuffer, .readOnly)
//                CVPixelBufferLockBaseAddress(lastDepthFrame, .readOnly)
//
//                defer {
//                    CVPixelBufferUnlockBaseAddress(videoBuffer, .readOnly)
//                    CVPixelBufferUnlockBaseAddress(lastDepthFrame, .readOnly)
//                }
//
//                let videoData = videoBuffer.extractNV12Data() // helper: converts NV12 pixel buffer to Data
//                let depthData = lastDepthFrame.extractUInt16Data() // helper: converts depth buffer to Data

                // --- Send over USB ---
                usbSender!.send(videoBuffer: videoBuffer, depthBuffer: lastDepthFrame)

        }
    }
}
