////
////  CameraManager.swift
////  Conjure
////
////  Created by Anthony Hunt on 2025-11-18.
////
//
import AVFoundation
import Combine
import SwiftUI

class CameraManager: NSObject, ObservableObject {
    // Must keep these here so they arent deallocated
    private let captureSession = AVCaptureSession()
    private let synchronizerQueue = DispatchQueue(label: "camera.sync.queue")

    // output stream
    private var rgbOut: AVCaptureVideoDataOutput!
    private var depthOut: AVCaptureDepthDataOutput!
    private var synchronizer: AVCaptureDataOutputSynchronizer!

    var isSessionSetUp: Bool = false
    var isSessionRunning: Bool = false

    private(set) var frameConsumers: [CameraFrameConsumer] = []

    lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        return previewLayer
    }()

    func addConsumer(_ consumer: CameraFrameConsumer) {
        frameConsumers.append(consumer)
    }

    func removeConsumer(_ consumer: CameraFrameConsumer) {
        frameConsumers.removeAll { $0 === consumer }
    }

    func setupSession() async -> String? {
        if isSessionRunning {
            stopSession()
        }
        isSessionSetUp = false

        if !(await checkPermissions()) {
            return "Camera permissions are not granted. Please enable camera access in settings."
        }

        captureSession.beginConfiguration()
        captureSession.outputs.forEach { captureSession.removeOutput($0) }
        captureSession.sessionPreset = .vga640x480

        let cameraError = setupCamera()
        if cameraError != nil {
            return cameraError
        }

        let rgbOutError = configureRGBOutput()
        if rgbOutError != nil {
            return rgbOutError
        }

        let depthOutError = configureDepthOutput()
        if depthOutError != nil {
            return depthOutError
        }

        // Synchronize video + depth
        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [rgbOut, depthOut])
        synchronizer.setDelegate(self, queue: synchronizerQueue)

        captureSession.commitConfiguration()
        isSessionSetUp = true
        return nil
    }

    func startSession() -> String? {
        if !isSessionSetUp {
            isSessionRunning = false
            return "Must set up session before starting/stopping session"
        }

        captureSession.startRunning()
        isSessionRunning = true
        return nil
    }

    func stopSession() {
        captureSession.stopRunning()
        isSessionRunning = false
    }
}

// MARK: Helpers for setupSession
extension CameraManager {
    private func setupCamera() -> String? {
        guard
            // Get the default (only) front facing camera
            let camera = AVCaptureDevice.default(
                .builtInTrueDepthCamera,
                for: .video,
                position: .front
            )
        else {
            return "Failed to get depth camera"
        }

        // Depth formats also carry RGB data
        let availableDepthFormats = camera.activeFormat.supportedDepthDataFormats
        let filteredAvailableDepthFormats = availableDepthFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription)
                == kCVPixelFormatType_DepthFloat16
        })
        // Get the highest resolution depth format (640x480)
        let selectedDepthFormat = filteredAvailableDepthFormats.max(by: {
            first, second in
            CMVideoFormatDescriptionGetDimensions(first.formatDescription).width
                < CMVideoFormatDescriptionGetDimensions(second.formatDescription).width
        })

        do {
            try camera.lockForConfiguration()
            camera.activeDepthDataFormat = selectedDepthFormat
            camera.unlockForConfiguration()
        } catch {
            print("Could not lock device for comera configuration (caught exception): \(error)")
        }

        let depthCameraInput: AVCaptureDeviceInput!
        do {
            depthCameraInput = try AVCaptureDeviceInput(device: camera)
        } catch {
            return
                "Failed to add TrueDepth camera input to capture session (caught exception): \(error)"
        }

        guard captureSession.canAddInput(depthCameraInput) else {
            return
                "Failed to add TrueDepth camera input to capture session (canAddInput returned false)"
        }
        captureSession.addInput(depthCameraInput)
        return nil
    }

    /// Add RGB output video stream to session
    private func configureRGBOutput() -> String? {
        rgbOut = AVCaptureVideoDataOutput()
        rgbOut.alwaysDiscardsLateVideoFrames = true
        rgbOut.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]

        // Send RGB frames to the synchronizer
        rgbOut.setSampleBufferDelegate(self, queue: synchronizerQueue)

        guard captureSession.canAddOutput(rgbOut) else {
            return "Failed to add RGB video output to capture session (canAddOutput returned false)"
        }
        captureSession.addOutput(rgbOut)
        return nil
    }

    /// Add depth output video stream to session
    private func configureDepthOutput() -> String? {
        depthOut = AVCaptureDepthDataOutput()
        depthOut.isFilteringEnabled = false  // changing this to true may be faster than rolling our own

        // Send depth frames to the synchronizer
        depthOut.setDelegate(self, callbackQueue: synchronizerQueue)

        guard captureSession.canAddOutput(depthOut) else {
            return
                "Failed to add depth video output to capture session (canAddOutput returned false)"
        }
        captureSession.addOutput(depthOut)
        return nil
    }

}

// MARK: Synchronization Delegate
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate,
    AVCaptureDepthDataOutputDelegate, AVCaptureDataOutputSynchronizerDelegate
{

    func dataOutputSynchronizer(
        _ synchronizer: AVCaptureDataOutputSynchronizer,
        didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection
    ) {
        guard
            let syncedVideo = synchronizedDataCollection.synchronizedData(for: rgbOut)
                as? AVCaptureSynchronizedSampleBufferData,
            let syncedDepth = synchronizedDataCollection.synchronizedData(for: depthOut)
                as? AVCaptureSynchronizedDepthData
        else {
            print("Failed to sync video/depth data")
            return
        }

        guard !syncedVideo.sampleBufferWasDropped,
            !syncedDepth.depthDataWasDropped
        else {
            print("Frame dropped")
            return
        }

        let rgbFrame = syncedVideo.sampleBuffer
        let timestamp = CMSampleBufferGetPresentationTimeStamp(rgbFrame)
        let ts = Int(timestamp.seconds * 1000)

        let depthFrame: CVPixelBuffer!
        if syncedDepth.depthData.depthDataType == kCVPixelFormatType_DisparityFloat16 {
            // In case depth data is in an unknown format, convert to 16 bit floating absolute depth
            let depthDataConverted = syncedDepth.depthData.converting(
                toDepthDataType: kCVPixelFormatType_DepthFloat16)
            depthFrame = depthDataConverted.depthDataMap
        } else {
            depthFrame = syncedDepth.depthData.depthDataMap
        }

        for consumer in frameConsumers {
            Task {
                await consumer.consumeFrame(
                    rgbFrame: rgbFrame,
                    depthFrame: depthFrame,
                    ts: ts
                )
            }
        }
    }

}

// MARK: Camrera Permissions
extension CameraManager {
    func checkPermissions() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    continuation.resume(returning: granted)
                }
            }
        default:
            return false
        }
    }
}

//import MediaPipeTasksVision
//
//enum CameraSessionSetupError: Error{
//    case notAuthorized
//    case configurationFailed
//    case failedToAddCamera
//    case failedToAddDepthSensor
//    case failedToAddDepthSensorCapture
//}
//
//class CameraManager: NSObject {
//
//    private let captureSession = AVCaptureSession()
//    private var videoOutput: AVCaptureVideoDataOutput!
//    private var depthOutput: AVCaptureDepthDataOutput!
//    private let synchronizerQueue = DispatchQueue(label: "camera.sync.queue")
//    private var synchronizer: AVCaptureDataOutputSynchronizer!
//
//    var isCameraAuthorized: Bool = false
//    var isSessionSetup: Bool = false
//    var isSessionRunning: Bool = false
//
//    let mediapipeManager: MediaPipeManager?
//    unowned var frameFuser: FrameFuser?
//    var webRTCClient: WebRTCClient?
//
//    let usbSender: USBSender?
//
//    init(frameFuser: FrameFuser) {
//        self.frameFuser = frameFuser
//        self.mediapipeManager = MediaPipeManager(frameFuser: frameFuser)
//        self.usbSender = nil
//        self.webRTCClient = nil
//    }
//    init(webRTCClient: WebRTCClient) {
//        self.webRTCClient = webRTCClient
//        print("Test:", webRTCClient)
//        self.mediapipeManager = nil
//        self.frameFuser = nil
//        self.usbSender = nil
//    }
//    init(usbSender: USBSender) {
//        self.usbSender = usbSender
//        self.mediapipeManager = nil
//        self.frameFuser = nil
//        self.webRTCClient = nil
//    }
//
//    private func promptCameraPermissions() -> Bool {
//
//            let status = AVCaptureDevice.authorizationStatus(for: .video)
//
//            // Determine if the user previously authorized camera access.
//            isCameraAuthorized = status == .authorized
//
//            // If the system hasn't determined the user's authorization status,
//            // explicitly prompt them for approval.
//            if status == .notDetermined {
//                AVCaptureDevice.requestAccess(for: .video,
//                                              completionHandler: {(result: Bool) -> Void in self.isCameraAuthorized=result})
//            }
//
//            return isCameraAuthorized
//
//    }
//
//    func setupSession() -> Result<Void, CameraSessionSetupError>{
//        if isSessionRunning {
//            stopSession()
//        }
//
//        isSessionSetup = false
//        if promptCameraPermissions() == false { return .failure(.notAuthorized)}
//
//        captureSession.beginConfiguration()
//        captureSession.outputs.forEach{captureSession.removeOutput($0)}
//        captureSession.sessionPreset = .vga640x480
//
////        // RGB camera
////        guard let frontCam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
////              let videoInput = try? AVCaptureDeviceInput(device: frontCam),
////              captureSession.canAddInput(videoInput) else { return .failure(.failedToAddCamera) }
////        captureSession.addInput(videoInput)
//
//
//        // Depth camera
//        guard let depthCam = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else { return .failure(.failedToAddDepthSensorCapture)}
//
////        var depthFormats: [AVCaptureDevice.Format] = []
////
////        for format in depthCam.formats {
////            for depthFormat in format.supportedDepthDataFormats {
////                depthFormats.append(depthFormat)
////            }
////        }
////        if let float16Format = depthCam.activeFormat.supportedDepthDataFormats.first(where: {
////            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
////        }) {
////            try! depthCam.lockForConfiguration()
////            depthCam.activeDepthDataFormat = float16Format
////            depthCam.unlockForConfiguration()
////        } else {print("Failed to edit depth format")}
//        let depthFormats = depthCam.activeFormat.supportedDepthDataFormats
//        print("depthFormats", depthFormats)
//                let filtered = depthFormats.filter({
//                    CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
//                })
//        print("Filtered formats", filtered)
//                let selectedFormat = filtered.max(by: {
//                    first, second in CMVideoFormatDescriptionGetDimensions(first.formatDescription).width < CMVideoFormatDescriptionGetDimensions(second.formatDescription).width
//                })
//
//                do {
//                    try depthCam.lockForConfiguration()
//                    print("Applying selected format", selectedFormat)
//                    depthCam.activeDepthDataFormat = selectedFormat
//                    depthCam.unlockForConfiguration()
//                } catch {
//                    print("Could not lock device for configuration: \(error)")
//                    return .failure(.configurationFailed)
//                }
//
//        guard let depthInput = try? AVCaptureDeviceInput(device: depthCam), captureSession.canAddInput(depthInput) else { return .failure(.failedToAddDepthSensorCapture)}
//
//        captureSession.addInput(depthInput)
//
//        // RGB output
//        videoOutput = AVCaptureVideoDataOutput()
//        videoOutput.alwaysDiscardsLateVideoFrames = true
//        if usbSender == nil {
//            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
//                                         kCVPixelBufferMetalCompatibilityKey as String: true]
//        }
//        else {
//            print(videoOutput.availableVideoPixelFormatTypes)
//            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
//        }
//        videoOutput.setSampleBufferDelegate(self, queue: synchronizerQueue)
//
//        guard captureSession.canAddOutput(videoOutput) else {
//            return .failure(.failedToAddCamera)
//        }
//        captureSession.addOutput(videoOutput)
//
//        // Depth output
//        depthOutput = AVCaptureDepthDataOutput()
//        depthOutput.isFilteringEnabled = false// changing this to true may be faster than rolling our own
//
//        depthOutput.setDelegate(self, callbackQueue: synchronizerQueue)
//        guard captureSession.canAddOutput(depthOutput) else {
//            return .failure(.failedToAddDepthSensor)
//        }
//        captureSession.addOutput(depthOutput)
//
//        // Synchronize video + depth
//        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
//        synchronizer.setDelegate(self, queue: synchronizerQueue)
//
//        captureSession.commitConfiguration()
//        isSessionSetup = true
//        return .success(())
//    }
//
//
//
//    func startSession() throws {
//
//        struct SessionNotEnabledError: Error {let msg: String}
//        if !isSessionSetup{
//            throw SessionNotEnabledError(msg: "Must set up session before starting/stopping session")
//        }
//        captureSession.startRunning()
//        isSessionRunning = true
//    }
//
//    func stopSession() {
//        captureSession.stopRunning()
//        isSessionRunning = false
//    }
//}
//
//// MARK: - Delegates
//
//extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate, AVCaptureDepthDataOutputDelegate, AVCaptureDataOutputSynchronizerDelegate {
//
//    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
//        guard let syncedVideo = synchronizedDataCollection.synchronizedData(for: videoOutput) as? AVCaptureSynchronizedSampleBufferData,
//              let syncedDepth = synchronizedDataCollection.synchronizedData(for: depthOutput) as? AVCaptureSynchronizedDepthData else {
//            print("failed to sync video/depth data")
//            return }
//
//        guard !syncedVideo.sampleBufferWasDropped,
//              !syncedDepth.depthDataWasDropped else {
//            print("frame dropped")
//            return }
//
//        let colorFrame = syncedVideo.sampleBuffer
//        let depthFrame: CVPixelBuffer!
//        if syncedDepth.depthData.depthDataType == kCVPixelFormatType_DisparityFloat16 {
//            let depthDataConverted = syncedDepth.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
//            depthFrame = depthDataConverted.depthDataMap
//        }
//        else {
//            depthFrame = syncedDepth.depthData.depthDataMap
//        }
//
//        let timestamp = CMSampleBufferGetPresentationTimeStamp(syncedVideo.sampleBuffer)
//        let ts = Int(timestamp.seconds * 1000)
//
//        if mediapipeManager != nil {
//            guard let lastFrame = try? MPImage(sampleBuffer: colorFrame) else {
//                print("Failed to cast frame into MPImage")
//                return
//            }
//
//            print("Sending frame to mediapipe with ts \(ts) at time \(DispatchTime.now())")
//            do {
//                try mediapipeManager!.handLandmark.recognizeAsync(image: lastFrame, timestampInMilliseconds: ts)
//            } catch {
//                print("Caught mediapipe error", error)
//                return
//            }
//
//            let cameraFrame = IntermediateCameraFrame(rgb: lastFrame, depth: depthFrame, ts: ts)
//
//            Task {await frameFuser!.sendCameraFrame(cameraFrame)}
//        } else if webRTCClient != nil {
//            guard let colorBuffer = CMSampleBufferGetImageBuffer(colorFrame) else {
//                print("Failed to convert colorFrame to colorBuffer")
//                return
//            }
//            webRTCClient!.send(colorBuffer: colorBuffer, depthBuffer: depthFrame)
//        } else {
//            guard let videoBuffer = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer) else {
//                print("Failed to get CVPixelBuffer from image frame")
//                return
//            }
////            CVPixelBufferLockBaseAddress(videoBuffer, .readOnly)
////                CVPixelBufferLockBaseAddress(lastDepthFrame, .readOnly)
////
////                defer {
////                    CVPixelBufferUnlockBaseAddress(videoBuffer, .readOnly)
////                    CVPixelBufferUnlockBaseAddress(lastDepthFrame, .readOnly)
////                }
////
////                let videoData = videoBuffer.extractNV12Data() // helper: converts NV12 pixel buffer to Data
////                let depthData = lastDepthFrame.extractUInt16Data() // helper: converts depth buffer to Data
//
//                // --- Send over USB ---
//                usbSender!.send(videoBuffer: videoBuffer, depthBuffer: depthFrame)
//
//        }
//    }
//}
