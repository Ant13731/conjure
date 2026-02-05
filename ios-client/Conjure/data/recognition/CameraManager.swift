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
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?

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

    func clearConsumers() {
        frameConsumers.removeAll()
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

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
        isSessionRunning = true

        // Observe device orientation changes to update rotation
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(deviceOrientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )

        return nil
    }

    @objc private func deviceOrientationDidChange() {
        updateVideoRotation()
    }

    private func updateVideoRotation() {
        guard let coordinator = rotationCoordinator else { return }
        let angle = coordinator.videoRotationAngleForHorizonLevelCapture

        // Update both RGB and depth rotation to keep them synchronized
        if let rgbConnection = rgbOut?.connection(with: .video) {
            rgbConnection.videoRotationAngle = angle
        }
        if let depthConnection = depthOut?.connection(with: .depthData) {
            depthConnection.videoRotationAngle = angle
        }
    }

    func stopSession() {
        NotificationCenter.default.removeObserver(
            self, name: UIDevice.orientationDidChangeNotification, object: nil)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.stopRunning()
        }
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

        rotationCoordinator = AVCaptureDevice.RotationCoordinator(device: camera, previewLayer: nil)

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

        // Configure initial rotation for RGB
        if let coordinator = rotationCoordinator,
            let connection = rgbOut.connection(with: .video)
        {
            connection.videoRotationAngle = coordinator.videoRotationAngleForHorizonLevelCapture
        }

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

        // Configure initial rotation for depth to match RGB
        if let coordinator = rotationCoordinator,
            let connection = depthOut.connection(with: .depthData)
        {
            connection.videoRotationAngle = coordinator.videoRotationAngleForHorizonLevelCapture
        }

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
        let orientation = UIDeviceOrientation_(from: UIDevice.current.orientation)
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
                    ts: ts,
                    orientation: orientation
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
