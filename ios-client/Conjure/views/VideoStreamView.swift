//
//  VideoStreamView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-17.
//
import ARKit
import AVFoundation
import Combine
import MediaPipeTasksVision
import SwiftUI
import WebRTC

struct VideoStreamView: View {
    var body: some View {
        Color.black
            .overlay(
                Text("Video Stream")
                    .foregroundColor(.white)
            )
    }
}

class CameraPermissionsManager {
    var isCameraAuthorized: Bool = false

    func checkPermissions() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            isCameraAuthorized = true
        case .notDetermined:
            isCameraAuthorized = await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    self.isCameraAuthorized = granted
                    continuation.resume(returning: granted)
                }
            }
        default:
            isCameraAuthorized = false
        }
        return isCameraAuthorized
    }
}

class CameraManager: NSObject, ObservableObject {
    // Must keep these here so they arent deallocated
    private let captureSession = AVCaptureSession()
    private let synchronizerQueue = DispatchQueue(label: "camera.sync.queue")

    // output stream
    private var rgbOut: AVCaptureVideoDataOutput!
    private var depthOut: AVCaptureDepthDataOutput!
    private var synchronizer: AVCaptureDataOutputSynchronizer!

    private let cameraPermissionsManager = CameraPermissionsManager()
    var isSessionSetUp: Bool = false
    var isSessionRunning: Bool = false

    let frameConsumers: [CameraFrameConsumer] = []

    lazy var previewLayer: AVCaptureVideoPreviewLayer = {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        return previewLayer
    }()

    func setupSession() async -> String? {
        if isSessionRunning {
            stopSession()
        }
        isSessionSetUp = false

        if await !cameraPermissionsManager.checkPermissions() {
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

// MARK: - Delegates

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
            consumer.consumeFrame(
                rgbFrame: rgbFrame,
                depthFrame: depthFrame,
                ts: ts
            )
        }
    }

}

class CameraFrameConsumer {
    func consumeFrame(rgbFrame: CMSampleBuffer, depthFrame: CVPixelBuffer, ts: Int) {
        // Override in subclasses
    }
}

// class MediapipeFrameConsumer: CameraFrameConsumer {
//     override func consumeFrame(rgbFrame: CMSampleBuffer, depthFrame: CVPixelBuffer, ts: Int) {
//         guard let mpImage = try? MPImage(sampleBuffer: rgbFrame) else {
//             print("Failed to cast frame into MPImage")
//             return
//         }

//         print("Sending frame to mediapipe with ts \(ts) at time \(DispatchTime.now())")
//         do {
//             try mediapipeManager!.handLandmark.recognizeAsync(
//                 image: mpImage, timestampInMilliseconds: ts)
//         } catch {
//             print("Caught mediapipe error", error)
//             return
//         }

//         // let cameraFrame = IntermediateCameraFrame(rgb: mpImage, depth: depthFrame, ts: ts)
//         // Task { await frameFuser!.sendCameraFrame(cameraFrame) }
//     }
// }

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

struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        previewLayer.frame = uiView.bounds
    }
}

struct FrontCameraView: View {
    @EnvironmentObject var cameraManager: CameraManager

    var body: some View {
        CameraPreviewView(
            // TODO fix this camera streaming not working...
            previewLayer: cameraManager.previewLayer
        )
        .ignoresSafeArea(.all)
    }
}
