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

struct CameraPreviewView: UIViewRepresentable {
    let previewLayer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        let view = PreviewLayerContainerView()
        view.setPreviewLayer(previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        guard let containerView = uiView as? PreviewLayerContainerView else { return }
        containerView.layoutIfNeeded()
    }
}

class PreviewLayerContainerView: UIView {
    private weak var previewLayer: AVCaptureVideoPreviewLayer?

    func setPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) {
        self.previewLayer = layer
        layer.videoGravity = .resizeAspect
        self.layer.addSublayer(layer)
        updateVideoOrientation()
        startOrientationMonitoring()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
        updateVideoOrientation()
    }

    private func startOrientationMonitoring() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
    }

    @objc private func orientationDidChange() {
        updateVideoOrientation()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    private func updateVideoOrientation() {
        guard let connection = previewLayer?.connection else {
            print("Failed to get preview layer connection")
            return
        }

        let deviceOrientation = UIDevice.current.orientation
        let videoRotationAngle: CGFloat

        switch deviceOrientation {
        case .portrait:
            videoRotationAngle = 90
        case .landscapeLeft:
            videoRotationAngle = 180
        case .landscapeRight:
            videoRotationAngle = 0
        default:
            return
        }

        if connection.isVideoRotationAngleSupported(videoRotationAngle) {
            connection.videoRotationAngle = videoRotationAngle
            print(
                "Preview orientation updated: device=\(deviceOrientation.rawValue) → angle=\(videoRotationAngle)°"
            )
        }
    }
}

struct FrontCameraView: View {
    @EnvironmentObject var cameraManager: CameraManager
    @EnvironmentObject var recognitionSettings: PersistentSettings<RecognitionSettings>
    @EnvironmentObject var skeletonOverlayConsumer: SkeletonOverlayFusedFrameConsumer

    let frameFuser: FrameFuser

    var body: some View {
        ZStack {
            CameraPreviewView(
                previewLayer: cameraManager.previewLayer
            )
            .ignoresSafeArea(.all)

            SkeletonOverlayView(skeletonConsumer: skeletonOverlayConsumer)
                .environmentObject(recognitionSettings)
        }
    }
}
