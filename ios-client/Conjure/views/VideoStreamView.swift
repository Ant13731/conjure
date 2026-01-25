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
        layer.videoGravity = .resizeAspectFill
        self.layer.addSublayer(layer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

struct FrontCameraView: View {
    @EnvironmentObject var cameraManager: CameraManager

    var body: some View {
        CameraPreviewView(
            previewLayer: cameraManager.previewLayer
        )
        .ignoresSafeArea(.all)
    }
}
