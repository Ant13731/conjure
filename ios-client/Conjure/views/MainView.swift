//
//  LoginView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-10-26.
//

import ARKit
import AVFoundation
import SwiftUI
import WebRTC

/// General idea: camera shows in the background, with a separate settings view
/// TODO: Implement a trackpad mode (need to export this to another view)
struct MainView: View {
    // State objs
    @EnvironmentObject var router: Router
    @EnvironmentObject var generalSettings: PersistentSettings<GeneralSettings>
    @EnvironmentObject var hostListSettings: PersistentSettings<HostListSettings>
    @EnvironmentObject var trackpadSettings: PersistentSettings<TrackpadSettings>
    @EnvironmentObject var recognitionSettings: PersistentSettings<RecognitionSettings>

    @StateObject private var cameraManager = CameraManager()
    private let frameFuser = FrameFuser()

    @State private var mediapipeManager: MediapipeManager?
    @State private var skeletonOverlayConsumer: SkeletonOverlayFusedFrameConsumer?
    @State private var webRTCClient: WebRTCClient?

    // Debug/functional vars
    @State private var connectionMessage: String = ""
    @State private var debugErrorMessage: String = ""
    @State private var isConnected: Bool = false
    @State private var isStreaming: Bool = false
    @State private var isProcessingStreamChange: Bool = false
    @State private var isPipelineSetup = false

    // Header information
    var displayStatus: String {
        if !isConnected {
            return "Not connected"
        }
        if !isStreaming {
            return "Streaming paused"
        }
        return "Streaming"
    }

    var body: some View {
        ZStack {
            // MARK: Background Camera View
            // TODO if in trackpad mode, this is a plain background. if in camera mode, show mediapipe frames
            backgroundView.ignoresSafeArea()

            // MARK: Messages and Foreground overlay
            VStack {
                Spacer()
                if !isConnected || !isStreaming {
                    disconnectedOverlay
                }
                Spacer()
                if !debugErrorMessage.isEmpty {
                    debugMessage
                }
                Spacer()

            }

            // MARK: Status and Action Buttons
            VStack {
                HStack {
                    header
                    Spacer()
                    connectButton
                    enableStreamButton
                    settingsButton
                }
                Spacer()
                // MARK: Trackpad

                if generalSettings.value.operationMode == .trackpad {
                    ZStack {
                        // After some inactivity (or on error), fade out the trackpad view
                        visibleTrackpadView
                        // TODO add invisible trackpad sensor layer
                    }
                }
            }

        }
        .onChange(of: generalSettings.value.operationMode) { _ in
            stopStreaming()
        }
        .onAppear {
            if webRTCClient != nil {
                print("Sending config update down WebRTC channel")
                webRTCClient!.sendConfigUpdate()
            }
        }
    }

    @ViewBuilder
    var backgroundView: some View {
        if isConnected && isStreaming
            && (generalSettings.value.operationMode == .handRecognition
                || generalSettings.value.operationMode == .handRecognitionDemoMode)
        {
            FrontCameraView()
                .environmentObject(cameraManager)
                .environmentObject(skeletonOverlayConsumer!)
        } else {
            Color.black
        }
    }
    var visibleTrackpadView: some View {
        GeometryReader { geo in
            VStack {
                Spacer()
                RoundedRectangle(cornerRadius: 32, style: .continuous)
                    .fill(.ultraThinMaterial.opacity(0.5))
                    .frame(height: geo.size.height * 0.97)
                    .frame(maxWidth: .infinity)
                    .overlay(
                        RoundedRectangle(cornerRadius: 32, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [
                                        Color.white.opacity(0.25),
                                        Color.white.opacity(0.05),
                                        Color.clear,
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                    ).shadow(
                        color: Color.black.opacity(0.15),
                        radius: 20,
                        y: 8
                    )

            }
            .ignoresSafeArea(edges: .bottom)
        }
    }
    var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(displayStatus)
                if let currentHost = hostListSettings.value.currentHost {
                    Text("\(currentHost.ipAddress):\(currentHost.port)")
                    if let friendlyName = currentHost.friendlyName, !friendlyName.isEmpty {
                        Text(friendlyName)
                    }
                }
            }
            Spacer()
        }
        .font(.footnote.monospaced())
        .foregroundColor(.white)
        .padding(8)
        .background(.ultraThinMaterial.opacity(0.8))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
    var disconnectedOverlay: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundColor(.white)

            Text(displayStatus)
                .font(.headline)
                .foregroundColor(.white)

            if !connectionMessage.isEmpty {
                Text(connectionMessage)
                    .font(.subheadline)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.black.opacity(0.6))
                .blur(radius: 1)
        )
    }
    var debugMessage: some View {
        Text("Debug: " + debugErrorMessage)
            .font(.caption)
            .foregroundColor(.red)
            .padding(8)
            .background(
                Color.white.opacity(0.8)
                    .blur(radius: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
    @ViewBuilder
    private func actionButtonBuilder(
        systemImage: String,
        isActive: Bool,
        activeColor: Color = .green,
        action: @escaping () -> Void
    ) -> some View {
        Button {
            action()
        } label: {
            Image(systemName: systemImage)
                .font(.system(size: 20))
                .padding()
                .foregroundStyle(isActive ? activeColor : .white)
                .background(.ultraThinMaterial)
                .clipShape(Circle())
                .shadow(radius: 4)
        }
    }
    var connectButton: some View {
        actionButtonBuilder(
            systemImage: "dot.radiowaves.left.and.right",
            isActive: isConnected,
        ) {
            if generalSettings.value.operationMode == .handRecognitionDemoMode {
                print("Connect button demo mode: toggling")
                isConnected.toggle()
                return
            }

            if isConnected {
                print("Connect button: stopping connection")
                stopConnection()
                return
            }

            print("Connect button: starting connection")
            startConnection()
        }
    }
    var enableStreamButton: some View {
        actionButtonBuilder(
            systemImage: "record.circle",
            isActive: isStreaming,
        ) {

            guard !isProcessingStreamChange else {
                print("Already processing stream change, ignoring repeated button press")
                return
            }

            isProcessingStreamChange = true
            Task {
                defer { isProcessingStreamChange = false }

                if !isStreaming {
                    await startStreaming()

                } else {
                    stopStreaming()
                }
            }
        }.disabled(isProcessingStreamChange)
    }
    var settingsButton: some View {
        actionButtonBuilder(
            systemImage: "gearshape.fill", isActive: false,
            action: {
                stopStreaming()
                router.path.append(Route.settings)
            })

    }
}

// MARK: - Streaming helpers
extension MainView {
    fileprivate func stopConnection() {
        if generalSettings.value.connectionMode == .webRTC {
            webRTCClient?.stopConnection()
            webRTCClient = nil
            frameFuser.clearFusedFrameConsumers()
        }
        isConnected = false
        connectionMessage = "Disconnected"
    }

    fileprivate func startConnection() {
        if generalSettings.value.connectionMode == .webRTC {
            isConnected = false
            let webRTCClient_ = WebRTCClient(
                generalSettings: generalSettings,
                hostListSettings: hostListSettings,
                trackpadSettings: trackpadSettings,
                recognitionSettings: recognitionSettings,
            )
            webRTCClient = webRTCClient_

            if generalSettings.value.operationMode == .handRecognition
                || generalSettings.value.operationMode == .handRecognitionDemoMode
            {
                print("Adding WebRTCClientFusedFrameConsumer to frame fuser")
                let fusedFrameConsumer = WebRTCClientFusedFrameConsumer(rtcClient: webRTCClient_)
                frameFuser.addFusedFrameConsumer(fusedFrameConsumer)
            }

            Task {
                print("Start connection: creating offer")
                let result = await webRTCClient_.createOffer()
                if case .failure(let errMsg) = result {
                    connectionMessage = "Connection Result: \(errMsg)"
                    return
                }
                let offer = try! result.get()

                print("Start connection: sending offer")
                let res = await webRTCClient_.sendOffer(offer)
                if case .failure(let errMsg) = res {
                    connectionMessage = "Connection Result: \(errMsg)"
                    return
                }
                let answer = try! res.get()

                print("Start connection: adding answer")
                if let errMsg = await webRTCClient_.addAnswer(answer) {
                    connectionMessage = "Connection Result: \(errMsg)"
                    return
                }

                print("Sending config update down WebRTC channel")
                webRTCClient_.sendConfigUpdate()
                isConnected = true
            }
            return
        }
        print("Connect button not yet implemented for this connection mode")
    }

    fileprivate func stopStreaming() {
        // TODO: Handle streaming for trackpads
        if generalSettings.value.operationMode == .handRecognition
            || generalSettings.value.operationMode == .handRecognitionDemoMode
        {
            if isStreaming {
                cameraManager.stopSession()
                isStreaming = false
            }

            // Reset pipeline in case settings have changed
            if isPipelineSetup {
                resetHandRecognitionProcessingPipeline()
            }
        }
    }

    fileprivate func startStreaming() async {
        // TODO: See if we should check for a connection before allowing streaming, otherwise show error
        // TODO: Handle streaming for trackpads

        if generalSettings.value.operationMode == .handRecognition
            || generalSettings.value.operationMode == .handRecognitionDemoMode
        {
            // Set up the processing pipeline if not already done
            if !isPipelineSetup {
                setupHandRecognitionProcessingPipeline()
                isPipelineSetup = true
            }

            if !cameraManager.isSessionSetUp {
                let setupMessage = await cameraManager.setupSession()
                if let msg = setupMessage {
                    debugErrorMessage = msg
                    return
                }
            }

            let startMessage = cameraManager.startSession()
            if let msg = startMessage {
                debugErrorMessage = msg
                return
            }
            isStreaming = true
        }

    }

    private func setupHandRecognitionProcessingPipeline() {
        // Create MediaPipe manager
        let mediapipe = MediapipeManager(recognitionSettings: recognitionSettings)
        mediapipe.addFrameFuser(frameFuser)
        mediapipeManager = mediapipe

        let skeletonOverlay = SkeletonOverlayFusedFrameConsumer()
        skeletonOverlayConsumer = skeletonOverlay

        // Create and register frame consumers
        let mediapipeConsumer = MediapipeFrameConsumer(mediapipeManager: mediapipe)
        let rgbConsumer = RGBFrameConsumer(frameFuser: frameFuser)

        cameraManager.addConsumer(mediapipeConsumer)
        cameraManager.addConsumer(rgbConsumer)

        frameFuser.addFusedFrameConsumer(skeletonOverlay)
    }
    private func resetHandRecognitionProcessingPipeline() {
        cameraManager.clearConsumers()
        frameFuser.clearFusedFrameConsumers()
        mediapipeManager = nil
        isPipelineSetup = false
    }
}
