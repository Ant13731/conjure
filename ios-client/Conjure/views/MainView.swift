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

    // Debug/functional vars
    @State private var connectionMessage: String = ""
    @State private var debugErrorMessage: String = ""
    @State private var isConnected: Bool = false
    @State private var isStreaming: Bool = false
    @State private var isProcessingStreamChange: Bool = false

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
                    ZStack{
                        // After some inactivity (or on error), fade out the trackpad view
                        visibleTrackpadView
                        // TODO add invisible trackpad sensor layer
                    }
                }
                // TODO if in camera mode, hide trackpad. If in trackpad mode, show trackpad overlay. this sould take up at least 75%-85% of the screen
            }

        }
        .onChange(of: generalSettings.value.operationMode) { _ in
            stopStreaming()
        }
    }

    @ViewBuilder
    var backgroundView: some View {
        if isConnected && isStreaming && generalSettings.value.operationMode == .handRecognition {
            // VideoStreamView()
            FrontCameraView()
                .environmentObject(cameraManager)
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
                                        Color.clear
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
        HStack{
            VStack(alignment: .leading, spacing: 4) {
                Text("Status:\n  \(displayStatus)")
                if let currentHost = hostListSettings.value.currentHost {
                    Text("  \(currentHost.ipAddress):\(currentHost.port)")
                    if let friendlyName = currentHost.friendlyName, !friendlyName.isEmpty {
                        Text("  \(friendlyName)")
                    }
                }
            }
            Spacer()
        }
        .font(.subheadline.monospaced())
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
            if !isConnected {
                isConnected = true
            } else {
                // TODO: If it is already connected, should we retry connection or just disconnect?
                isConnected = false
            }
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
                defer {isProcessingStreamChange = false}

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
private extension MainView {
    func stopStreaming() {
        // TODO: Handle streaming for trackpads
        if generalSettings.value.operationMode == .handRecognition {
            if isStreaming {
                cameraManager.stopSession()
                isStreaming = false
            }
        }
    }

    func startStreaming() async {
        // TODO: See if we should check for a connection before allowing streaming, otherwise show error
        // TODO: Handle streaming for trackpads

        if generalSettings.value.operationMode == .handRecognition {
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
}
