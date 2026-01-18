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
    @EnvironmentObject var connectionConfig: ConnectionConfigStore
    @EnvironmentObject var recognitionConfig: RecognitionConfigStore

    // Debug/functional vars
    @State private var connectionMessage: String = ""
    @State private var debugErrorMessage: String = ""
    @State private var isConnected: Bool = false
    @State private var isStreaming: Bool = false

    // Header information
    var displayStatus: String {
        if isConnected {
            if isStreaming {
                return "Streaming"
            }
            return "Streaming paused"
        }
        return "Not connected"
    }

    var body: some View {
        ZStack {
            // MARK: Background Camera View
            backgroundView.ignoresSafeArea()

            // MARK: Foreground Overlay
            VStack {
                header
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
            // MARK: Control buttons
            HStack {
                Spacer()
                VStack {
                    Spacer()
                    connectButton
                    enableCameraButton
                    settingsButton
                }
            }

        }
    }

    @ViewBuilder
    var backgroundView: some View {
        if isConnected && isStreaming {
            VideoStreamView()
        } else {
            Color.black
        }
    }
    var header: some View {
        VStack {
            Text("Status: " + displayStatus)
            if let currentHost = connectionConfig.currentHostConfig {
                Text(currentHost.ipAddress + ":" + currentHost.port)
            }
        }
        .font(.subheadline.monospaced())
        .foregroundColor(.white)
        .padding(.vertical, 8)
        .background(
            Color.black.opacity(0.4)
                .blur(radius: 10)
        )
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
                .font(.system(size: 22))
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
    var enableCameraButton: some View {
        actionButtonBuilder(
            systemImage: "video.fill",
            isActive: isStreaming,
        ) {
            // TODO: See if we should check for a connection before allowing streaming, otherwise show error
            isStreaming.toggle()
        }
    }
    var settingsButton: some View {
        actionButtonBuilder(
            systemImage: "gearshape.fill", isActive: false,
            action: {
                router.path.append(Route.settings)
            })

    }
}
