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

enum Route: Hashable {
    case settings
}

/// General idea: camera shows in the background, with a separate settings view
struct MainView: View {
    // Debug/functional vars
    @State private var connectionMessage: String = "test12"
    @State private var debugErrorMessage: String = "test1"
    @State private var isConnected: Bool = true
    @State private var isStreaming: Bool = false

    // Header information
    @State private var displayConnectedIP: String = ""
    @State private var displayConnectedPort: String = ""
    var displayStatus: String {
        if isConnected {
            if isStreaming {
                return "Streaming"
            }
            return "Streaming paused"
        }
        return "Not connected"
    }

    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
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
                    HStack {
                        Spacer()
                        settingsButton
                    }
                }
                .padding()
            }
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .settings:
                    SettingsView()
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
            if !displayConnectedIP.isEmpty && !displayConnectedPort.isEmpty {
                Text(displayConnectedIP + ":" + displayConnectedPort)
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
    var settingsButton: some View {
        Button {
            path.append(Route.settings)
        } label: {
            Image(systemName: "gearshape.fill")
                .font(.system(size: 22))
                .padding()
                .foregroundStyle(.white)
                .background(.ultraThinMaterial)
                .clipShape(Circle())
                .shadow(radius: 4)
        }
        .padding()
    }
}

// Stubs
struct VideoStreamView: View {
    var body: some View {
        Color.black
            .overlay(
                Text("Video Stream")
                    .foregroundColor(.white)
            )
    }
}

struct SettingsView: View {
    var body: some View {
        Text("Settings")
            .navigationTitle("Settings")
    }
}
