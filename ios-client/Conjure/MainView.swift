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
struct MainView: View {
    // Debug/functional vars
    @State private var debugErrorMessage: String = ""
    @State private var connectionMessage: String = ""
    @State private var isConnected: Bool = false
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

    var body: some View {
        ZStack {
            // MARK: Background Camera View
            backgroundView.ignoresSafeArea()

            // MARK: Foreground Overlay
            VStack {
                header
                Spacer()
                if !isConnected {
                    disconnectedOverlay
                    Spacer()
                }
                if !debugErrorMessage.isEmpty {
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
            }
            .padding()
        }
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                settingsButton
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
        VStack(alignment: .leading) {
            Text("Status: " + displayStatus)
            Spacer()
            if isConnected {
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

            Text(connectionMessage)
                .font(.subheadline)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.black.opacity(0.6))
                .blur(radius: 1)
        )
    }
    var settingsButton: some View {
        NavigationLink {
            SettingsView()
        } label: {
            Image(systemName: "gearshape.fill")
                .foregroundColor(.white)
        }
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
