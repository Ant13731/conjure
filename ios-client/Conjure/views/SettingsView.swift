//
//  SettingsView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-17.
//
import ARKit
import AVFoundation
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var generalSettings: PersistentSettings<GeneralSettings>
    @EnvironmentObject var hostListSettings: PersistentSettings<HostListSettings>
    @EnvironmentObject var trackpadSettings: PersistentSettings<TrackpadSettings>
    @EnvironmentObject var recognitionSettings: PersistentSettings<RecognitionSettings>
    @EnvironmentObject var router: Router

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                generalSettingsView
                hostListSettingsView
                trackpadSettingsView
                recognitionSettingsView
            }
        }
        .padding(.bottom, 24)
        .navigationTitle("Settings")
    }

    var generalSettingsView: some View {
        configViewBuilder(
            title: "General Settings",
            systemImage: "gearshape.fill",
            appendToPath: .settingsGeneral
        ) {
            VStack(alignment: .leading, spacing: 4) {
                Text("WebRTC Channel Label: \(generalSettings.value.webRTCChannelLabel)")
                Text("Queue Size: \(generalSettings.value.queueSize)")
                Text("Connection Mode: \(generalSettings.value.connectionMode.rawValue)")
                Text("Operation Mode: \(generalSettings.value.operationMode.rawValue)")
            }
        }
    }

    var hostListSettingsView: some View {
        configViewBuilder(
            title: "Host Settings",
            systemImage: "list.bullet",
            appendToPath: .settingsHostList
        ) {
            VStack(alignment: .leading, spacing: 4) {
                if let currentHost = hostListSettings.value.currentHost {
                    if let friendlyName = currentHost.friendlyName, !friendlyName.isEmpty {
                        Text("Friendly Name: \(friendlyName)")
                    }
                    Text("IP Address: \(currentHost.ipAddress)")
                    Text("Port: \(currentHost.port)")
                } else {
                    Text("IP Address: None selected")
                    Text("Port: None selected")
                }
                Text("Number of saved hosts: \(hostListSettings.value.hosts.count)")
            }
        }
    }

    var trackpadSettingsView: some View {
        configViewBuilder(
            title: "Trackpad Settings",
            systemImage: "rectangle.and.hand.point.up.left.filled",
            appendToPath: .settingsTrackpad
        ) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Sensitivity: \(trackpadSettings.value.sensitivity, specifier: "%.2f")")
                Text("Inverted Y-Axis: \(trackpadSettings.value.invertY ? "Yes" : "No")")
                Text("Inverted X-Axis: \(trackpadSettings.value.invertX ? "Yes" : "No")")
            }
        }
    }

    var recognitionSettingsView: some View {
        configViewBuilder(
            title: "Recognition Settings",
            systemImage: "hand.raised.fill",
            appendToPath: .settingsRecognition
        ) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Hands to detect: \(recognitionSettings.value.numHands)")
                Text(
                    "Landmark depth pixel search radius: \(recognitionSettings.value.landmarkDepthPixelRadius)"
                )
                Text(
                    "Minimum depth: \(recognitionSettings.value.minDepth, specifier: "%.2f") meters"
                )
                Text(
                    "Maximum depth: \(recognitionSettings.value.maxDepth, specifier: "%.2f") meters"
                )
            }
        }
    }
    @ViewBuilder
    private func configViewBuilder(
        title: String, systemImage: String, appendToPath: Route, content: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Spacer()
                    Text(title)
                        .font(.headline)
                    Spacer()
                }

                content()
                    .padding(.leading, 8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(.ultraThinMaterial)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(.white.opacity(0.2), lineWidth: 0.5)
            )
            .padding(.horizontal)

            subsettingNavigationEntry(
                title: title,
                systemImage: systemImage,
                appendToPath: appendToPath
            )
        }

    }
    @ViewBuilder
    private func subsettingNavigationEntry(
        title: String,
        systemImage: String,
        appendToPath: Route
    ) -> some View {
        Button {
            router.path.append(appendToPath)
        } label: {
            HStack(spacing: 12) {
                Image(systemName: systemImage)
                Text(title)
                    .font(.body)
                Spacer()
                Image(systemName: "chevron.right")
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.ultraThinMaterial)
            )
            .padding(.horizontal)
        }
    }
}
