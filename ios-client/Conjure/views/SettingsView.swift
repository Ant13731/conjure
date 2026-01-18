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
    @EnvironmentObject var router: Router
    @EnvironmentObject var connectionConfigStore: ConnectionConfigStore
    @EnvironmentObject var recognitionConfigStore: RecognitionConfigStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                currentConnectionConfigView
                recognitionConfigView
                VStack(spacing: 12) {
                    subsettingNavigationEntry(
                        title: "Host Configurations",
                        systemImage: "list.bullet",
                        appendToPath: .settingsHostList
                    )
                    subsettingNavigationEntry(
                        title: "Recognition Settings",
                        systemImage: "hand.raised.fill",
                        appendToPath: .settingsRecognition
                    )
                }
            }
        }

        // Add a list of settings
        // Open Host list view
        // Open Recognition settings view
        .padding(.bottom, 24)
        .navigationTitle("Settings")
    }

    var currentConnectionConfigView: some View {
        configViewBuilder(title: "Connection Configuration") {
            VStack(alignment: .leading, spacing: 4) {
                if let currentHost = connectionConfigStore.currentHostConfig {
                    if let friendlyName = currentHost.friendlyName, !friendlyName.isEmpty {
                        Text("Friendly Name: \(friendlyName)")
                    }
                    Text("IP Address: \(currentHost.ipAddress)")
                    Text("Port: \(currentHost.port)")
                } else {
                    Text("IP Address: None selected")
                    Text("Port: None selected")
                }

                Text("Mode: \(connectionConfigStore.connectionConfig.mode.rawValue)")
            }
        }
    }
    var recognitionConfigView: some View {
        configViewBuilder(title: "Recognition Configuration") {
            VStack(alignment: .leading, spacing: 4) {
                Text("Hands to detect: \(recognitionConfigStore.recognitionConfig.numHands)")
                Text(
                    "Landmark depth pixel search radius: \(recognitionConfigStore.recognitionConfig.landmarkDepthPixelRadius)"
                )
                Text(
                    "Minimum depth: \(recognitionConfigStore.recognitionConfig.minDepth, specifier: "%.2f") meters"
                )
                Text(
                    "Maximum depth: \(recognitionConfigStore.recognitionConfig.maxDepth, specifier: "%.2f") meters"
                )
            }
        }
    }
    @ViewBuilder
    private func configViewBuilder(title: String, content: () -> some View) -> some View {
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
        }
    }
}
