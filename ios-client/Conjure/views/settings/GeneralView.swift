//
//  GeneralView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-20.
//
import ARKit
import AVFoundation
import SwiftUI

struct GeneralView: View {
    @EnvironmentObject var generalSettings: PersistentSettings<GeneralSettings>

    var body: some View {
        Form {
            VStack(alignment: .leading) {
                Text(
                    "WebRTC Stream Channel Label (unchangable): \(generalSettings.value.webRTCStreamChannelLabel)",
                )
                Text(
                    "WebRTC Settings Channel Label (unchangable): \(generalSettings.value.webRTCSettingsChannelLabel)",
                )

                Text(
                    "Queue size: \(generalSettings.value.queueSize)",
                )
                Slider(
                    value: .convert($generalSettings.value.queueSize),
                    in: 1.0...10.0,
                    step: 1.0
                )

                Picker(
                    "Connection mode",
                    selection: $generalSettings.value.connectionMode
                ) {
                    ForEach(ConnectionMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }

                .pickerStyle(.menu)

                Picker(
                    "Operation mode",
                    selection: $generalSettings.value.operationMode
                ) {
                    ForEach(OperationMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.menu)
            }
        }
        .navigationTitle("General Settings")
    }
}
