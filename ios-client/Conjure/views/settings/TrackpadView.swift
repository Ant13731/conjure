//
//  TrackpadView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-20.
//
import ARKit
import AVFoundation
import SwiftUI

struct TrackpadView: View {
    @EnvironmentObject var trackpadSettings: PersistentSettings<TrackpadSettings>

    var body: some View {
        Form {
            VStack(alignment: .leading) {
                Text(
                    "Sensitivity: \(trackpadSettings.value.sensitivity, specifier: "%.2f")",
                )
                Slider(
                    value: $trackpadSettings.value.sensitivity,
                    in: 0.0...10.0,
                    step: 0.1
                )

                Toggle(
                    "Inverted Y-Axis",
                    isOn: $trackpadSettings.value.invertY
                )
                Toggle(
                    "Inverted X-Axis",
                    isOn: $trackpadSettings.value.invertX
                )
            }
        }
        .navigationTitle("Trackpad Settings")
    }
}
