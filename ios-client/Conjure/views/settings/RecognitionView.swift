//
//  RecognitionConfigView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-17.
//
import ARKit
import AVFoundation
import SwiftUI

extension Binding {

    public static func convert<TInt, TFloat>(_ intBinding: Binding<TInt>) -> Binding<TFloat>
    where
        TInt: BinaryInteger,
        TFloat: BinaryFloatingPoint
    {

        Binding<TFloat>(
            get: { TFloat(intBinding.wrappedValue) },
            set: { intBinding.wrappedValue = TInt($0) }
        )
    }

    public static func convert<TFloat, TInt>(_ floatBinding: Binding<TFloat>) -> Binding<TInt>
    where
        TFloat: BinaryFloatingPoint,
        TInt: BinaryInteger
    {

        Binding<TInt>(
            get: { TInt(floatBinding.wrappedValue) },
            set: { floatBinding.wrappedValue = TFloat($0) }
        )
    }
}

struct RecognitionView: View {
    @EnvironmentObject var recognitionSettings: PersistentSettings<RecognitionSettings>

    var body: some View {
        Form {
            VStack(alignment: .leading) {
                Text(
                    "Hands to detect: \(recognitionSettings.value.numHands)",
                )
                Slider(
                    value: .convert($recognitionSettings.value.numHands),
                    in: 1.0...2.0,
                    step: 1.0
                )

                Text(
                    "Landmark depth pixel search radius: \(recognitionSettings.value.landmarkDepthPixelRadius)",
                )
                Slider(
                    value: .convert(
                        $recognitionSettings.value.landmarkDepthPixelRadius),
                    in: 0.0...10.0,
                    step: 1.0
                )

                Text(
                    "Min depth: \(recognitionSettings.value.minDepth, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.minDepth,
                    in: 0.0...1.0,
                    step: 0.05
                )

                Text(
                    "Max depth: \(recognitionSettings.value.maxDepth, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.maxDepth,
                    in: 1.0...5.0,
                    step: 0.05
                )
            }
        }
        .navigationTitle("Recognition Settings")
    }
}
