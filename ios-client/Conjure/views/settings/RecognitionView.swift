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

                Text(
                    "Skeleton line width: \(recognitionSettings.value.lineWidth, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.lineWidth,
                    in: 1.0...5.0,
                    step: 0.05
                )
                Text(
                    "Skeleton joint radius: \(recognitionSettings.value.jointRadius, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.jointRadius,
                    in: 1.0...10.0,
                    step: 0.05
                )

                Text(
                    "Click depth threshold: \(recognitionSettings.value.clickDepthThreshold, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.clickDepthThreshold,
                    in: 0.0...2.5,
                    step: 0.05
                )
                Text(
                    "Move depth threshold: \(recognitionSettings.value.moveDepthThreshold, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.moveDepthThreshold,
                    in: 0.0...2.5,
                    step: 0.05
                )
                Text(
                    "Click depth limit: \(recognitionSettings.value.clickDepthLimit, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.clickDepthLimit,
                    in: 0.0...2.5,
                    step: 0.05
                )
                Text(
                    "Move depth limit: \(recognitionSettings.value.moveDepthLimit, specifier: "%.2f") m"
                )
                Slider(
                    value: $recognitionSettings.value.moveDepthLimit,
                    in: 0.0...2.5,
                    step: 0.05
                )

                Divider()
                    .padding(.vertical, 8)

                // MARK: - Skeleton Visualization Settings
                Text("Skeleton Visualization")
                    .font(.headline)
                    .padding(.top, 8)

                Toggle("Show Skeleton Lines", isOn: $recognitionSettings.value.showSkeletonLines)
                Toggle("Show Joints", isOn: $recognitionSettings.value.showJoints)
                Toggle("Show Finger Tips", isOn: $recognitionSettings.value.showFingerTips)
                Toggle(
                    "Show Invisible Landmarks",
                    isOn: $recognitionSettings.value.showInvisibleLandmarks)

                Text(
                    "Skeleton Line Width: \(recognitionSettings.value.lineWidth, specifier: "%.2f") px"
                )
                Slider(
                    value: $recognitionSettings.value.lineWidth,
                    in: 0.5...5.0,
                    step: 0.1
                )

                Text(
                    "Joint Radius: \(recognitionSettings.value.jointRadius, specifier: "%.1f") px"
                )
                Slider(
                    value: $recognitionSettings.value.jointRadius,
                    in: 2.0...20.0,
                    step: 0.5
                )

                ColorPickerWidget(
                    title: "Skeleton Line Color",
                    color: $recognitionSettings.value.skeletonLineColor
                )

                Text("Finger Tip Colors")
                    .font(.subheadline)
                    .padding(.top, 4)

                ColorPickerWidget(
                    title: "Finger tip near (click threshold)",
                    color: $recognitionSettings.value.fingerTipColorNear
                )

                ColorPickerWidget(
                    title: "Finger tip far (click limit)",
                    color: $recognitionSettings.value.fingerTipColorFar
                )

                Text("Joint Colors")
                    .font(.subheadline)
                    .padding(.top, 4)

                ColorPickerWidget(
                    title: "Joint near (move threshold)",
                    color: $recognitionSettings.value.jointColorNear
                )

                ColorPickerWidget(
                    title: "Joint far (move limit)",
                    color: $recognitionSettings.value.jointColorFar
                )
            }
        }
        .navigationTitle("Recognition Settings")
    }
}

struct ColorTextWidget: View {
    let title: String
    let color: Color_

    var body: some View {
        HStack {
            Text(title)
                .font(.body)
            Spacer()
            HStack(spacing: 4) {
                Text("R: \(color.red)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("G: \(color.green)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("B: \(color.blue)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                RoundedRectangle(cornerRadius: 4)
                    .fill(SwiftUI.Color(color.toUIColor()))
                    .frame(width: 30, height: 20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(SwiftUI.Color.gray.opacity(0.3), lineWidth: 1)
                    )
            }
        }
    }
}

struct ColorPickerWidget: View {
    let title: String
    @Binding var color: Color_

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ColorTextWidget(title: title, color: $color.wrappedValue)

            VStack(spacing: 4) {
                HStack {
                    Text("R")
                        .frame(width: 20)
                        .foregroundColor(.red)
                    Slider(
                        value: Binding(
                            get: { Double(color.red) },
                            set: { color.red = Int($0) }
                        ),
                        in: 0...255,
                        step: 1
                    )
                    Text("\(color.red)")
                        .frame(width: 35, alignment: .trailing)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                HStack {
                    Text("G")
                        .frame(width: 20)
                        .foregroundColor(.green)
                    Slider(
                        value: Binding(
                            get: { Double(color.green) },
                            set: { color.green = Int($0) }
                        ),
                        in: 0...255,
                        step: 1
                    )
                    Text("\(color.green)")
                        .frame(width: 35, alignment: .trailing)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                HStack {
                    Text("B")
                        .frame(width: 20)
                        .foregroundColor(.blue)
                    Slider(
                        value: Binding(
                            get: { Double(color.blue) },
                            set: { color.blue = Int($0) }
                        ),
                        in: 0...255,
                        step: 1
                    )
                    Text("\(color.blue)")
                        .frame(width: 35, alignment: .trailing)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.leading, 8)
        }
        .padding(.vertical, 4)
    }
}
