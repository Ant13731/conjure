//
//  StorableSettings.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-20.
//

import AVFoundation
import Combine
import SwiftUI

// MARK: - General Settings
/// Possible connection modes for video streaming and processing
enum ConnectionMode: String, CaseIterable, Identifiable, Codable {
    case onDevice = "On-device ML"
    case streamWebRTC = "WebRTC Video"
    case streamTCP = "TCP Stream"
    case streamUDP = "UDP Stream"

    var id: String { self.rawValue }
    var description: String {
        switch self {
        case .onDevice:
            return
                "Process video frames on the device using on-device ML models. Data is transferred to the server via WebRTC"
        case .streamWebRTC:
            return "Stream video frames to the server using WebRTC."
        case .streamTCP:
            return
                "Stream video frames to the server using TCP (preferably used with a USB connection)."
        case .streamUDP:
            return
                "Stream video frames to the server using UDP (preferably used with a USB connection)."
        }
    }
    static var defaultValue = ConnectionMode.onDevice
}
enum OperationMode: String, CaseIterable, Identifiable, Codable {
    case trackpad = "Trackpad"
    case handRecognition = "Hand Recognition"
    case handRecognitionDemoMode = "Hand Recognition (Demo Mode)"

    var id: String { self.rawValue }
    var description: String {
        switch self {
        case .trackpad:
            return
                "Device acts as a trackpad, sending touch inputs to the server."
        case .handRecognition:
            return
                "Device uses hand recognition to interpret gestures and send inputs to the server."
        case .handRecognitionDemoMode:
            return
                "Device uses hand recognition in demo mode, which displays a skeleton preview on the device, but does not send any data to the server."
        }
    }
    static var defaultValue = OperationMode.trackpad
}

/// Misc. connection and operation settings, including webRTC channel label, queue size, etc.
struct GeneralSettings: PersistentlyStorable {
    var webRTCStreamChannelLabel: String
    var webRTCSettingsChannelLabel: String
    var queueSize: Int  // TODO apply to frame fuser
    var connectionMode: ConnectionMode
    var operationMode: OperationMode

    static var defaultValue = GeneralSettings(
        webRTCStreamChannelLabel: "stream",
        webRTCSettingsChannelLabel: "settings",
        queueSize: 1,
        connectionMode: .defaultValue,
        operationMode: .defaultValue,
    )
    static var storageKey = "generalSettings"
}

// MARK: - Host Settings
/// Server configurations to store known hosts
struct HostSettings: Identifiable, Codable, Equatable {
    let id: UUID
    var ipAddress: String
    var port: String
    var friendlyName: String?

    init(id: UUID = UUID(), ipAddress: String, port: String, friendlyName: String? = nil) {
        self.id = id
        self.ipAddress = ipAddress
        self.port = port
        self.friendlyName = friendlyName
    }
}

struct HostListSettings: PersistentlyStorable {
    var hosts: [HostSettings]
    var currentHost: HostSettings?

    static var defaultValue = HostListSettings(hosts: [], currentHost: nil)
    static var storageKey = "hostListSettings"
}

// MARK: - Trackpad Settings
struct TrackpadSettings: PersistentlyStorable {
    var sensitivity: Float
    var invertY: Bool
    var invertX: Bool

    static var defaultValue = TrackpadSettings(
        sensitivity: 1.0,
        invertY: false,
        invertX: false
    )
    static var storageKey = "trackpadSettings"
}

// MARK: - Recognition Settings
struct RecognitionSettings: PersistentlyStorable {
    var numHands: Int
    var landmarkDepthPixelRadius: Int
    var minDepth: Float  //TODO do we need min/max depth here? Especially for on device ML?
    var maxDepth: Float

    var lineWidth: Float
    var jointRadius: Float
    var fingerTipColorNear: Color_
    var fingerTipColorFar: Color_
    var jointColorNear: Color_
    var jointColorFar: Color_
    var skeletonLineColor: Color_
    var clickDepthThreshold: Float
    var moveDepthThreshold: Float
    var clickDepthLimit: Float
    var moveDepthLimit: Float

    // Skeleton visualization toggles
    var showSkeletonLines: Bool
    var showJoints: Bool
    var showFingerTips: Bool
    var showInvisibleLandmarks: Bool

    var showCameraPreview: Bool

    static var defaultValue = RecognitionSettings(
        numHands: 1,
        landmarkDepthPixelRadius: 2,
        minDepth: 0.1,
        maxDepth: 1.5,
        lineWidth: 4.0,
        jointRadius: 12.0,
        fingerTipColorNear: Color_(red: 0, green: 200, blue: 255),
        fingerTipColorFar: Color_(red: 0, green: 40, blue: 50),
        jointColorNear: Color_(red: 255, green: 255, blue: 255),
        jointColorFar: Color_(red: 0, green: 0, blue: 0),
        skeletonLineColor: Color_(red: 255, green: 255, blue: 255),
        clickDepthThreshold: 0.3,
        moveDepthThreshold: 0.5,
        clickDepthLimit: 0,
        moveDepthLimit: 1.1,
        showSkeletonLines: true,
        showJoints: true,
        showFingerTips: true,
        showInvisibleLandmarks: false,
        showCameraPreview: false,
    )
    static var storageKey = "recognitionSettings"
}

struct Color_: PersistentlyStorable {
    var red: Int
    var green: Int
    var blue: Int

    static var defaultValue = Color_(red: 255, green: 255, blue: 255)
    static var storageKey = "colourSettings"

    func toUIColor() -> Color {
        return Color(
            red: Double(red) / 255.0,
            green: Double(green) / 255.0,
            blue: Double(blue) / 255.0,
        )
    }

    func fromUIColor(_ color: Color) -> Color_ {
        let uiColor = UIColor(color)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        uiColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return Color_(
            red: Int(red * 255),
            green: Int(green * 255),
            blue: Int(blue * 255),
        )
    }

    /// Interpolates between two colors based on depth and threshold/limit values
    static func interpolateColor(
        near: Color_,
        far: Color_,
        depth: Float,
        threshold: Float,
        limit: Float
    ) -> Color {
        // If closer than threshold, use near color
        if depth <= threshold {
            return near.toUIColor()
        }

        // If farther than limit, use far color
        if depth >= limit {
            return far.toUIColor()
        }

        // Interpolate between threshold and limit
        let progress = Double((depth - threshold) / (limit - threshold))

        let r = Double(near.red) + (Double(far.red) - Double(near.red)) * progress
        let g = Double(near.green) + (Double(far.green) - Double(near.green)) * progress
        let b = Double(near.blue) + (Double(far.blue) - Double(near.blue)) * progress

        return Color(
            red: r / 255.0,
            green: g / 255.0,
            blue: b / 255.0
        )
    }

}
