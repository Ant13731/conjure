//
//  StorableSettings.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-20.
//

import AVFoundation
import Combine
import UIKit

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

    var id: String { self.rawValue }
    var description: String {
        switch self {
        case .trackpad:
            return
                "Device acts as a trackpad, sending touch inputs to the server."
        case .handRecognition:
            return
                "Device uses hand recognition to interpret gestures and send inputs to the server."
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
        operationMode: .defaultValue
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

    var lineWidth: Float = 2.0
    var jointRadius: Float = 4.0
    var fingerTipColorNear: UIColor
    var fingerTipColorFar: UIColor
    var jointColorNear: UIColor
    var jointColorFar: UIColor
    var clickDepthThreshold: Float
    var moveDepthThreshold: Float
    var clickDepthLimit: Float
    var moveDepthLimit: Float

    static var defaultValue = RecognitionSettings(
        numHands: 1,
        landmarkDepthPixelRadius: 2,
        minDepth: 0.1,
        maxDepth: 1.5,
        lineWidth: 2.0,
        jointRadius: 4.0,
        fingerTipColorNear: UIColor(red: 50, green: 40, blue: 0, alpha: 1.0),
        fingerTipColorFar: UIColor(red: 255, green: 200, blue: 0, alpha: 1.0),
        jointColorNear: UIColor(red: 255, green: 255, blue: 255, alpha: 1.0),
        jointColorFar: UIColor(red: 0, green: 0, blue: 0, alpha: 1.0),
        clickDepthThreshold: 0.3,
        moveDepthThreshold: 0.5,
        clickDepthLimit: 0,
        moveDepthLimit: 1.1,
    )
    static var storageKey = "recognitionSettings"
}
