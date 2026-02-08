//
//  CommunicationManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-02-08.
//
import ARKit
import AVFoundation
import Accelerate
import SwiftUI
import WebRTC

enum PacketType: UInt8, Codable {
    case stream = 0
    case configUpdate = 1
    case ack = 2
}

struct SendableConfigUpdate: Codable {
    let generalSettings: GeneralSettings
    let hostListSettings: HostListSettings
    let trackpadSettings: TrackpadSettings
    let recognitionSettings: RecognitionSettings
}

class CommunicationManager {
    var isConnected: Bool = false

    unowned let generalSettings: PersistentSettings<GeneralSettings>!
    unowned let hostListSettings: PersistentSettings<HostListSettings>!
    unowned let trackpadSettings: PersistentSettings<TrackpadSettings>!
    unowned let recognitionSettings: PersistentSettings<RecognitionSettings>!

    init(
        generalSettings: PersistentSettings<GeneralSettings>,
        hostListSettings: PersistentSettings<HostListSettings>,
        trackpadSettings: PersistentSettings<TrackpadSettings>,
        recognitionSettings: PersistentSettings<RecognitionSettings>,
    ) {
        self.generalSettings = generalSettings
        self.hostListSettings = hostListSettings
        self.trackpadSettings = trackpadSettings
        self.recognitionSettings = recognitionSettings
    }

    func startConnection() async -> String? {
        let errMsg = await startConnection_()
        if errMsg == nil {
            isConnected = true
        }
        return errMsg
    }
    func stopConnection() {
        stopConnection_()
        isConnected = false
    }

    func sendConfigUpdate() -> String? {
        if !isConnected {
            return "Communication client not connected"
        }
        let configUpdate = SendableConfigUpdate(
            generalSettings: generalSettings.value,
            hostListSettings: hostListSettings.value,
            trackpadSettings: trackpadSettings.value,
            recognitionSettings: recognitionSettings.value
        )
        guard var data = try? JSONEncoder().encode(configUpdate) else {
            return "Failed to encode config update"
        }

        let packetType: UInt8 = PacketType.configUpdate.rawValue
        data = Data([packetType]) + data
        sendConfigUpdate_(data: data)
        return nil
    }

    func send(frame: LandmarkedFrame) -> String? {
        if !isConnected {
            return "Communication client not connected"
        }

        guard var data = try? JSONEncoder().encode(frame) else {
            return "Failed to encode landmarked frame"
        }

        let packetType: UInt8 = PacketType.stream.rawValue
        data = Data([packetType]) + data
        send_(data: data)
        return nil
    }

    //TODO
    // func send(frame: TrackpadFrame) -> String? {}

    func startConnection_() async -> String? {
        fatalError("Not implemented")
    }
    func stopConnection_() {
        fatalError("Not implemented")
    }
    func sendConfigUpdate_(data: Data) {
        fatalError("Not implemented")
    }
    func send_(data: Data) {
        fatalError("Not implemented")
    }

}
