//
//  RTCClient.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-19.
//

import ARKit
import AVFoundation
import Accelerate
import SwiftUI
import WebRTC

enum WebRTCClientError: Error {
    case FailedToSendFrame
    case notConnected
}

class WebRTCClient {
    private let factory = RTCPeerConnectionFactory()
    private var peerConnection: RTCPeerConnection!

    // Fast, real time data channel
    private var streamChannel: RTCDataChannel!
    // Slow, configuration data channel, only send data on change
    private var settingsChannel: RTCDataChannel!

    var isConnected: Bool = false

    private unowned let generalSettings: PersistentSettings<GeneralSettings>!
    private unowned let hostListSettings: PersistentSettings<HostListSettings>!
    private unowned let trackpadSettings: PersistentSettings<TrackpadSettings>!
    private unowned let recognitionSettings: PersistentSettings<RecognitionSettings>!

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

        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan

        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        peerConnection = factory.peerConnection(
            with: config,
            constraints: constraints,
            delegate: nil,
        )

        let channelConfig = RTCDataChannelConfiguration()
        streamChannel = peerConnection.dataChannel(
            forLabel: generalSettings.value.webRTCStreamChannelLabel,
            configuration: channelConfig
        )
        settingsChannel = peerConnection.dataChannel(
            forLabel: generalSettings.value.webRTCSettingsChannelLabel,
            configuration: channelConfig
        )
    }

    func createOffer() async -> String? {
        self.isConnected = false
        let constraints = RTCMediaConstraints(
            mandatoryConstraints: [
                "OfferToReceiveAudio": "false",
                "OfferToReceiveVideo": "false",
            ],
            optionalConstraints: nil
        )

        return await withCheckedContinuation { continuation in
            peerConnection.offer(for: constraints) { sdp, error in
                // check sdp is not nil
                if let error = error {
                    continuation.resume(
                        returning: "Error creating RTC offer: \(error.localizedDescription)")
                    return
                }

                guard let sdp = sdp else {
                    continuation.resume(returning: "Failed to create RTC offer: no SDP returned")
                    return
                }

                self.peerConnection.setLocalDescription(sdp) { error in
                    if let error = error {
                        continuation.resume(
                            returning:
                                "Error setting local description: \(error.localizedDescription)")
                    } else {
                        continuation.resume(returning: nil)
                    }
                }
            }
        }
    }

    func addAnswer(_ sdp: RTCSessionDescription) async -> String? {
        return await withCheckedContinuation { continuation in
            peerConnection.setRemoteDescription(
                sdp,
                completionHandler: { error in
                    if let error = error {
                        continuation.resume(
                            returning: "Error receiving answer: \(error.localizedDescription)")
                    } else {
                        self.isConnected = true
                        continuation.resume(returning: nil)
                    }
                })
        }
    }

    struct RTCConfigUpdate: Codable {
        let generalSettings: GeneralSettings
        let hostListSettings: HostListSettings
        let trackpadSettings: TrackpadSettings
        let recognitionSettings: RecognitionSettings
    }

    func sendConfigUpdate() -> String? {
        if !isConnected {
            return "RTC client not connected"
        }
        let configUpdate = RTCConfigUpdate(
            generalSettings: generalSettings.value,
            hostListSettings: hostListSettings.value,
            trackpadSettings: trackpadSettings.value,
            recognitionSettings: recognitionSettings.value
        )
        if let data = try? JSONEncoder().encode(configUpdate) {
            settingsChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
        } else {
            return "Failed to send config update"
        }
        return nil

    }

    func send(frame: LandmarkedFrame) -> String? {
        if !isConnected {
            return "RTC client not connected"
        }
        if let data = try? JSONEncoder().encode(frame) {
            streamChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
        } else {
            return "Failed to send landmarked frame"
        }
        return nil
    }

    //TODO send function for trackpad info
    // func send(frame: TrackpadFrame) -> String? {
    //     if !isConnected {
    //         return "RTC client not connected"
    //     }
    //     if let data = try? JSONEncoder().encode(frame) {
    //         streamChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
    //     } else {
    //         return "Failed to send trackpad frame"
    //     }
    //     return nil
    // }

}
