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

class WebRTCManager: CommunicationManager {
    private var peerConnection: RTCPeerConnection!

    // Fast, real time data channel
    private var streamChannel: RTCDataChannel!
    // Slow, configuration data channel, only send data on change
    private var settingsChannel: RTCDataChannel!

    override init(
        generalSettings: PersistentSettings<GeneralSettings>,
        hostListSettings: PersistentSettings<HostListSettings>,
        trackpadSettings: PersistentSettings<TrackpadSettings>,
        recognitionSettings: PersistentSettings<RecognitionSettings>,
    ) {
        super.init(
            generalSettings: generalSettings,
            hostListSettings: hostListSettings,
            trackpadSettings: trackpadSettings,
            recognitionSettings: recognitionSettings
        )

        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan

        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        peerConnection = RTCPeerConnectionFactory().peerConnection(
            with: config,
            constraints: constraints,
            delegate: nil,
        )

        let streamConfiguration = RTCDataChannelConfiguration()
        streamConfiguration.isOrdered = false
        streamConfiguration.maxRetransmits = 0
        // streamConfiguration.protocol = "udp"
        streamConfiguration.isNegotiated = false

        streamChannel = peerConnection.dataChannel(
            forLabel: generalSettings.value.webRTCStreamChannelLabel,
            configuration: streamConfiguration
        )
        settingsChannel = peerConnection.dataChannel(
            forLabel: generalSettings.value.webRTCSettingsChannelLabel,
            configuration: RTCDataChannelConfiguration()
        )
    }

    override func startConnection_() async -> String? {
        print("WebRTC Start connection: creating offer")
        let result = await createOffer()
        if case .failure(let errMsg) = result {
            return "Connection Result: \(errMsg)"

        }
        let offer = try! result.get()

        print("WebRTC Start connection: sending offer")
        let res = await sendOffer(offer)
        if case .failure(let errMsg) = res {
            return "Connection Result: \(errMsg)"
        }
        let answer = try! res.get()

        print("WebRTC Start connection: adding answer")
        if let errMsg = await addAnswer(answer) {
            return "Connection Result: \(errMsg)"
        }
        return nil
    }

    private func createOffer() async -> Result<RTCSessionDescription, StrError> {
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
                if let error = error {
                    continuation.resume(
                        returning: .failure(
                            StrError("Error creating RTC offer: \(error.localizedDescription)")))
                    return
                }

                guard let sdp = sdp else {
                    continuation.resume(
                        returning: .failure(StrError("Failed to create RTC offer: no SDP returned"))
                    )
                    return
                }

                self.peerConnection.setLocalDescription(sdp) { error in
                    if let error = error {
                        continuation.resume(
                            returning:
                                .failure(
                                    StrError(
                                        "Error setting local description: \(error.localizedDescription)"
                                    )
                                ))
                    } else {
                        continuation.resume(returning: .success(sdp))
                    }
                }
            }
        }
    }

    private func sendOffer(_ offer: RTCSessionDescription) async -> Result<
        RTCSessionDescription, StrError
    > {
        guard let ipAddress = hostListSettings.value.currentHost?.ipAddress,
            let port = hostListSettings.value.currentHost?.port
        else {
            return .failure(StrError("No current host selected"))
        }

        guard URL(string: "http://\(ipAddress):\(port)/offer") != nil else {
            return .failure(
                StrError(
                    "Malformed input http://\(ipAddress):\(port)/offer: Please check IP address and port"
                )
            )

        }
        let url = URL(string: "http://\(ipAddress):\(port)/offer")!

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["sdp": offer.sdp, "type": "offer"]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        let data_: Data?
        print("Send offer: sending request to \(url.absoluteString)")
        do {
            data_ = try await URLSession.shared.data(for: request).0
            print("Send offer: received response")
        } catch let error {
            return .failure(StrError("Failed to send URL connection request: \(error)"))
        }

        guard let data = data_ else {
            return .failure(
                StrError("Failed to get URL response. Got \(String(describing: data_))"))
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: AnyObject]
        else {
            let data_str = String(bytes: data, encoding: .utf8) ?? "nil"
            return .failure(StrError("Failed to parse URL response. Got \(data_str)"))
        }

        guard let json_data = json["data"] as? [String: String],
            let sdpString = json_data["sdp"]
        else {
            return .failure(
                StrError("Expected fields `sdp` and `type` are not in the json response: \(json)"))
        }

        let answer = RTCSessionDescription(type: .answer, sdp: sdpString)
        return .success(answer)

    }

    private func addAnswer(_ answer: RTCSessionDescription) async -> String? {
        return await withCheckedContinuation { continuation in
            peerConnection.setRemoteDescription(
                answer,
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

    override func sendConfigUpdate_(data: Data) {
        settingsChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
    }

    override func send_(data: Data) {
        streamChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
    }

    override func stopConnection_() {
        streamChannel.close()
        settingsChannel.close()
        peerConnection.close()
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
