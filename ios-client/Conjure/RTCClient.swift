//
//  RTCClient.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-19.
//
import SwiftUI
import WebRTC
import ARKit
import AVFoundation

enum WebRTCClientError: Error {
    case FailedToSendFrame
    case notConnected
}

class WebRTCClient: NSObject {
    private let factory = RTCPeerConnectionFactory()
    private var peerConnection: RTCPeerConnection!
    private var dataChannel: RTCDataChannel!
    
    var isConnected: Bool = false

    override init() {
        super.init()
        
        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: ["DtlsSrtpKeyAgreement": "true"])
        peerConnection = factory.peerConnection(with: config, constraints: constraints, delegate: nil)
        
        let channelConfig = RTCDataChannelConfiguration()
        dataChannel = peerConnection.dataChannel(forLabel: DataConfig.webRTCChannelLabel, configuration: channelConfig)
    }

    func createOffer(completion: @escaping (Result<RTCSessionDescription, Error>) -> Void) {
        self.isConnected = false
        let constraints = RTCMediaConstraints(
            mandatoryConstraints: ["OfferToReceiveAudio": "false", "OfferToReceiveVideo": "false"],
            optionalConstraints: nil
        )
        
        // async with closures
        peerConnection.offer(for: constraints) { sdp, error in
            // check sdp is not nil
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let sdp = sdp else {
                let error = NSError(
                    domain: "WebRTCOffer",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create offer: no SDP returned"]
                )
                completion(.failure(error))
                return
            }
            // set local description and run complete function
            self.peerConnection.setLocalDescription(sdp) { error in
                completion(.success(sdp))
            }
        }
    }

    func addAnswer(_ sdp: RTCSessionDescription) {
        peerConnection.setRemoteDescription(sdp, completionHandler: { _ in })
        self.isConnected = true
    }
    
    func send(frame: Frame) -> Result<Void, WebRTCClientError>{
        if !isConnected {
            return .failure(.notConnected)
        }
        if let data = try? JSONEncoder().encode(frame) {
            dataChannel.sendData(RTCDataBuffer(data: data, isBinary: true))
        }
        else {
            return .failure(.FailedToSendFrame)
        }
        return .success(())
    }
    
}
