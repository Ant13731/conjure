//
//  LoginView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-10-26.
//

import SwiftUI
import WebRTC

class WebRTCClient: NSObject {
    private var peerConnection: RTCPeerConnection!
    private let factory = RTCPeerConnectionFactory()
    
    private var localVideoTrack: RTCVideoTrack!
    private var videoCapturer: RTCCameraVideoCapturer!

    override init() {
        super.init()
        
        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: ["DtlsSrtpKeyAgreement": "true"])
        peerConnection = factory.peerConnection(with: config, constraints: constraints, delegate: nil)
    }

    private func startCapture() {
        //TODO fix
        // Create video source
        let videoSource = factory.videoSource()
        videoCapturer = RTCCameraVideoCapturer(delegate: videoSource)
        localVideoTrack = factory.videoTrack(with: videoSource, trackId: "video0")

        // Add track to connection
        let stream = factory.mediaStream(withStreamId: "stream0")
        stream.addVideoTrack(localVideoTrack)
        peerConnection.add(stream)
        
        // Start front camera
        let device = RTCCameraVideoCapturer.captureDevices().first { $0.position == .front }!
        let format = RTCCameraVideoCapturer.supportedFormats(for: device).last!
        let fps = format.videoSupportedFrameRateRanges.first!.maxFrameRate
        videoCapturer.startCapture(with: device, format: format, fps: Int(fps))
    }

    func createOffer(completion: @escaping (RTCSessionDescription) -> Void) {
        let constraints = RTCMediaConstraints(
            mandatoryConstraints: ["OfferToReceiveAudio": "false", "OfferToReceiveVideo": "false"],
            optionalConstraints: nil
        )
        // async with closures
        peerConnection.offer(for: constraints) { sdp, error in
            // check sdp is not nil
            guard let sdp = sdp else { return }
            // set local description and run complete function
            self.peerConnection.setLocalDescription(sdp) { error in
                completion(sdp)
            }
        }
    }

    func addAnswer(_ sdp: RTCSessionDescription) {
        peerConnection.setRemoteDescription(sdp, completionHandler: { _ in })
    }
}

struct LoginView: View {
    @State private var ip_address: String = ""
    @State private var port: String = ""
    @State private var connectionResultMessage = ""
    
    private let webRTCClient = WebRTCClient()
    
    var body: some View {
            VStack(spacing: 20) {
                Spacer()
                Text("Conjure Client")
                    .font(.largeTitle)
                    .bold()
                
                Spacer()

                TextField("Server IP Address", text: $ip_address)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(8)

                SecureField("Server Port", text: $port)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(8)
                
                Spacer()

                Button(action: handleLogin) {
                    Text("Connect")
                        .frame(width: 0.7 * UIScreen.main.bounds.width)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }

                Text(connectionResultMessage)
                    .font(.subheadline)
                    .padding(.top, 10)

                Spacer()
            }
            .padding()
        }

        func handleLogin() {
            // TODO: structurally validate input (ip must have 4 dots and numbers, port must have 4 numbers)
            webRTCClient.createOffer { offer in
                // Send offer to handshake server
                let url = URL(string: "http://\($ip_address):\($port)/offer")!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                let body: [String: Any] = ["sdp": offer.sdp, "type": "offer"]
                request.httpBody = try? JSONSerialization.data(withJSONObject: body)

                URLSession.shared.dataTask(with: request) { data, _, _ in
                    guard let data = data,
                          let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                          let sdpString = json["sdp"],
                          let typeString = json["type"] else { return }

                    let answer = RTCSessionDescription(type: .answer, sdp: sdpString)
                    webRTCClient.addAnswer(answer)
                }.resume()
            }
            connectionResultMessage = "Connection successful"
            
        }
    }

#Preview {
    LoginView()
}
