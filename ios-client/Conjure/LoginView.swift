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

    func startCapture() {
//        AVCaptureDevice.requestAccess(for: .video) {granted in
//            if granted{
//                
//            }
//            else {
//                
//            }
//        }
        
        //TODO fix
        // Create video source
//        let videoCapture = AVCaptureDevice.default(for: .video)
////        let depthCapture = AVCaptureDevice.default(for: .depthData)
//        let mediaStream = RTCMediaStream(streamId: "localStream")
//        mediaStream.addVideoTrack(try! RTCVideoTrack(source: videoSource))
        
        
        
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

    func createOffer(completion: @escaping (Result<RTCSessionDescription, Error>) -> Void) {
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
    }
}

struct LoginView: View {
    @State private var ip_address: String = ""
    @State private var port: String = ""
    @State private var connectionResultMessage = ""
    @State private var connected: Bool = false
    
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

                TextField("Server Port", text: $port)
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
            connected = false
            guard URL(string: "http://\(ip_address):\(port)/offer") != nil else {
                connectionResultMessage = "Malformed input http://\(ip_address):\(port)/offer: Please check IP address and port"
                return
            }
            let url = URL(string: "http://\(ip_address):\(port)/offer")!
            
            
            webRTCClient.createOffer { res in
                switch res {
                case .success(let offer):
                    // Send offer to handshake server
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    let body: [String: Any] = ["sdp": offer.sdp, "type": "offer"]
                    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
                    
                    URLSession.shared.dataTask(with: request) { data, _, err in
                        if let err = err {
                            connectionResultMessage = "Failed to send URL connection request: \(err)"
                            return
                        }
                        
                        guard let data = data else {
                            connectionResultMessage = "Failed to get URL response. Got \(data)"
                            return
                        }
                        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: AnyObject]
                              else {
                                  let data_str = String(bytes: data, encoding: .utf8) ?? "nil"
                                  connectionResultMessage = "Failed to parse URL response. Got \(data_str)"
                                  return
                              }
                        guard let json_data = json["data"] as? [String: String],
                              //                        let typeString = json_data["type"]
                              let sdpString = json_data["sdp"] else {
                                  connectionResultMessage = "Expected fields `sdp` and `type` are not in the json response: \(json)"
                                  return
                              }
                        
                        let answer = RTCSessionDescription(type: .answer, sdp: sdpString)
                        webRTCClient.addAnswer(answer)
                    }.resume()
                    connectionResultMessage = "Connection successful"
                    connected = true
                    
                    
                    let oldConnectionResultMessage = connectionResultMessage
                    connectionResultMessage.append("\nStarting Camera...")
                        
//                    webRTCClient.startCapture()
                    
                    
                case .failure(let err):
                    connectionResultMessage = "Failed to create offer: \(err)"
                }
            }
            
           
            
            
            
        }
    }

#Preview {
    LoginView()
}
