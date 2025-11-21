//
//  LoginView.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-10-26.
//

import SwiftUI
import WebRTC
import ARKit
import AVFoundation

struct LoginView: View {
    @State private var ip_address: String = LoginConfig.defaultIPAddress
    @State private var port: String = LoginConfig.defaultPort
    
    @State private var connectionResultMessage = ""
    @State private var cameraStreamMessage = ""
    @State private var connected: Bool = false
    
    @State private var webRTCClient: WebRTCClient!
    @State private var cameraManager: CameraManager!
    @State private var frameFuser: FrameFuser!
    
    var body: some View {
            VStack(spacing: 20) {
                Spacer()
                Text("Conjure Client")
                    .font(.largeTitle)
                    .bold()
                
                TextField("Server IP Address", text: $ip_address)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(8)

                TextField("Server Port", text: $port)
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(8)
                
                Button(action: handleLogin) {
                    Text("Connect")
                        .frame(width: 0.7 * UIScreen.main.bounds.width)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                
                Button(action: startCameraStream) {
                    Text("Start Camera Stream")
                        .frame(width: 0.7 * UIScreen.main.bounds.width)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                Button(action: stopCameraStream) {
                    Text("Stop Camera Stream")
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
                
                Text(cameraStreamMessage)
                    .font(.subheadline)
                    .padding(.top, 10)

                Spacer()
                
               
            }
            .padding()
        }

        func handleLogin() {
            print("Initiating webRTCClient")
            webRTCClient = WebRTCClient()
            
            // TODO: structurally validate input (ip must have 4 dots and numbers, port must have 4 numbers)
            connected = false
            guard URL(string: "http://\(ip_address):\(port)/offer") != nil else {
                connectionResultMessage = "Connection Result: Malformed input http://\(ip_address):\(port)/offer: Please check IP address and port"
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
                            connectionResultMessage = "Connection Result: Failed to send URL connection request: \(err)"
                            return
                        }
                        
                        guard let data = data else {
                            connectionResultMessage = "Connection Result: Failed to get URL response. Got \(data)"
                            return
                        }
                        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: AnyObject]
                              else {
                                  let data_str = String(bytes: data, encoding: .utf8) ?? "nil"
                                  connectionResultMessage = "Connection Result: Failed to parse URL response. Got \(data_str)"
                                  return
                              }
                        guard let json_data = json["data"] as? [String: String],
                              //                        let typeString = json_data["type"]
                              let sdpString = json_data["sdp"] else {
                                  connectionResultMessage = "Connection Result: Expected fields `sdp` and `type` are not in the json response: \(json)"
                                  return
                              }
                        
                        let answer = RTCSessionDescription(type: .answer, sdp: sdpString)
                        webRTCClient.addAnswer(answer)
                    }.resume()
                    connectionResultMessage = "Connection Result: Connection successful"
                    connected = true
                    
                case .failure(let err):
                    connectionResultMessage = "Connection Result: Failed to create offer: \(err)"
                }
            }
        }
    
    func startCameraStream(){
        if cameraManager != nil && cameraManager.isSessionRunning {
            cameraManager.stopSession()
        }
        
        cameraStreamMessage = "Setting up camera..."
        frameFuser = FrameFuser(webRTCClient)
        cameraManager = CameraManager(frameFuser: frameFuser)
        
        let res = cameraManager.setupSession()
        
        switch res {
        case .failure(let error):
            print("Error setting up camera:",error)
            cameraStreamMessage = "Error setting up camera: \(error)"
            switch error {
            case .notAuthorized:
                cameraStreamMessage = "Camera permission error"
            case .configurationFailed:
                cameraStreamMessage = "Camera configuration failed"
            case .failedToAddCamera:
                cameraStreamMessage = "Failed to add camera to capture session"
            case .failedToAddDepthSensor:
                cameraStreamMessage = "Failed to add depth sensor to capture session"
            case .failedToAddDepthSensorCapture:
                cameraStreamMessage = "Failed to add depth sensor input"
            }
            return
            
        case _:
            break
        }
        
        cameraStreamMessage = "Starting Camera..."
        try! cameraManager.startSession()
        cameraStreamMessage = "Camera started"
            
        }
    
    func stopCameraStream(){
        if cameraManager == nil {
            return
        }
        cameraManager.stopSession()
        cameraStreamMessage = "Camera stream stopped"
    }
    
}

#Preview {
    LoginView()
}
