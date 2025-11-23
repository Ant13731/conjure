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

enum ConnectionMode: String, CaseIterable, Identifiable {
    case onDevice = "On-device ML"
    case streamWebRTC = "WebRTC Stream"
    case streamTCP = "TCP Stream"

    var id: String {self.rawValue}
}


struct LoginView: View {
    @State private var ip_address: String = LoginConfig.defaultIPAddress
    @State private var port: String = LoginConfig.defaultPort

    @State private var connectionResultMessage = ""
    @State private var cameraStreamMessage = ""

    @State private var connectedWebRTC: Bool = false
    @State private var connectedUSB: Bool = false
    @State private var connectionMode: ConnectionMode = .streamTCP

    @State private var webRTCClient: WebRTCClient!
    @State private var cameraManager: CameraManager!
    @State private var frameFuser: FrameFuser!
    @State private var usbSender: USBSender!

    var body: some View {
            VStack(spacing: 20) {
                Spacer()
                Text("Conjure Client")
                    .font(.largeTitle)
                    .bold()

                Picker("Connection Mode", selection: $connectionMode) {
                                ForEach(ConnectionMode.allCases) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(SegmentedPickerStyle())

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
//        handleLoginWebRTC()
        switch connectionMode {
        case .streamTCP:
            handleLoginUSB()
        case _:
            handleLoginWebRTC()
        }
    }

    func handleLoginUSB() {
        connectedUSB = false
        guard let int_port = Int(port) else {
            connectionResultMessage = "Invalid port given (must be an integer)"
            return
        }
        usbSender = USBSender(port: int_port)
        usbSender.connect {success in
            if success {
                connectionResultMessage = "Connection Result: USB connected"
                    connectedUSB = true
            }
            else {
                connectionResultMessage = "Connection Result: USB connection failed"
            }
        }
    }

        func handleLoginWebRTC() {
            print("Initiating webRTCClient")
            guard let turboShader = try? TurboLUTManager() else {
                print("Failed to initialize shader")
                return
            }
            webRTCClient = WebRTCClient(turboShader: turboShader)

            // TODO: structurally validate input (ip must have 4 dots and numbers, port must have 4 numbers)
            connectedWebRTC = false
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
                    connectedWebRTC = true

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

        switch connectionMode {
        case .onDevice:
            print("Starting On-device ML Streaming")
            frameFuser = FrameFuser(webRTCClient)
            cameraManager = CameraManager(frameFuser: frameFuser)
        case .streamWebRTC:
            print("Starting Image-only streaming via WebRTC")
            cameraManager = CameraManager(webRTCClient: webRTCClient)
        case .streamTCP:
            print("Starting Image-only streaming via TCP")
            cameraManager = CameraManager(usbSender: usbSender)
        }

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
