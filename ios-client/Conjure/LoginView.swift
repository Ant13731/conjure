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

enum PermissionError: Error {
    case cameraPermError
    case cameraStartError
    case depthDeviceError
    case depthStartError
}

class WebRTCClient: NSObject, AVCaptureDataOutputSynchronizerDelegate {
    private var peerConnection: RTCPeerConnection!
    private let factory = RTCPeerConnectionFactory()
    
    private var localVideoTrack: RTCVideoTrack!
    private var videoCapturer: RTCCameraVideoCapturer!
    
    private var depthDataChannel: RTCDataChannel?
    private var videoSource: RTCVideoSource!

    override init() {
        super.init()
        
        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: ["DtlsSrtpKeyAgreement": "true"])
        peerConnection = factory.peerConnection(with: config, constraints: constraints, delegate: nil)
        
        let dcConfig = RTCDataChannelConfiguration()
        depthDataChannel = peerConnection.dataChannel(forLabel: "depthChannel", configuration: dcConfig)
    }

    func startCaptureSend() {
        videoSource = factory.videoSource()
        localVideoTrack = factory.videoTrack(with: videoSource, trackId: "video0")
        
        let stream = factory.mediaStream(withStreamId: "stream0")
        stream.addVideoTrack(localVideoTrack)
        peerConnection.add(stream)
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
    
    private var captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session queue", attributes: [], autoreleaseFrequency: .workItem)
    private var videoOutput = AVCaptureVideoDataOutput()
    private var depthOutput = AVCaptureDepthDataOutput()
    private var synchronizer: AVCaptureDataOutputSynchronizer?
    private let videoDeviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInTrueDepthCamera],
                                                                               mediaType: .video,
                                                                               position: .front)
    private var videoDeviceInput: AVCaptureDeviceInput!
    private let session = AVCaptureSession()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let dataOutputQueue = DispatchQueue(label: "video data queue", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    
    var isCameraAuthorized: Bool {
        get async {
            let status = AVCaptureDevice.authorizationStatus(for: .video)
            
            // Determine if the user previously authorized camera access.
            var isAuthorized = status == .authorized
            
            // If the system hasn't determined the user's authorization status,
            // explicitly prompt them for approval.
            if status == .notDetermined {
                isAuthorized = await AVCaptureDevice.requestAccess(for: .video)
            }
            
            return isAuthorized
        }
    }
    private enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
    private var setupResult: SessionSetupResult = .success
    private func configureSession() {
        if setupResult != .success {
            return
        }
        let defaultVideoDevice: AVCaptureDevice? = videoDeviceDiscoverySession.devices.first
        
        guard let videoDevice = defaultVideoDevice else {
            print("Could not find any video device")
            setupResult = .configurationFailed
            return
        }
        
        do {
            videoDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
        } catch {
            print("Could not create video device input: \(error)")
            setupResult = .configurationFailed
            return
        }
        
        session.beginConfiguration()
        
        session.sessionPreset = AVCaptureSession.Preset.vga640x480
        
        // Add a video input
        guard session.canAddInput(videoDeviceInput) else {
            print("Could not add video device input to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        session.addInput(videoDeviceInput)
        
        // Add a video data output
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        } else {
            print("Could not add video data output to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // Add a depth data output
        if session.canAddOutput(depthDataOutput) {
            session.addOutput(depthDataOutput)
            depthDataOutput.isFilteringEnabled = false
            if let connection = depthDataOutput.connection(with: .depthData) {
                connection.isEnabled = true
            } else {
                print("No AVCaptureConnection")
            }
        } else {
            print("Could not add depth data output to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // Search for highest resolution with half-point depth values
        let depthFormats = videoDevice.activeFormat.supportedDepthDataFormats
        let filtered = depthFormats.filter({
            CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat16
        })
        let selectedFormat = filtered.max(by: {
            first, second in CMVideoFormatDescriptionGetDimensions(first.formatDescription).width < CMVideoFormatDescriptionGetDimensions(second.formatDescription).width
        })
        
        do {
            try videoDevice.lockForConfiguration()
            videoDevice.activeDepthDataFormat = selectedFormat
            videoDevice.unlockForConfiguration()
        } catch {
            print("Could not lock device for configuration: \(error)")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // Use an AVCaptureDataOutputSynchronizer to synchronize the video data and depth data outputs.
        // The first output in the dataOutputs array, in this case the AVCaptureVideoDataOutput, is the "master" output.
        outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        outputSynchronizer!.setDelegate(self, queue: dataOutputQueue)
        session.commitConfiguration()
        
        
    }
    func setupCaptureSession() async -> Result<String, PermissionError> {
        guard await isCameraAuthorized else { return .failure(.cameraPermError)}
        
        sessionQueue.async {self.configureSession()}
        
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo
        
        guard let depthDevice = AVCaptureDevice.default(
            .builtInTrueDepthCamera,
            for: .video,
            position: .front) else {
            return .failure(.depthDeviceError)
        }
        
//        guard let videoInput = try? AVCaptureDeviceInput.default(for: .video),
//              captureSession.canAddInput(videoInput) else {
//            return .failure(.depthStartError)
//        }
//        captureSession.addInput(videoInput)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            videoOutput.alwaysDiscardsLateVideoFrames = true
//            videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        }
        
        if captureSession.canAddOutput(depthOutput) {
            captureSession.addOutput(depthOutput)
            depthOutput.isFilteringEnabled = false
//            depthOutput.setDelegate(self, callbackQueue: sessionQueue)
        }
                
        // Align depth with video stream
//        for connection in depthOutput.connections {
//            connection.isEnabled = true
//        }
        
        synchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoOutput, depthOutput])
        synchronizer?.setDelegate(self, queue: sessionQueue)
                
        captureSession.commitConfiguration()
        captureSession.startRunning()
        
        return .success("Camera capture started successfully")
        }
    
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                    didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
            guard let syncedVideoData = synchronizedDataCollection.synchronizedData(for: videoOutput)
                    as? AVCaptureSynchronizedSampleBufferData,
                  let syncedDepthData = synchronizedDataCollection.synchronizedData(for: depthOutput)
                    as? AVCaptureSynchronizedDepthData else { return }

            guard !syncedVideoData.sampleBufferWasDropped,
                  !syncedDepthData.depthDataWasDropped else { return }

            // RGB frame
            let sampleBuffer = syncedVideoData.sampleBuffer
            guard let colorBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

            // Depth frame
            let depthData = syncedDepthData.depthData
            let depthMap = depthData.depthDataMap

            // Process or stream them together here
            processFrame(colorBuffer: colorBuffer, depthMap: depthMap)
        }

    private func processFrame(colorBuffer: CVPixelBuffer, depthMap: CVPixelBuffer) {
        // ---- 1. Send RGB frame via WebRTC video track ----
                let rtcPixelBuffer = RTCCVPixelBuffer(pixelBuffer: colorBuffer)
                let timestampNs = Int64(Date().timeIntervalSince1970 * 1_000_000_000)
                let frame = RTCVideoFrame(buffer: rtcPixelBuffer, rotation: ._0, timeStampNs: timestampNs)
                
                videoSource.capturer(videoCapturer, didCapture: frame)
                
                // ---- 2. Send Depth via Data Channel ----
                guard let dc = depthDataChannel, dc.readyState == .open else { return }

                CVPixelBufferLockBaseAddress(depthMap, .readOnly)
                let width = CVPixelBufferGetWidth(depthMap)
                let height = CVPixelBufferGetHeight(depthMap)
                let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
                let depthPtr = CVPixelBufferGetBaseAddress(depthMap)!
                let depthSize = bytesPerRow * height
                let depthData = Data(bytes: depthPtr, count: depthSize)
                CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
                
                var payload = Data()
                var ts = timestampNs
                var w = Int32(width)
                var h = Int32(height)
                payload.append(Data(bytes: &ts, count: MemoryLayout<Int64>.size))
                payload.append(Data(bytes: &w, count: MemoryLayout<Int32>.size))
                payload.append(Data(bytes: &h, count: MemoryLayout<Int32>.size))
                payload.append(depthData)

                let buffer = RTCDataBuffer(data: payload, isBinary: true)
                dc.sendData(buffer)
        }
    
    
    
}

struct LoginView: View {
    @State private var ip_address: String = ""
    @State private var port: String = ""
    @State private var connectionResultMessage = ""
    @State private var cameraStreamMessage = ""
    @State private var connected: Bool = false
    
    private let webRTCClient = WebRTCClient()
    
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
        Task {cameraStreamMessage = "Starting Camera..."
            webRTCClient.startCaptureSend()
            let result = await webRTCClient.setupCaptureSession()
            switch result {
            case .failure(let error):
                switch error {
                case .cameraPermError:
                    cameraStreamMessage = "Camera permission error"
                case .depthDeviceError:
                    cameraStreamMessage = "TrueDepth camera not available"
                case .depthStartError:
                    cameraStreamMessage = "Failed to start capture session"
                }
                return
                
            case .success(let message):
                cameraStreamMessage = message
            }
            
//            webRTCClient.videoCapturer = RTCCameraVideoCapturer(delegate: webRTCClient.videoSource)
//            webRTCClient.videoSource = webRTCClient.factory.videoSource()
            cameraStreamMessage = "Camera started, streaming frames..."
            
            // Start camera capture session from front facin g camera
            // Start depth camera capture session and fuse together
            // Send fused video frames through webRTC
            
            
        }
    }
    func stopCameraStream(){}
    
}

#Preview {
    LoginView()
}
