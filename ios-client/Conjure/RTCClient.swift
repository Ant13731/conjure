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
    
    private var colorSource: RTCVideoSource!
    private var colorTrack: RTCVideoTrack!
    private var videoCapturer: RTCCameraVideoCapturer!
    private var depthSource: RTCVideoSource!
    private var depthTrack: RTCVideoTrack!
//    private var depthCapturer: RTCCameraVideoCapturer!
    
    var isConnected: Bool = false

    override init() {
        super.init()
        
        // Peer-to-peer connection settings
        // Use STUN connectivity port offered by google to find peer-to-peer connections over the internet
        // DTLS to negotiate keys for encrypting SRTP media streams
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan
        let constraints = RTCMediaConstraints(mandatoryConstraints: [
            "minWidth": "640",
            "maxWidth": "640",
            "minHeight": "480",
            "maxHeight": "480",
            "maxFrameRate": "30"
        ], optionalConstraints: [
            "DtlsSrtpKeyAgreement": "true",
            "googCpuOveruseDetection": "true",
            "googVideoCodec": "H264"
        ])
        peerConnection = factory.peerConnection(with: config, constraints: constraints, delegate: nil)
        
        let channelConfig = RTCDataChannelConfiguration()
        dataChannel = peerConnection.dataChannel(forLabel: DataConfig.webRTCChannelLabel, configuration: channelConfig)
        
        colorSource = factory.videoSource()
        colorTrack = factory.videoTrack(with: colorSource, trackId: "video0")
        
        peerConnection.add(colorTrack, streamIds: ["stream0"])
                
        depthSource = factory.videoSource()
        depthTrack = factory.videoTrack(with: depthSource, trackId: "video1")
        peerConnection.add(depthTrack, streamIds: ["stream0"])
        
        videoCapturer = RTCCameraVideoCapturer(delegate: colorSource)
    }
    

    func setMediaBitrate(sdp: String, bitrate: Int) -> String {
      
        let mediaType = "video"
      var lines = sdp.components(separatedBy: "\n")
      var line = -1
            
      for (index, lineString) in lines.enumerated() {
        if lineString.hasPrefix("m=\(mediaType)") {
          line = index
          break
        }
      }
            
      guard line != -1 else {
        //Couldn't find the m (media) line return the original sdp
        print("Couldn't find the m line in SDP so returning the original sdp")
        return sdp
      }
      
      // Go to next line i.e. line after m
      line += 1
            
      //Now skip i and c lines
      while (lines[line].hasPrefix("i=") || lines[line].hasPrefix("c=")) {
        line += 1
      }
      
      let newLine = "b=AS:\(bitrate)"
      //Check if we're on b (bitrate) line, if so replace it
      if lines[line].hasPrefix("b") {
        print("Replacing the b line of the SDP")
        lines[line] = newLine
      } else {
        //If there's no b line, add a new b line
        lines.insert(newLine, at: line)
      }
      
      let modifiedSDP = lines.joined(separator: "\n")
      return modifiedSDP
        
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
            var modifiedSDPString = self.setMediaBitrate(sdp: sdp.sdp, bitrate: 30000000)
//            modifiedSDPString = self.disableBFrames(sdp: modifiedSDPString)
            
            let modifiedSDP = RTCSessionDescription(type: .offer, sdp: modifiedSDPString)
            
            
            // set local description and run complete function
            self.peerConnection.setLocalDescription(modifiedSDP) { error in
                if error != nil {
                    completion(.failure(error!))
                    return
                }
//                self.sendLocalDescriptionSignalling(sdp: modifiedSDP)
                completion(.success(sdp))
            }
        }
    }

    func addAnswer(_ sdp: RTCSessionDescription) {
        peerConnection.setRemoteDescription(sdp, completionHandler: { error in
            if error != nil {
                print("Error receiving answer:", error)
                return
            }})
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
    
    func send(colorBuffer: CVPixelBuffer, depthBuffer: CVPixelBuffer) {
        let timestampNs = Int64(Date().timeIntervalSince1970 * 1_000_000_000)
        let orientation = UIDevice.current.orientation
        
        let colorRTCPixelBuffer = RTCCVPixelBuffer(pixelBuffer: colorBuffer)
        let colorFrame = RTCVideoFrame(buffer: colorRTCPixelBuffer, rotation: rotationFromDeviceOrientation(orientation), timeStampNs: timestampNs)
        
        let formattedDepthBuffer = formatDepthBuffer(depthBuffer: depthBuffer)
        let depthRTCPixelBuffer = RTCCVPixelBuffer(pixelBuffer: formattedDepthBuffer)
        let depthFrame = RTCVideoFrame(buffer: depthRTCPixelBuffer, rotation: rotationFromDeviceOrientation(orientation), timeStampNs: timestampNs)
        
//        print(colorFrame)
        print(depthFrame)
        print("Format", CVPixelBufferGetPixelFormatType(depthBuffer))
        
        colorSource.capturer(videoCapturer, didCapture: colorFrame)
        depthSource.capturer(videoCapturer, didCapture: depthFrame)

        
    }
    
    func formatDepthBuffer(depthBuffer: CVPixelBuffer) -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(depthBuffer)
            let height = CVPixelBufferGetHeight(depthBuffer)

            // Create a new RGBA CVPixelBuffer
            var rgbaBufferOptional: CVPixelBuffer?
            let status = CVPixelBufferCreate(
                nil,
                width,
                height,
                kCVPixelFormatType_32BGRA, // RGBA 8-bit per channel
                nil,
                &rgbaBufferOptional
            )

            guard status == kCVReturnSuccess, let rgbaBuffer = rgbaBufferOptional else {
                print("Failed to create RGBA pixel buffer")
                return depthBuffer
            }

            CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
            CVPixelBufferLockBaseAddress(rgbaBuffer, [])

            let depthBase = CVPixelBufferGetBaseAddress(depthBuffer)!.assumingMemoryBound(to: UInt16.self)
            let rgbaBase = CVPixelBufferGetBaseAddress(rgbaBuffer)!.assumingMemoryBound(to: UInt8.self)

            let depthRowBytes = CVPixelBufferGetBytesPerRow(depthBuffer) / 2 // 2 bytes per Float16
            let rgbaRowBytes = CVPixelBufferGetBytesPerRow(rgbaBuffer)

            for y in 0..<height {
                let depthRow = depthBase.advanced(by: y * (depthRowBytes / 2))
                let rgbaRow = rgbaBase.advanced(by: y * rgbaRowBytes)

                for x in 0..<width {
                    let depthValue = depthRow[x]
                    let highByte = UInt8(depthValue >> 8)
                    let lowByte = UInt8(depthValue & 0xFF)

                    let pixelOffset = x * 4
                    rgbaRow[pixelOffset + 0] = highByte // R
                    rgbaRow[pixelOffset + 1] = lowByte  // G
                    rgbaRow[pixelOffset + 2] = 0        // B
                    rgbaRow[pixelOffset + 3] = 0        // A
                }
            }

            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(rgbaBuffer, [])

            return rgbaBuffer
        
    }
    
    func rotationFromDeviceOrientation(_ o: UIDeviceOrientation) -> RTCVideoRotation {
        switch o {
        case .portrait:
            return ._90
        case .landscapeLeft:
            return ._180
        case .portraitUpsideDown:
            return ._270
        case .landscapeRight:
            return ._0
        default:
            return ._90
        }
    }
}
