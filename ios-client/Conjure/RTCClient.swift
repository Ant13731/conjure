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
import Accelerate

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
    let turboShader: TurboLUTManager!

    init(turboShader: TurboLUTManager) {
//        super.init()
        self.turboShader = turboShader

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
//        print(depthFrame)
//        print("Format", CVPixelBufferGetPixelFormatType(depthBuffer))

        colorSource.capturer(videoCapturer, didCapture: colorFrame)
        depthSource.capturer(videoCapturer, didCapture: depthFrame)


    }



    func formatDepthBufferSIMDAttempt(depthBuffer: CVPixelBuffer) -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

//        if CVPixelBufferGetPixelFormatType(depthBuffer) != kCVPixelFormatType_DepthFloat16 {
//                    print("Pixel format not in the expected Float16 depth format")
////                    return depthBuffer
//                }

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

            defer {
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
                CVPixelBufferUnlockBaseAddress(rgbaBuffer, [])
            }

            let depthBase = CVPixelBufferGetBaseAddress(depthBuffer)!.assumingMemoryBound(to: Float16.self)
            let rgbaBase = CVPixelBufferGetBaseAddress(rgbaBuffer)!.assumingMemoryBound(to: UInt8.self)

//            let depthRowBytes = CVPixelBufferGetBytesPerRow(depthBuffer)
//            let rgbaRowBytes = CVPixelBufferGetBytesPerRow(rgbaBuffer)



        // Convert Float16 → Float32
        var depthF32 = [Float](repeating: 0, count: width*height)
        var srcBuffer = vImage_Buffer(data: depthBase,
                                      height: vImagePixelCount(height),
                                      width: vImagePixelCount(width),
                                      rowBytes: CVPixelBufferGetBytesPerRow(depthBuffer))

        depthF32.withUnsafeMutableBytes { dstBytes in
            var dstBuffer = vImage_Buffer(
                data: dstBytes.baseAddress!,
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: width * MemoryLayout<Float>.size
            )

            vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
        }
//        var dstBuffer = vImage_Buffer(data: &depthF32,
//                                      height: vImagePixelCount(height),
//                                      width: vImagePixelCount(width),
//                                      rowBytes: width*MemoryLayout<Float>.size)
//        vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)


        // Apply normalization and LUT in parallel
//        DispatchQueue.concurrentPerform(iterations: depthF32.count) { i in
//            let normalized: Float
//            let d = depthF32[i]
//            if d.isNaN {
//                normalized = 0.0
//            } else {
//                let clamped = max(min(depthF32[i], DataConfig.maxDepth), DataConfig.minDepth)
//                normalized = (clamped - DataConfig.minDepth) / (DataConfig.maxDepth - DataConfig.minDepth)
//            }
//
//                // Get Turbo index
//            let idx = Int(max(0, min(255, Int(normalized * 255.0))))
//            let (r,g,b,a) = TurboLUTManager.turboTableUInt8[idx]
//            // write to output BGRA8 buffer
//            rgbaBase[i*4+0] = b
//            rgbaBase[i*4+1] = g
//            rgbaBase[i*4+2] = r
//            rgbaBase[i*4+3] = a
//        }
        let rowBytes = CVPixelBufferGetBytesPerRow(rgbaBuffer)
        let bytesPerPixel = 4

        DispatchQueue.concurrentPerform(iterations: height) { y in
            let rowStart = y * width
            let outRow = rgbaBase + y * rowBytes

            for x in 0..<width {
                let i = rowStart + x
                let d = depthF32[i]

                let normalized: Float
                if d.isNaN {
                    normalized = 0
                } else {
                    let clamped = max(min(d, DataConfig.maxDepth), DataConfig.minDepth)
                    normalized = (clamped - DataConfig.minDepth) / (DataConfig.maxDepth - DataConfig.minDepth)
                }

                let idx = min(max(Int(normalized * 255), 255), 0)
                let (r,g,b,a) = TurboLUTManager.turboTableUInt8[idx]

                let p = outRow + x * bytesPerPixel
                p[0] = b
                p[1] = g
                p[2] = r
                p[3] = a
            }
        }
        return rgbaBuffer
    }


    func formatDepthBuffer(depthBuffer: CVPixelBuffer) -> CVPixelBuffer {
        // Attempt to get metal working for superfast encoding - no dice
//        if turboShader != nil {
//            if let ret = turboShader.formatDepthBufferMetal(depthBuffer: depthBuffer) {
//                return ret
//            }
//            print("Failed to apply shader to depth frame")
//        }


        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        if CVPixelBufferGetPixelFormatType(depthBuffer) != kCVPixelFormatType_DepthFloat16 {
                    print("Pixel format not in the expected Float16 depth format")
//                    return depthBuffer
                }

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

            let depthBase = CVPixelBufferGetBaseAddress(depthBuffer)!.assumingMemoryBound(to: Float16.self)
            let rgbaBase = CVPixelBufferGetBaseAddress(rgbaBuffer)!.assumingMemoryBound(to: UInt8.self)

            let depthRowBytes = CVPixelBufferGetBytesPerRow(depthBuffer)
            let rgbaRowBytes = CVPixelBufferGetBytesPerRow(rgbaBuffer)

            for y in 0..<height {
                let depthRow = depthBase.advanced(by: y * (depthRowBytes / MemoryLayout<Float16>.size))
                let rgbaRow = rgbaBase.advanced(by: y * rgbaRowBytes)

                for x in 0..<width {
                    // Converting to RG scale method - kind-of works but compression kills it
//                    let depthValue = depthRow[x]
//                    var scaledValue: UInt16!
//                    if depthValue.isNaN {
//                        scaledValue = 0
//                    }
//                    else {
//                        let clampedValue = max(min(Double(depthValue), DataConfig.maxDepth), DataConfig.minDepth)
//                        let scaledValueNumerator = clampedValue - DataConfig.minDepth
//                        let scaledValueDenomenator = DataConfig.maxDepth-DataConfig.minDepth
//                        let scaledValueFloat = scaledValueNumerator / scaledValueDenomenator * pow(2.0, 2*8)
//
//                        scaledValue = UInt16(max(min(scaledValueFloat, Double(UInt16.max)), 0))
//                    }
//
//                    let highByte = UInt8(scaledValue >> 8)
//                    let lowByte = UInt8(scaledValue & 0xFF)
//
//                    let pixelOffset = x * 4
//                    rgbaRow[pixelOffset + 0] = highByte // R
//                    rgbaRow[pixelOffset + 1] = lowByte  // G
//                    rgbaRow[pixelOffset + 2] = 0        // B
//                    rgbaRow[pixelOffset + 3] = 0        // A
                    //Converting to turbo color map (made by google) - but latency kills it
                    let d = depthRow[x]
                    let normalized: Float
                                if d.isNaN {
                                    normalized = 0.0
                                } else {
                                    let v = Float(d)
                                    let clamped = max(min(v, DataConfig.maxDepth), DataConfig.minDepth)
                                    normalized = (clamped - DataConfig.minDepth) / (DataConfig.maxDepth - DataConfig.minDepth)
                                }

                                // Get Turbo index
                                let idx = Int(max(0, min(255, Int(normalized * 255.0))))

                    let (r, g, b, a) = TurboLUTManager.turboTableUInt8[idx]

                                let px = x * 4
                                rgbaRow[px + 0] = b     // BGRA
                                rgbaRow[px + 1] = g
                                rgbaRow[px + 2] = r
                                rgbaRow[px + 3] = 255   // full opacity
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
