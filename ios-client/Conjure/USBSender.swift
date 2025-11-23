//
//  USBSender.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-11-20.
//

import Foundation
import AVFoundation
import VideoToolbox
import Network
import UIKit

class USBSender {

    private var connection: NWConnection?
    private let port: UInt16
    private let host: NWEndpoint.Host = "172.20.10.7" // loopback for iproxy
    private var compressionSession: VTCompressionSession?

    init(port: Int) {
        self.port = UInt16(port)
        setupCompression()
    }

    /// Check if USB connection is possible (e.g., iproxy running)
    static func isUSBAvailable() -> Bool {
        UIDevice.current.isBatteryMonitoringEnabled = true
        // In practice, we cant actually check if a usb connection is possible
        return UIDevice.current.batteryState == .charging
//        return true
    }

    /// Connect TCP to host
    func connect(completion: @escaping (Bool) -> Void) {
        connection = NWConnection(
            host: host,
            port: NWEndpoint.Port(rawValue: port)!,
            using: .tcp
        )
        connection?.stateUpdateHandler = { state in
            switch state {
            case .ready:
                completion(true)
            case .failed(_), .cancelled:
                completion(false)
            default:
                print("Connection state", state)
                break
            }
        }
        connection?.start(queue: .global())
    }

    /// Set up H.264 compression for video frames
    private func setupCompression() {
        let width = 640
        let height = 480
        VTCompressionSessionCreate(allocator: nil,
                                   width: Int32(width),
                                   height: Int32(height),
                                   codecType: kCMVideoCodecType_H264,
                                   encoderSpecification: nil,
                                   imageBufferAttributes: nil,
                                   compressedDataAllocator: nil,
                                   outputCallback: compressionCallback,
                                   refcon: UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
                                   compressionSessionOut: &compressionSession)

        guard let session = compressionSession else {
            print("Failed to set up compression Session")
            return }
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, value: 1 as CFTypeRef)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_H264_Baseline_AutoLevel)
        VTCompressionSessionPrepareToEncodeFrames(session)
    }

    /// Send a single frame over USB
    func send(videoBuffer: CVImageBuffer, depthBuffer: CVPixelBuffer) {
        guard let connection = connection else {
            print("No connection available, dropping frame")
            return }
        guard let session = compressionSession else {
            print("Compression session not initialized")
            return }

        var flags: VTEncodeInfoFlags = []
        let status = VTCompressionSessionEncodeFrame(session,
                                        imageBuffer: videoBuffer,
                                        presentationTimeStamp: CMTime.invalid,
                                        duration: .invalid,
                                        frameProperties: nil,
                                        sourceFrameRefcon: UnsafeMutableRawPointer(Unmanaged.passUnretained(depthBuffer).toOpaque()),
                                        infoFlagsOut: &flags)
        guard status == noErr else {
            print("Failed to encode and send frame:", status)
            return
        }
    }

    /// H.264 compression callback
    private let compressionCallback: VTCompressionOutputCallback = { (outputCallbackRefCon,
                                                                     sourceFrameRefCon,
                                                                     status,
                                                                     infoFlags,
                                                                     sampleBuffer) in
        if status != noErr {
                print("[CompressionCallback]: Encoder returned error: \(status)")
                return
            }
            guard let sampleBuffer = sampleBuffer else {
                print("[CompressionCallback]: sampleBuffer is nil")
                return
            }
            guard CMSampleBufferDataIsReady(sampleBuffer) else {
                print("[CompressionCallback]: sampleBuffer data not ready")
                return
            }

        let sender = Unmanaged<USBSender>.fromOpaque(outputCallbackRefCon!).takeUnretainedValue()
        let depthBuffer = Unmanaged<CVPixelBuffer>.fromOpaque(sourceFrameRefCon!).takeUnretainedValue()

        // --- Extract H.264 NAL data ---
        guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var length: Int = 0
        var dataPointer: UnsafeMutablePointer<Int8>? = nil
        let dataStatus = CMBlockBufferGetDataPointer(
            dataBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )

        if dataStatus != noErr {
            print("[CompressionCallback]: Failed to access block buffer data: \(dataStatus)")
            return
        }

        guard let ptr = dataPointer else {
            print("[CompressionCallback]: dataPointer is nil")
            return
        }
        let videoData = Data(bytes: ptr, count: length)

        // --- Extract depth buffer as UInt16 ---
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        let depthSize = CVPixelBufferGetDataSize(depthBuffer)
        print(CVPixelBufferGetPlaneCount(depthBuffer))
        let depthPtr = CVPixelBufferGetBaseAddress(depthBuffer)!
        let depthData = Data(bytes: depthPtr, count: depthSize)
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)

        // --- Build packet ---
        // [depthSize (4 bytes) | videoSize (4 bytes) | depthData | videoData]
        var packet = Data()
        packet.append(withUnsafeBytes(of: UInt32(depthData.count).littleEndian, { Data($0) }))
        packet.append(withUnsafeBytes(of: UInt32(videoData.count).littleEndian, { Data($0) }))
        packet.append(depthData)
        packet.append(videoData)

        // --- Send over TCP ---
        sender.connection?.send(content: packet, completion: .contentProcessed({ sendError in
            if let sendError = sendError {
                print("[CompressionCallback]: TCP send failed: \(sendError)")
            }
        }))
    }

    /// Close connection
    func disconnect() {
        connection?.cancel()
        connection = nil
    }
}
