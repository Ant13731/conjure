//
//  UDPManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-02-08.
//

import AVFoundation
import Accelerate
import Foundation
import Network
import UIKit
import VideoToolbox

class UDPManager: CommunicationManager {
    private var connection: NWConnection!
    private let udpQueue = DispatchQueue(label: "UDP Communication Queue")
    private var hasRetried = false

    private var waitForAck = true
    private let configRetryInterval: UInt64 = 500_000_000  // 500ms

    override func startConnection_() async -> String? {
        guard let currentHost = hostListSettings.value.currentHost else {
            return "No current host selected"
        }

        let host = NWEndpoint.Host(currentHost.ipAddress)
        let port = NWEndpoint.Port(rawValue: UInt16(currentHost.port)!)!

        connection = NWConnection(
            host: host,
            port: port,
            using: .udp
        )

        connection!.stateUpdateHandler = { [weak self] state in
            guard let self else {
                print("UDPManager: couldn't get weak ref to self in stateUpdateHandler")
                return
            }

            Task { @MainActor in
                switch state {

                case .failed(let error):
                    self.isConnected = false
                    print("UDP connection failed with error: \(error)")
                    self.stopConnection()
                    print("Attempting to reconnect...")
                    guard !self.hasRetried else {
                        print("UDP reconnection already attempted, not retrying")
                        return
                    }
                    self.hasRetried = true
                    let errMsg = await self.startConnection()
                    if let errMsg {
                        print("UDP reconnection failed: \(errMsg)")
                    } else {
                        print("UDP reconnection successful")
                    }
                    break
                case .ready:
                    self.isConnected = true
                    self.hasRetried = false
                    print("UDP connection ready")
                    self.startReceiver()
                    break
                case .cancelled:
                    self.isConnected = false
                default:
                    print("UDP connection state changed: \(state)")
                    break
                }
            }
        }
        connection!.start(queue: udpQueue)
        return nil
    }
    // Design note: we want to make sure configs are received by the server,
    // so we implement a retry mechanism until we receive an ACK back from the server.
    // Stream frames are send-and-forget
    func startReceiver() {
        connection!.receiveMessage { [weak self] data, _, _, error in
            guard let self else {
                print("UDPManager: couldn't get weak ref to self in stateUpdateHandler")
                return
            }

            if let data {
                guard let packetType = data.first else { return }

                switch packetType {
                case PacketType.ack.rawValue:
                    print("Config ACK received")
                    waitForAck = false
                    break
                default:
                    print("Received packet with unknown type: \(packetType)")
                }
            }

            if error == nil {
                Task { @MainActor in
                    self.startReceiver()
                }
            }
        }
    }

    override func stopConnection_() {
        connection?.cancel()
        connection = nil
        isConnected = false
        hasRetried = false
    }

    override func sendConfigUpdate_(data: Data) {
        waitForAck = true
        Task {
            let start = ContinuousClock.now
            while waitForAck {
                if start.duration(to: .now) > .seconds(5) {
                    print("Config ACK timeout")
                    waitForAck = false
                    break
                }

                send_(data: data)
                try? await Task.sleep(nanoseconds: configRetryInterval)
            }
        }
    }

    override func send_(data: Data) {
        connection.send(
            content: data,
            completion: .contentProcessed { sendError in
                if let sendError = sendError {
                    print("UDP send failed: \(sendError)")
                }
            })
    }

}
