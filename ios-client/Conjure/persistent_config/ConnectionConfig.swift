import AVFoundation
import Combine

/// Possible connection modes for video streaming and processing
enum ConnectionMode: String, CaseIterable, Identifiable, Codable {
    case onDevice = "On-device ML"
    case streamWebRTC = "WebRTC Video"
    case streamTCP = "TCP Stream"
    case streamUDP = "UDP Stream"

    var id: String { self.rawValue }
    var description: String {
        switch self {
        case .onDevice:
            return
                "Process video frames on the device using on-device ML models. Data is transferred to the server via WebRTC"
        case .streamWebRTC:
            return "Stream video frames to the server using WebRTC."
        case .streamTCP:
            return
                "Stream video frames to the server using TCP (preferably used with a USB connection)."
        case .streamUDP:
            return
                "Stream video frames to the server using UDP (preferably used with a USB connection)."
        }
    }
    static let `default` = ConnectionMode.onDevice
}

/// Server configurations to store known hosts
struct HostConfig: Identifiable, Codable, Equatable {
    let id: UUID
    var ipAddress: String
    var port: String
    var friendlyName: String?

    init(id: UUID = UUID(), ipAddress: String, port: String, friendlyName: String? = nil) {
        self.id = id
        self.ipAddress = ipAddress
        self.port = port
        self.friendlyName = friendlyName
    }

}

/// Misc. connection settings, including webRTC channel label, queue size, etc.
struct ConnectionConfig: Codable {
    var webRTCChannelLabel: String
    var queueSize: Int
    var connectionMode: ConnectionMode

    static let `default` = ConnectionConfig(
        webRTCChannelLabel: "hand_landmarks",
        queueSize: 1,
        connectionMode: .default
    )
}

/// Observable store for connection configurations. Includes ConnectionMode and list of HostConfigs
@MainActor
final class ConnectionConfigStore: ObservableObject {
    @Published var hosts: [HostConfig] = [] {
        didSet { saveHosts() }
    }
    @Published var connectionConfig: ConnectionConfig = ConnectionConfig.default {
        didSet { saveConnectionConfig() }
    }

    private let keyHosts = "hostConfigs"
    private let keyConnectionConfig = "connectionConfig"

    init() {
        loadHosts()
        loadConnectionConfig()
    }

    private func loadHosts() {
        guard let data = UserDefaults.standard.data(forKey: keyHosts) else {
            print("No saved host configurations found - key not found in UserDefaults.")
            hosts = []
            return
        }
        do {
            hosts = try JSONDecoder().decode([HostConfig].self, from: data)
        } catch {
            hosts = []
            print("Failed to load host configurations (json decoding error): \(error)")
            return
        }
    }
    private func saveHosts() {
        do {
            let data = try JSONEncoder().encode(hosts)
            UserDefaults.standard.set(data, forKey: keyHosts)
        } catch {
            print("Failed to save host configurations (json encoding error): \(error)")
        }
    }

    private func loadConnectionConfig() {
        guard let data = UserDefaults.standard.data(forKey: keyConnectionConfig) else {
            print("No saved connection configuration found - key not found in UserDefaults.")
            connectionConfig = ConnectionConfig.default
            return
        }
        do {
            connectionConfig = try JSONDecoder().decode(ConnectionConfig.self, from: data)
        } catch {
            connectionConfig = ConnectionConfig.default
            print("Failed to load connection configuration (json decoding error): \(error)")
            return
        }

    }
    private func saveConnectionConfig() {
        do {
            let data = try JSONEncoder().encode(connectionConfig)
            UserDefaults.standard.set(data, forKey: keyConnectionConfig)
        } catch {
            print("Failed to save connection configuration (json encoding error): \(error)")
        }
    }
}
