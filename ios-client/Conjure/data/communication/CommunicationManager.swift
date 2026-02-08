//
//  CommunicationManager.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-02-08.
//

struct SendableConfigUpdate: Codable {
    let generalSettings: GeneralSettings
    let hostListSettings: HostListSettings
    let trackpadSettings: TrackpadSettings
    let recognitionSettings: RecognitionSettings
}

class CommunicationManager {
    var isConnected: Bool = false

    unowned let generalSettings: PersistentSettings<GeneralSettings>!
    unowned let hostListSettings: PersistentSettings<HostListSettings>!
    unowned let trackpadSettings: PersistentSettings<TrackpadSettings>!
    unowned let recognitionSettings: PersistentSettings<RecognitionSettings>!

    init(
        generalSettings: PersistentSettings<GeneralSettings>,
        hostListSettings: PersistentSettings<HostListSettings>,
        trackpadSettings: PersistentSettings<TrackpadSettings>,
        recognitionSettings: PersistentSettings<RecognitionSettings>,
    ) {
        self.generalSettings = generalSettings
        self.hostListSettings = hostListSettings
        self.trackpadSettings = trackpadSettings
        self.recognitionSettings = recognitionSettings
    }

    func startConnection() async -> String? {
        let errMsg = await startConnection_()
        if errMsg == nil {
            isConnected = true
        }
        return errMsg
    }
    func stopConnection() {
        stopConnection_()
        isConnected = false
    }

    func sendConfigUpdate() -> String? {
        if !isConnected {
            return "Communication client not connected"
        }
        let configUpdate = SendableConfigUpdate(
            generalSettings: generalSettings.value,
            hostListSettings: hostListSettings.value,
            trackpadSettings: trackpadSettings.value,
            recognitionSettings: recognitionSettings.value
        )
        return sendConfigUpdate_(configUpdate: configUpdate)
    }

    func send(frame: LandmarkedFrame) -> String? {
        if !isConnected {
            return "Communication client not connected"
        }
        return send_(frame: frame)
    }

    func startConnection_() async -> String? {
        fatalError("Not implemented")
    }
    func stopConnection_() {
        fatalError("Not implemented")
    }
    func sendConfigUpdate_(configUpdate: SendableConfigUpdate) -> String? {
        fatalError("Not implemented")
    }
    func send_(frame: LandmarkedFrame) -> String? {
        fatalError("Not implemented")
    }

    //TODO
    // func send(frame: TrackpadFrame) -> String? {}
}
