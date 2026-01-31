//
//  ConjureApp.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-10-26.
//

import Combine
import SwiftUI

enum StrError: Error {
    case msg(String)

    init(_ value: String) {
        self = .msg(value)
    }
}

enum Route: Hashable {
    case settings
    case settingsGeneral
    case settingsHostList
    case settingsTrackpad
    case settingsRecognition
}

@MainActor
final class Router: ObservableObject {
    @Published var path = NavigationPath()
}

@main
struct ConjureApp: App {
    @StateObject private var generalSettings = PersistentSettings<GeneralSettings>()
    @StateObject private var hostListSettings = PersistentSettings<HostListSettings>()
    @StateObject private var trackpadSettings = PersistentSettings<TrackpadSettings>()
    @StateObject private var recognitionSettings = PersistentSettings<RecognitionSettings>()
    @StateObject private var router = Router()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $router.path) {
                MainView()
                    .environmentObject(generalSettings)
                    .environmentObject(hostListSettings)
                    .environmentObject(trackpadSettings)
                    .environmentObject(recognitionSettings)
                    .environmentObject(router)

                    .navigationDestination(for: Route.self) { route in
                        switch route {
                        case .settings:
                            SettingsView()
                                .environmentObject(generalSettings)
                                .environmentObject(hostListSettings)
                                .environmentObject(trackpadSettings)
                                .environmentObject(recognitionSettings)
                                .environmentObject(router)
                        case .settingsHostList:
                            HostListView()
                                .environmentObject(generalSettings)
                                .environmentObject(hostListSettings)
                                .environmentObject(trackpadSettings)
                                .environmentObject(recognitionSettings)
                                .environmentObject(router)
                        case .settingsRecognition:
                            RecognitionView()
                                .environmentObject(generalSettings)
                                .environmentObject(hostListSettings)
                                .environmentObject(trackpadSettings)
                                .environmentObject(recognitionSettings)
                                .environmentObject(router)
                        case .settingsTrackpad:
                            TrackpadView()
                                .environmentObject(generalSettings)
                                .environmentObject(hostListSettings)
                                .environmentObject(trackpadSettings)
                                .environmentObject(recognitionSettings)
                                .environmentObject(router)
                        case .settingsGeneral:
                            GeneralView()
                                .environmentObject(generalSettings)
                                .environmentObject(hostListSettings)
                                .environmentObject(trackpadSettings)
                                .environmentObject(recognitionSettings)
                                .environmentObject(router)
                        }
                    }
            }
        }
    }
}
