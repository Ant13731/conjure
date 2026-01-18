//
//  ConjureApp.swift
//  Conjure
//
//  Created by Anthony Hunt on 2025-10-26.
//

import Combine
import SwiftUI

enum Route: Hashable {
    case settings
    case settingsHostList
    case settingsRecognition
}

@MainActor
final class Router: ObservableObject {
    @Published var path = NavigationPath()
}

@main
struct ConjureApp: App {
    @StateObject private var connectionConfigStore = ConnectionConfigStore()
    @StateObject private var recognitionConfigStore = RecognitionConfigStore()
    @StateObject private var router = Router()

    var body: some Scene {
        WindowGroup {
            NavigationStack(path: $router.path) {
                MainView()
                    .environmentObject(connectionConfigStore)
                    .environmentObject(recognitionConfigStore)
                    .environmentObject(router)
                    .navigationDestination(for: Route.self) { route in
                        switch route {
                        case .settings:
                            SettingsView()
                                .environmentObject(connectionConfigStore)
                                .environmentObject(recognitionConfigStore)
                                .environmentObject(router)
                        case .settingsHostList:
                            HostListView()
                                .environmentObject(connectionConfigStore)
                                .environmentObject(recognitionConfigStore)
                                .environmentObject(router)
                        case .settingsRecognition:
                            RecognitionConfigView()
                                .environmentObject(connectionConfigStore)
                                .environmentObject(recognitionConfigStore)
                                .environmentObject(router)
                        }
                    }
            }
        }
    }
}
