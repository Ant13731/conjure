//
//  Settings.swift
//  Conjure
//
//  Created by Anthony Hunt on 2026-01-20.
//

import AVFoundation
import Combine

protocol PersistentlyStorable: Codable, Equatable {
    static var defaultValue: Self { get }
    static var storageKey: String { get }
}

@MainActor
class PersistentSettings<T: PersistentlyStorable>: ObservableObject {

    @Published var value: T {
        didSet { save() }
    }

    init() {
        self.value = Self.load()
    }

    private static func load() -> T {
        guard let data = UserDefaults.standard.data(forKey: T.storageKey) else {
            print("No saved value found for key \(T.storageKey), using default.")
            return T.defaultValue
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            print("Failed to load value for key \(T.storageKey) (json decoding error): \(error)")
            return T.defaultValue
        }
    }

    private func save() {
        do {
            let data = try JSONEncoder().encode(value)
            UserDefaults.standard.set(data, forKey: T.storageKey)
        } catch {
            print("Failed to save value for key \(T.storageKey) (json encoding error): \(error)")
        }
    }
}
