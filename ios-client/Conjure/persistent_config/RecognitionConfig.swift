import AVFoundation
import Combine

struct RecognitionConfig: Codable {
    var numHands: Int
    var landmarkDepthPixelRadius: Int
    var minDepth: Float  //TODO do we need min/max depth here? Especially for on device ML?
    var maxDepth: Float

    static let `default` = RecognitionConfig(
        numHands: 1,
        landmarkDepthPixelRadius: 2,
        minDepth: 0.1,
        maxDepth: 1.5
    )
}

@MainActor
final class RecognitionConfigStore: ObservableObject {
    @Published var recognitionConfig: RecognitionConfig = RecognitionConfig.default {
        didSet { saveRecognitionConfig() }
    }

    private let keyRecognitionConfig = "recognitionConfig"

    init() {
        loadRecognitionConfig()
    }

    private func loadRecognitionConfig() {
        guard let data = UserDefaults.standard.data(forKey: keyRecognitionConfig) else {
            print("No saved recognition configuration found, using default.")
            recognitionConfig = RecognitionConfig.default
            return
        }
        do {
            recognitionConfig = try JSONDecoder().decode(RecognitionConfig.self, from: data)
        } catch {
            print("Failed to load recognition configuration (json decoding error): \(error)")
            recognitionConfig = RecognitionConfig.default
        }
    }
    private func saveRecognitionConfig() {
        do {
            let data = try JSONEncoder().encode(recognitionConfig)
            UserDefaults.standard.set(data, forKey: keyRecognitionConfig)
        } catch {
            print("Failed to save recognition configuration (json encoding error): \(error)")
        }
    }
}
