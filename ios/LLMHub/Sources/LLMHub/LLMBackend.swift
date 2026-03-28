import Foundation
import LlamaCPPRuntime
import RunAnywhere
#if canImport(UIKit)
import UIKit
import ImageIO
#endif

@MainActor
class LLMBackend: ObservableObject {
    static let shared = LLMBackend()

    @Published var isLoaded: Bool = false
    @Published var currentlyLoadedModel: String? = nil
    @Published var isBackendLoading: Bool = false
    @Published var loadedContextWindow: Int? = nil

    // Generation parameters
    var maxTokens: Int = 2048
    var contextWindow: Int = 2048
    var topK: Int = 64
    var topP: Float = 0.95
    var temperature: Float = 1.0
    var selectedBackend: String = "GPU"
    var enableVision: Bool = true
    var enableAudio: Bool = true
    var enableThinking: Bool = true

    private var isSDKInitialized = false
    private var areModelsRegistered = false
    private var loadedLLMModelId: String?
    private var loadedVLMModelId: String?
    private var loadedVLMProjectorPath: String?

    private init() {}

    private func legacyModelDirectory(for model: AIModel) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
    }

    private func hasAllRequiredFiles(in directory: URL, for model: AIModel) -> Bool {
        guard !model.requiredFileNames.isEmpty else { return false }
        return model.requiredFileNames.allSatisfy { fileName in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(fileName).path)
        }
    }

    private func migrateLegacyModelIfNeeded(_ model: AIModel) throws -> Bool {
        if RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) {
            return false
        }

        guard let legacyDir = legacyModelDirectory(for: model),
              FileManager.default.fileExists(atPath: legacyDir.path),
              hasAllRequiredFiles(in: legacyDir, for: model) else {
            return false
        }

        let destinationDir = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)

        for fileName in model.requiredFileNames {
            let sourceURL = legacyDir.appendingPathComponent(fileName)
            let destinationURL = destinationDir.appendingPathComponent(fileName)

            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try? FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        }

        print("[LLMBackend] migrated legacy model files for \(model.id)")
        return true
    }

    private func filename(from url: URL) -> String {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?.path.split(separator: "/").last.map(String.init) ?? url.lastPathComponent
    }

    private func loadedAIModel() -> AIModel? {
        guard let modelName = currentlyLoadedModel else { return nil }
        return ModelData.models.first(where: { $0.name == modelName })
    }

    private func framework(for model: AIModel) -> InferenceFramework {
        model.inferenceFramework
    }

    private func listGGUFFiles(in directory: URL) -> [URL] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return contents
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .sorted { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() }
    }

    private func resolveModelGGUFPath(for model: AIModel) throws -> String {
        let folderURL = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        let files = listGGUFFiles(in: folderURL)

        if let modelURL = URL(string: model.url) {
            let preferredFilename = filename(from: modelURL).lowercased()
            if let exact = files.first(where: { $0.lastPathComponent.lowercased() == preferredFilename }) {
                return exact.path
            }
        }

        if let quantTag = quantizationTag(from: model.name),
           let quantMatched = files.first(where: {
               let lower = $0.lastPathComponent.lowercased()
               return !lower.contains("mmproj") && lower.contains(quantTag)
           }) {
            return quantMatched.path
        }

        if let preferred = files.first(where: { !$0.lastPathComponent.lowercased().contains("mmproj") }) {
            return preferred.path
        }

        if let first = files.first {
            return first.path
        }

        throw NSError(domain: "LLMBackend", code: -101, userInfo: [NSLocalizedDescriptionKey: "Main GGUF file not found for model \(model.name)"])
    }

    private func quantizationTag(from modelName: String) -> String? {
        guard let leftParen = modelName.lastIndex(of: "("),
              let rightParen = modelName.lastIndex(of: ")"),
              leftParen < rightParen else {
            return nil
        }
        let tag = modelName[modelName.index(after: leftParen)..<rightParen]
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return tag.isEmpty ? nil : tag
    }

    private func familyStem(from modelName: String) -> String {
        modelName
            .replacingOccurrences(of: "\\s*\\([^)]*\\)\\s*$", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private func resolveVisionProjectorPath(for model: AIModel) -> String? {
        let stem = familyStem(from: model.name)
        let quantTag = quantizationTag(from: model.name)

        let candidates = ModelData.models.filter {
            $0.isDependencyOnly
                && $0.inferenceFramework == model.inferenceFramework
                && RunAnywhere.isModelDownloaded($0.id, framework: $0.inferenceFramework)
        }

        let scored = candidates.compactMap { candidate -> (score: Int, path: String)? in
            let candidateName = candidate.name.lowercased()
            var score = 0

            if candidateName.contains(stem) {
                score += 3
            }
            if let quantTag,
               candidateName.contains(quantTag) || candidate.url.lowercased().contains(quantTag) {
                score += 3
            }
            if candidateName.contains("vision projector") || candidateName.contains("mmproj") {
                score += 1
            }

            guard let folderURL = try? SimplifiedFileManager.shared.getModelFolderURL(modelId: candidate.id, framework: candidate.inferenceFramework) else {
                return nil
            }

            let files = listGGUFFiles(in: folderURL)
            guard let mmprojFile = files.first(where: { $0.lastPathComponent.lowercased().contains("mmproj") }) ?? files.first else {
                return nil
            }

            return (score, mmprojFile.path)
        }
        .sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.path < rhs.path
            }
            return lhs.score > rhs.score
        }

        return scored.first?.path
    }

    private func ensureVLMLoaded(for model: AIModel) async throws {
        let modelPath = try resolveModelGGUFPath(for: model)
        let mmprojPath = resolveVisionProjectorPath(for: model)

        guard let mmprojPath, !mmprojPath.isEmpty else {
            throw NSError(
                domain: "LLMBackend",
                code: -102,
                userInfo: [NSLocalizedDescriptionKey: "Vision projector (mmproj) is missing for \(model.name)"]
            )
        }

        let modelSummary = fileSummary(at: modelPath)
        let projectorSummary = fileSummary(at: mmprojPath)
        let modelHeader = ggufHeaderSummary(at: modelPath)
        let projectorHeader = ggufHeaderSummary(at: mmprojPath)

        print("[LLMBackend] VLM prepare model=\(model.id) main={\(modelSummary)} header={\(modelHeader)} mmproj={\(projectorSummary)} header={\(projectorHeader)}")

        guard modelHeader == "GGUF" else {
            throw NSError(
                domain: "LLMBackend",
                code: -103,
                userInfo: [NSLocalizedDescriptionKey: "Main model file is invalid: \(modelSummary) header=\(modelHeader)"]
            )
        }

        guard projectorHeader == "GGUF" else {
            throw NSError(
                domain: "LLMBackend",
                code: -104,
                userInfo: [NSLocalizedDescriptionKey: "Vision projector file is invalid: \(projectorSummary) header=\(projectorHeader)"]
            )
        }

        let shouldReload = !((await RunAnywhere.isVLMModelLoaded)
            && loadedVLMModelId == model.id
            && loadedVLMProjectorPath == mmprojPath)

        guard shouldReload else { return }

        if await RunAnywhere.isModelLoaded {
            do {
                try await RunAnywhere.unloadModel()
                loadedLLMModelId = nil
                print("[LLMBackend] Unloaded text LLM before VLM load to avoid duplicate model residency")
            } catch {
                print("[LLMBackend] Failed to unload text LLM before VLM load: \(error)")
            }
        }

        await RunAnywhere.unloadVLMModel()
        do {
            try await RunAnywhere.loadVLMModel(modelPath, mmprojPath: mmprojPath, modelId: model.id, modelName: model.name)
        } catch {
            let details = "main={\(modelSummary)} header={\(modelHeader)} mmproj={\(projectorSummary)} header={\(projectorHeader)}"
            throw NSError(
                domain: "LLMBackend",
                code: -111,
                userInfo: [NSLocalizedDescriptionKey: "VLM load failed for \(model.name): \(error.localizedDescription). \(details)"]
            )
        }
        loadedVLMModelId = model.id
        loadedVLMProjectorPath = mmprojPath
    }

    private func ensureTextModelLoaded(for model: AIModel) async throws {
        if await RunAnywhere.isVLMModelLoaded {
            await RunAnywhere.unloadVLMModel()
            loadedVLMModelId = nil
            loadedVLMProjectorPath = nil
            print("[LLMBackend] Unloaded VLM before text generation to avoid duplicate model residency")
        }

        let shouldLoad = !((await RunAnywhere.isModelLoaded) && loadedLLMModelId == model.id)
        guard shouldLoad else { return }

        try await RunAnywhere.loadModel(model.id)
        loadedLLMModelId = model.id
    }

    private func fileSummary(at path: String) -> String {
        let url = URL(fileURLWithPath: path)
        let name = url.lastPathComponent

        guard FileManager.default.fileExists(atPath: path) else {
            return "\(name) missing"
        }

        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? -1
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        let sizeLabel = size >= 0 ? formatter.string(fromByteCount: size) : "unknown"

        return "\(name) \(sizeLabel)"
    }

    private func ggufHeaderSummary(at path: String) -> String {
        guard let handle = FileHandle(forReadingAtPath: path) else { return "unreadable" }
        defer {
            try? handle.close()
        }

        let data = handle.readData(ofLength: 4)
        guard data.count == 4, let header = String(data: data, encoding: .ascii) else {
            return "invalid"
        }
        return header
    }

#if canImport(UIKit)
    private func downsampledUIImage(from imageURL: URL, maxDimension: CGFloat = 1024) -> UIImage? {
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, sourceOptions) else {
            return nil
        }

        let downsampleOptions = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceThumbnailMaxPixelSize: maxDimension,
        ] as CFDictionary

        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, downsampleOptions) else {
            return nil
        }

        return UIImage(cgImage: cgImage)
    }
#endif

    private func vlmImage(from imageURL: URL) -> VLMImage {
        #if canImport(UIKit)
        if let uiImage = downsampledUIImage(from: imageURL) {
            return VLMImage(image: uiImage)
        }
        #endif
        return VLMImage(filePath: imageURL.path)
    }

    private func modelMaxContextWindow(for model: AIModel) -> Int {
        let advertised = model.contextWindowSize > 0 ? model.contextWindowSize : 2048
        return max(1, advertised)
    }

    func clampedContextWindow(_ requested: Int, for model: AIModel) -> Int {
        min(max(1, requested), modelMaxContextWindow(for: model))
    }

    private func registerModel(_ model: AIModel, contextLengthOverride: Int? = nil) {
        guard let primaryURL = URL(string: model.url) else { return }
        _ = contextLengthOverride

        if model.additionalFiles.isEmpty {
            RunAnywhere.registerModel(
                id: model.id,
                name: model.name,
                url: primaryURL,
                framework: framework(for: model),
                modality: model.supportsVision ? .multimodal : .language,
                memoryRequirement: model.sizeBytes,
                supportsThinking: model.supportsThinking
            )
            return
        }

        let descriptors = model.allDownloadURLs.map {
            ModelFileDescriptor(url: $0, filename: filename(from: $0), isRequired: true)
        }

        RunAnywhere.registerMultiFileModel(
            id: model.id,
            name: model.name,
            files: descriptors,
            framework: framework(for: model),
            modality: model.supportsVision ? .multimodal : .language,
            memoryRequirement: model.sizeBytes
        )
    }

    private func ensureSDKReady() async throws {
        if !isSDKInitialized {
            try RunAnywhere.initialize(environment: .development)
            LlamaCPP.register()
            isSDKInitialized = true
        }

        if !areModelsRegistered {
            for model in ModelData.models {
                registerModel(model)
            }
            areModelsRegistered = true
        }

        // Ensure model path APIs are configured before storage checks/migration.
        try await RunAnywhere.completeServicesInitialization()
    }

    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }

        print("[LLMBackend] loadModel name=\(model.name) visionEnabled=\(enableVision) audioEnabled=\(enableAudio)")

        try await ensureSDKReady()
        let effectiveContext = clampedContextWindow(contextWindow, for: model)
        registerModel(model, contextLengthOverride: effectiveContext)
        await RunAnywhere.flushPendingRegistrations()
        _ = try? migrateLegacyModelIfNeeded(model)
        _ = await RunAnywhere.discoverDownloadedModels()

        // Only local load here. Downloads are handled by the model download screen.
        guard RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) else {
            throw NSError(domain: "LLMBackend", code: -100, userInfo: [NSLocalizedDescriptionKey: "Model is not downloaded locally"])
        }

        try await RunAnywhere.loadModel(model.id)

        isLoaded = true
        currentlyLoadedModel = model.name
        loadedContextWindow = effectiveContext
        loadedLLMModelId = model.id
        loadedVLMModelId = nil
        loadedVLMProjectorPath = nil
    }

    func unloadModel() {
        Task {
            do {
                try await RunAnywhere.unloadModel()
            } catch {
                print("[LLMBackend] unloadModel error=\(error)")
            }
            await RunAnywhere.unloadVLMModel()
            await MainActor.run {
                self.isLoaded = false
                self.currentlyLoadedModel = nil
                self.loadedContextWindow = nil
                self.loadedLLMModelId = nil
                self.loadedVLMModelId = nil
                self.loadedVLMProjectorPath = nil
            }
        }
    }

    func generate(
        prompt: String,
        imageURL: URL? = nil,
        audioURL: URL? = nil,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        _ = imageURL
        _ = audioURL

        try await ensureSDKReady()

        let effectiveMaxTokens: Int = {
            if let model = loadedAIModel() {
                let effectiveContext = clampedContextWindow(contextWindow, for: model)
                return min(max(1, maxTokens), effectiveContext)
            }
            return max(1, maxTokens)
        }()

        let options = LLMGenerationOptions(
            maxTokens: effectiveMaxTokens,
            temperature: temperature,
            topP: topP,
            streamingEnabled: true
        )

        if let imageURL,
           enableVision,
           let model = loadedAIModel(),
           model.supportsVision {
            try await ensureVLMLoaded(for: model)

            let image = vlmImage(from: imageURL)
            let streamResult = try await RunAnywhere.processImageStream(
                image,
                prompt: prompt,
                maxTokens: Int32(effectiveMaxTokens),
                temperature: temperature,
                topP: topP
            )

            var currentOutput = ""
            for try await token in streamResult.stream {
                currentOutput += token
                onUpdate(currentOutput, 0, 0)
            }

            let result = try await streamResult.metrics.value
            onUpdate(currentOutput, result.completionTokens, result.tokensPerSecond)
            return
        }

        if let model = loadedAIModel() {
            try await ensureTextModelLoaded(for: model)
        }

        print("[LLMBackend] generate visionEnabled=\(enableVision) audioEnabled=\(enableAudio) images=0 videos=0")

        let streamResult = try await RunAnywhere.generateStream(prompt, options: options)
        var currentOutput = ""

        for try await token in streamResult.stream {
            currentOutput += token
            onUpdate(currentOutput, 0, 0)
        }

        let result = try await streamResult.result.value
        onUpdate(currentOutput, result.tokensUsed, result.tokensPerSecond)
    }
}
