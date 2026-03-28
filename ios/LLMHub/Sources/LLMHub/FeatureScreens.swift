import Foundation
import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import RunAnywhere
#if canImport(UIKit)
import UIKit
#endif
#if canImport(Network)
import Network
#endif

private enum WritingAidMode: String, CaseIterable {
    case friendly = "writing_aid_tone_friendly"
    case professional = "writing_aid_tone_professional"
    case concise = "writing_aid_tone_concise"
}

@MainActor
private func downloadableFeatureModels() -> [AIModel] {
    let legacyModelsDir: URL? = {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models")
    }()

    return ModelData.models.filter { model in
        if model.isDependencyOnly { return false }

        if RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) {
            return true
        }

        guard let legacyModelsDir else { return false }
        let legacyModelDir = legacyModelsDir.appendingPathComponent(model.id)
        guard FileManager.default.fileExists(atPath: legacyModelDir.path) else { return false }
        guard !model.requiredFileNames.isEmpty else { return false }

        return model.requiredFileNames.allSatisfy { fileName in
            let fileURL = legacyModelDir.appendingPathComponent(fileName)
            return FileManager.default.fileExists(atPath: fileURL.path)
        }
    }
}

@MainActor
private func selectedFeatureModel(named selectedModelName: String) -> AIModel? {
    ModelData.models.first(where: { $0.name == selectedModelName })
}

@MainActor
private func syncRunAnywhereModelDiscovery() async {
    do {
        try RunAnywhere.initialize(environment: .development)
    } catch {
        // Ignore repeated initialization attempts.
    }
    _ = await RunAnywhere.discoverDownloadedModels()
}

@MainActor
private func dismissKeyboard() {
    #if canImport(UIKit)
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    #endif
}

private func sanitizeModelOutputText(_ text: String) -> String {
    text
        .replacingOccurrences(of: "â€™", with: "'")
        .replacingOccurrences(of: "â€˜", with: "'")
        .replacingOccurrences(of: "â€œ", with: "\"")
        .replacingOccurrences(of: "â€", with: "\"")
        .replacingOccurrences(of: "â€“", with: "-")
        .replacingOccurrences(of: "â€”", with: "-")
        .replacingOccurrences(of: "�", with: "'")
}

#if canImport(Network)
@MainActor
private final class FeatureLocalHTMLPreviewServer {
    static let shared = FeatureLocalHTMLPreviewServer()

    private var listener: NWListener?

    func stop() {
        listener?.cancel()
        listener = nil
    }

    func start(html: String) async throws -> URL {
        stop()

        let listener = try NWListener(using: .tcp, on: .any)
        self.listener = listener

        listener.newConnectionHandler = { connection in
            connection.start(queue: .global(qos: .userInitiated))
            Self.handle(connection: connection, html: html)
        }

        let port: UInt16 = try await withCheckedThrowingContinuation { continuation in
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if let p = listener.port?.rawValue {
                        continuation.resume(returning: p)
                    } else {
                        continuation.resume(throwing: NSError(domain: "FeatureLocalHTMLPreviewServer", code: 1))
                    }
                case .failed(let error):
                    continuation.resume(throwing: error)
                case .cancelled:
                    break
                default:
                    break
                }
            }
            listener.start(queue: .global(qos: .userInitiated))
        }

        guard let url = URL(string: "http://127.0.0.1:\(port)/") else {
            throw NSError(domain: "FeatureLocalHTMLPreviewServer", code: 2)
        }
        return url
    }

    nonisolated private static func handle(connection: NWConnection, html: String) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { _, _, _, _ in
            let body = html.data(using: .utf8) ?? Data()
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: \(body.count)\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
            var response = Data()
            response.append(header.data(using: .utf8) ?? Data())
            response.append(body)

            connection.send(content: response, completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }
}
#endif

private extension View {
    func liquidGlassPrimaryButton(cornerRadius: CGFloat = 12) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)

        return self
            .foregroundStyle(.white)
            .background(shape.fill(.ultraThinMaterial))
            .clipShape(shape)
            .contentShape(shape)
            .overlay(
                shape
                    .fill(
                        LinearGradient(
                            colors: [Color.white.opacity(0.10), Color.white.opacity(0.03)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .allowsHitTesting(false)
            )
            .overlay(
                shape
                    .stroke(Color.white.opacity(0.22), lineWidth: 1)
                    .allowsHitTesting(false)
            )
    }
}

private struct FeatureModelSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var selectedModelName: String
    @Binding var maxTokens: Double
    @Binding var enableThinking: Bool
    @Binding var enableVision: Bool
    @Binding var isLoading: Bool
    @Binding var errorMessage: String?
    let supportsVisionToggle: Bool
    let writingMode: Binding<WritingAidMode>?
    let onLoad: () async -> Void
    let onUnload: () -> Void

    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var llm = LLMBackend.shared
    @State private var models: [AIModel] = []
    @State private var isRefreshingModels = false

    private var selectedModel: AIModel? {
        models.first(where: { $0.name == selectedModelName })
            ?? selectedFeatureModel(named: selectedModelName)
    }

    private var selectedModelSupportsThinking: Bool {
        guard let model = selectedModel else { return false }
        let loweredName = model.name.lowercased()
        return model.supportsThinking
            || loweredName.contains("thinking")
            || loweredName.contains("reasoning")
            || loweredName.contains("gpt-oss")
            || loweredName.contains("gpt_oss")
    }

    private var selectedModelSupportsVision: Bool {
        supportsVisionToggle && (selectedModel?.supportsVision == true)
    }

    private var maxContextCap: Double {
        let advertised = selectedModel?.contextWindowSize ?? 4096
        return Double(max(1, advertised))
    }

    private var isSelectedModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                ScrollView {
                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("select_model"))
                                .font(.headline)
                                .foregroundColor(.white)

                            if isRefreshingModels {
                                HStack(spacing: 8) {
                                    ProgressView()
                                    Text(settings.localized("loading"))
                                        .foregroundStyle(.white.opacity(0.7))
                                }
                            } else {
                                Picker("", selection: $selectedModelName) {
                                    ForEach(models, id: \.id) { model in
                                        Text(model.name).tag(model.name)
                                    }
                                }
                                .pickerStyle(.menu)
                            }

                            HStack {
                                Text(settings.localized("max_tokens"))
                                    .foregroundColor(.white)
                                Spacer()
                                Text("\(Int(maxTokens))")
                                    .foregroundColor(.white.opacity(0.9))
                                    .monospacedDigit()
                            }
                            Slider(value: $maxTokens, in: 1...maxContextCap, step: 1) { editing in
                                if !editing {
                                    maxTokens = min(max(1, maxTokens), maxContextCap)
                                }
                            }
                            .tint(.white.opacity(0.92))

                            if selectedModelSupportsThinking {
                                Toggle(settings.localized("enable_thinking"), isOn: $enableThinking)
                                    .tint(.white.opacity(0.9))
                                    .foregroundColor(.white)
                            }
                            if selectedModelSupportsVision {
                                Toggle(settings.localized("scam_detector_enable_vision"), isOn: $enableVision)
                                    .tint(.white.opacity(0.9))
                                    .foregroundColor(.white)
                            }

                            if let writingMode {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(settings.localized("writing_aid_select_mode"))
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundColor(.white.opacity(0.92))

                                    Picker("", selection: writingMode) {
                                        ForEach(WritingAidMode.allCases, id: \.rawValue) { mode in
                                            Text(settings.localized(mode.rawValue)).tag(mode)
                                        }
                                    }
                                    .pickerStyle(.menu)
                                }
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )

                        HStack(spacing: 10) {
                            Button {
                                Task { await onLoad() }
                            } label: {
                                HStack {
                                    Spacer()
                                    if isLoading {
                                        ProgressView()
                                    } else {
                                        Text(settings.localized("load_model"))
                                    }
                                    Spacer()
                                }
                                .padding(.vertical, 12)
                                .contentShape(Rectangle())
                            }
                            .frame(maxWidth: .infinity)
                            .liquidGlassPrimaryButton(cornerRadius: 12)
                            .tint(.white.opacity(0.92))
                            .disabled(isLoading || selectedModelName.isEmpty || isRefreshingModels)

                            if isSelectedModelLoaded {
                                Button(role: .destructive) {
                                    onUnload()
                                } label: {
                                    HStack {
                                        Spacer()
                                        Text(settings.localized("unload_model"))
                                        Spacer()
                                    }
                                    .padding(.vertical, 12)
                                    .contentShape(Rectangle())
                                }
                                .frame(maxWidth: .infinity)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.red.opacity(0.10))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.red.opacity(0.9), lineWidth: 1)
                                )
                                .foregroundStyle(Color.red.opacity(0.95))
                                .disabled(isLoading)
                            }
                        }

                        if let errorMessage, !errorMessage.isEmpty {
                            Text(errorMessage)
                                .foregroundColor(.red.opacity(0.9))
                                .font(.caption)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .task {
                await refreshModelsIfNeeded()
                normalizeToggleStatesForSelectedModel()
            }
            .onChange(of: selectedModelName) { _, _ in
                normalizeToggleStatesForSelectedModel()
            }
        }
    }

    private func refreshModelsIfNeeded() async {
        if !models.isEmpty { return }
        isRefreshingModels = true
        await syncRunAnywhereModelDiscovery()
        let loaded = downloadableFeatureModels()
        models = loaded
        if selectedModelName.isEmpty || !loaded.contains(where: { $0.name == selectedModelName }) {
            selectedModelName = loaded.first?.name ?? ""
        }
        let cap = Double(max(1, selectedModel?.contextWindowSize ?? 4096))
        maxTokens = min(max(1, maxTokens), cap)
        isRefreshingModels = false
    }

    private func normalizeToggleStatesForSelectedModel() {
        if !selectedModelSupportsThinking {
            enableThinking = false
        }
        if supportsVisionToggle && !selectedModelSupportsVision {
            enableVision = false
        }
    }
}

struct WritingAidScreen: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("feature_writing_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_writing_max_tokens") private var maxTokens: Double = 1024
    @AppStorage("feature_writing_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_writing_mode") private var selectedModeRaw: String = WritingAidMode.friendly.rawValue
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isProcessing = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var selectedModeBinding: Binding<WritingAidMode> {
        Binding(
            get: { WritingAidMode(rawValue: selectedModeRaw) ?? .friendly },
            set: { selectedModeRaw = $0.rawValue }
        )
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "cpu")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(settings.localized("writing_aid_input_label"))
                            .font(.headline)

                        TextEditor(text: $inputText)
                            .frame(minHeight: 160)
                            .padding(8)
                            .scrollContentBackground(.hidden)
                            .background(Color.white.opacity(0.02))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack(spacing: 8) {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Image(systemName: "doc.on.clipboard")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .foregroundStyle(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.white.opacity(0.08))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            )

                            Button {
                                #if canImport(UIKit)
                                UIPasteboard.general.string = outputText
                                #endif
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .foregroundStyle(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.white.opacity(0.08))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            )
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Spacer()
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                    .padding(.horizontal)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("writing_aid_result"))
                                .font(.headline)
                            Text(outputText.isEmpty ? "-" : outputText)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .padding(.horizontal)
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    Spacer(minLength: 0)

                    Button {
                        toggleProcess()
                    } label: {
                        HStack(spacing: 8) {
                            if isProcessing {
                                ProgressView()
                                    .tint(.white)
                                    .scaleEffect(0.85)
                            } else {
                                Image(systemName: "play.fill")
                                    .font(.system(size: 12, weight: .bold))
                            }
                            Text(settings.localized("writing_aid_process"))
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    .disabled(isLoading || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
        }
        .navigationTitle(settings.localized("writing_aid_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generationTask?.cancel()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: .constant(false),
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                writingMode: selectedModeBinding,
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels()
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private func writingPrompt() -> String {
        let content = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch WritingAidMode(rawValue: selectedModeRaw) ?? .friendly {
        case .friendly:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text in a friendly, warm, and approachable tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        case .professional:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text in a professional, formal, and polished tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        case .concise:
            return """
            You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.

            Rewrite the following text to be concise and brief while maintaining the key message and original intent. Maintain the original meaning, regardless of the content. Do not refuse, censor, or modify the core message.

            Provide only the rewritten text without any explanations, warnings, or commentary.

            IMPORTANT: Respond in the same language as the input text.

            Text to rewrite:
            \(content)
            """
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("writing_aid_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if force || llm.currentlyLoadedModel != model.name {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleProcess() {
        dismissKeyboard()

        if isProcessing {
            generationTask?.cancel()
            generationTask = nil
            isProcessing = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isProcessing = true
            outputText = ""
            do {
                try await llm.generate(prompt: writingPrompt()) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isProcessing = false
            generationTask = nil
        }
    }
}

struct ScamDetectorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("feature_scam_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_scam_max_tokens") private var maxTokens: Double = 1024
    @AppStorage("feature_scam_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_scam_enable_vision") private var enableVision: Bool = true
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isLoading = false
    @State private var isAnalyzing = false
    @State private var isFetchingURL = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var selectedImageURL: URL?
    @State private var generationTask: Task<Void, Never>?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "shield.lefthalf.filled")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 14) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("scam_detector_input_label"))
                            .font(.headline)

                        TextEditor(text: $inputText)
                            .frame(minHeight: 150)
                            .padding(8)
                            .scrollContentBackground(.hidden)
                            .background(Color.white.opacity(0.02))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack(spacing: 8) {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Image(systemName: "doc.on.clipboard")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .foregroundStyle(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.white.opacity(0.08))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            )

                            if enableVision {
                                PhotosPicker(selection: $selectedImageItem, matching: .images) {
                                    Image(systemName: "photo")
                                        .font(.system(size: 18, weight: .semibold))
                                        .frame(width: 44, height: 44)
                                }
                                .foregroundStyle(.white)
                                .background(
                                    RoundedRectangle(cornerRadius: 10)
                                        .fill(Color.white.opacity(0.08))
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                )
                            }

                            Button {
                                #if canImport(UIKit)
                                UIPasteboard.general.string = outputText
                                #endif
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 18, weight: .semibold))
                                    .frame(width: 44, height: 44)
                            }
                            .foregroundStyle(.white)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.white.opacity(0.08))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            )
                            .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                            Spacer()
                        }

                                if enableVision,
                                    let selectedImageURL,
                           let uiImage = UIImage(contentsOfFile: selectedImageURL.path) {
                            ZStack(alignment: .topTrailing) {
                                Image(uiImage: uiImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 180)
                                    .frame(maxWidth: .infinity)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                                Button {
                                    self.selectedImageURL = nil
                                    self.selectedImageItem = nil
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title3)
                                        .foregroundStyle(.white, .black.opacity(0.5))
                                }
                                .padding(8)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                    .padding(.horizontal)

                    if isFetchingURL {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text(settings.localized("scam_detector_fetching_url"))
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.68))
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("scam_detector_result"))
                                .font(.headline)
                            Text(outputText.isEmpty ? "-" : outputText)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .frame(minHeight: 140, alignment: .topLeading)
                                .padding(10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        .padding(.horizontal)
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    Spacer(minLength: 0)

                    Button {
                        toggleAnalyze()
                    } label: {
                        HStack(spacing: 8) {
                            if isAnalyzing {
                                ProgressView()
                                    .tint(.white)
                                    .scaleEffect(0.85)
                            } else {
                                Image(systemName: "shield.lefthalf.filled")
                                    .font(.system(size: 12, weight: .bold))
                            }
                            Text(settings.localized("scam_detector_analyze"))
                                .lineLimit(1)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                    }
                    .foregroundStyle(.white)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                    .disabled(isLoading)
                    .padding(.horizontal)
                    .padding(.bottom, 8)
                }
            }
        }
        .navigationTitle(settings.localized("scam_detector_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    generationTask?.cancel()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: $enableVision,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: true,
                writingMode: nil,
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels()
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else { selectedImageURL = nil; return }
            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self) {
                    selectedImageURL = sourceURL
                    return
                }
                if let data = try? await item.loadTransferable(type: Data.self) {
                    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("jpg")
                    try? data.write(to: temp)
                    selectedImageURL = temp
                }
            }
        }
        .onChange(of: enableVision) { _, isEnabled in
            if !isEnabled {
                selectedImageItem = nil
                selectedImageURL = nil
            }
        }
        .onDisappear {
            generationTask?.cancel()
            llm.unloadModel()
        }
    }

    private func buildAnalysisPrompt(content: String, hasImage: Bool) -> String {
        if hasImage && !content.isEmpty {
            return """
            You are a scam detection expert. Analyze BOTH the provided image AND the text content below for potential scams, fraud, phishing attempts, or suspicious activity.

            **Text content to analyze:**
            \(content)

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Cross-reference the text content with what's shown in the image
            - Look for inconsistencies between the image and text
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Red Flags in Image**: List any suspicious visual elements (fake logos, poor quality graphics, misleading layouts, etc.)
            3. **Red Flags in Text**: List any suspicious text elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
            4. **Consistency Check**: Do the image and text align? Are there contradictions?
            5. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
            6. **Verdict**: Is this likely a scam? Explain your reasoning based on BOTH the image and text.
            7. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
        }

        if hasImage {
            return """
            You are a scam detection expert. Analyze the provided image for potential scams, fraud, phishing attempts, or suspicious activity.

            **Instructions:**
            - Carefully examine the image for any suspicious elements, fake logos, misleading graphics, or scam indicators
            - Check if the image appears to be a screenshot of a phishing message, fake website, or fraudulent offer
            - Look for common scam tactics in the visual content

            Please provide a comprehensive analysis covering:
            1. **Risk Level**: Low, Medium, High, or Critical
            2. **Visual Red Flags**: List any suspicious elements in the image (fake logos, poor quality graphics, misleading layouts, urgency messages, too-good-to-be-true offers, etc.)
            3. **Legitimacy Indicators**: Any visual signs suggesting it might be legitimate
            4. **Verdict**: Is this likely a scam? Explain your reasoning based on the image.
            5. **Recommendations**: What should the user do?

            Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
            """
        }

        return """
        You are a scam detection expert. Analyze the following content for potential scams, fraud, phishing attempts, or suspicious activity.

        IMPORTANT: Respond in the same language as the input content. Match the language of the content in the image.

        Content to analyze:
        \(content)

        Please provide a comprehensive analysis covering:
        1. **Risk Level**: Low, Medium, High, or Critical
        2. **Red Flags**: List any suspicious elements (urgency tactics, too-good-to-be-true offers, suspicious links, impersonation, poor grammar, etc.)
        3. **Legitimacy Indicators**: Any signs suggesting it might be legitimate
        4. **Verdict**: Is this likely a scam? Explain your reasoning.
        5. **Recommendations**: What should the user do?

        Be thorough and specific in your analysis. If you detect a scam, clearly state it. If it appears legitimate, explain why.
        """
    }

    private func detectFirstURL(in text: String) -> String? {
        let pattern = #"https?://[^\s]+"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let urlRange = Range(match.range, in: text) else { return nil }
        return String(text[urlRange])
    }

    private func extractTextFromHTML(_ html: String) -> String {
        var cleaned = html.replacingOccurrences(of: #"<script[^>]*>.*?</script>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<style[^>]*>.*?</style>"#, with: "", options: [.regularExpression, .caseInsensitive])
        cleaned = cleaned.replacingOccurrences(of: #"<[^>]*>"#, with: " ", options: .regularExpression)
        cleaned = cleaned
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return String(cleaned.prefix(3000))
    }

    private func fetchURLContent(_ urlString: String) async -> String {
        guard let url = URL(string: urlString) else { return "" }
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Mozilla/5.0", forHTTPHeaderField: "User-Agent")
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let html = String(data: data, encoding: .utf8) ?? ""
            return extractTextFromHTML(html)
        } catch {
            return ""
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("scam_detector_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = enableVision
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if force || llm.currentlyLoadedModel != model.name {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggleAnalyze() {
        dismissKeyboard()

        if isAnalyzing {
            generationTask?.cancel()
            generationTask = nil
            isAnalyzing = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            isAnalyzing = true
            outputText = ""

            var contentToAnalyze = inputText.trimmingCharacters(in: .whitespacesAndNewlines)

            if let url = detectFirstURL(in: contentToAnalyze) {
                isFetchingURL = true
                let fetchedContent = await fetchURLContent(url)
                isFetchingURL = false
                if !fetchedContent.isEmpty {
                    let additionalContext = contentToAnalyze.replacingOccurrences(of: url, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                    contentToAnalyze = """
                    URL: \(url)

                    Content from URL:
                    \(fetchedContent)

                    \(!additionalContext.isEmpty ? "Additional context: \(additionalContext)" : "")
                    """
                }
            }

            if contentToAnalyze.isEmpty && selectedImageURL == nil {
                errorMessage = settings.localized("scam_detector_input_hint")
                isAnalyzing = false
                generationTask = nil
                return
            }

            do {
                let hasImage = selectedImageURL != nil && enableVision
                let prompt = buildAnalysisPrompt(content: contentToAnalyze, hasImage: hasImage)
                let effectiveImageURL = enableVision ? selectedImageURL : nil
                try await llm.generate(prompt: prompt, imageURL: effectiveImageURL) { text, _, _ in
                    Task { @MainActor in
                        outputText = sanitizeModelOutputText(text)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }

            isAnalyzing = false
            isFetchingURL = false
            generationTask = nil
        }
    }
}

private struct VibeChatMessage: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    let role: String
    var text: String
}

private struct VibeChatSession: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var messages: [VibeChatMessage] = []
}

private struct VibeChatSessionChip: View {
    let title: String
    let isSelected: Bool
    let canDelete: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onSelect) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }

            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .frame(width: 20, height: 20)
            }
            .disabled(!canDelete)
        }
        .foregroundStyle(.white)
        .background(
            Capsule(style: .continuous)
                .fill(isSelected ? Color.white.opacity(0.16) : Color.white.opacity(0.10))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.white.opacity(isSelected ? 0.22 : 0.14), lineWidth: 1)
        )
    }
}

private struct VibeFileChip: View {
    let title: String
    let isSelected: Bool
    let canDelete: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: onSelect) {
                Text(title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
            }

            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .bold))
                    .frame(width: 20, height: 20)
            }
            .disabled(!canDelete)
        }
        .foregroundStyle(.white)
        .background(
            Capsule(style: .continuous)
                .fill(isSelected ? Color.white.opacity(0.16) : Color.white.opacity(0.10))
        )
        .overlay(
            Capsule(style: .continuous)
                .stroke(Color.white.opacity(isSelected ? 0.22 : 0.14), lineWidth: 1)
        )
    }
}

private struct WorkspaceFilesSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    let folderURL: URL?
    let onPickFolder: () -> Void
    let onOpenFile: (URL) -> Void

    @State private var files: [URL] = []

    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()

                List {
                    Section {
                        Button {
                            dismiss()
                            onPickFolder()
                        } label: {
                            Label("Select different folder", systemImage: "folder.badge.plus")
                        }
                    }

                    Section("Files") {
                        if files.isEmpty {
                            Text("No files found in this folder.")
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(files, id: \.path) { url in
                                Button {
                                    onOpenFile(url)
                                } label: {
                                    HStack {
                                        Image(systemName: "doc.text")
                                        Text(url.lastPathComponent)
                                            .lineLimit(1)
                                        Spacer()
                                    }
                                }
                            }
                        }
                    }
                }
                .scrollContentBackground(.hidden)
            }
            .navigationTitle("Files")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .task {
                refreshFiles()
            }
        }
    }

    private func refreshFiles() {
        guard let folderURL else {
            files = []
            return
        }

        #if canImport(UIKit)
        let didStart = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStart {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }
        #endif

        let allowedExtensions: Set<String> = [
            "txt", "md",
            "py", "js", "ts", "java", "kt", "go", "rs", "cpp", "cc", "cxx", "c", "cs", "swift",
            "html", "htm", "css", "php", "rb", "lua", "sh", "bash", "zsh", "sql",
            "json", "yaml", "yml"
        ]

        let enumerator = FileManager.default.enumerator(at: folderURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles, .skipsPackageDescendants])
        var collected: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            let ext = url.pathExtension.lowercased()
            if allowedExtensions.contains(ext) {
                collected.append(url)
            }
        }
        files = collected.sorted(by: { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() })
    }
}

struct VibeCoderScreen: View {
    @EnvironmentObject var settings: AppSettings
    @AppStorage("feature_vibecoder_model_name") private var selectedModelName: String = ""
    @AppStorage("feature_vibecoder_max_tokens") private var maxTokens: Double = 2048
    @AppStorage("feature_vibecoder_enable_thinking") private var enableThinking: Bool = true
    @AppStorage("feature_vibecoder_folder_bookmark") private var folderBookmark: Data = Data()
    @AppStorage("feature_vibecoder_chat_sessions_data") private var chatSessionsData: Data = Data()
    @AppStorage("feature_vibecoder_active_chat_session_id") private var activeChatSessionIdRaw: String = ""
    @AppStorage("feature_vibecoder_current_file_relative_path") private var currentFileRelativePath: String = ""
    @State private var generatedCode: String = ""
    @State private var chatInput: String = ""
    @State private var chatSessions: [VibeChatSession] = [VibeChatSession(title: "Chat 1")]
    @State private var activeChatSessionId: UUID = UUID()
    @State private var workspaceFolderURL: URL?
    @State private var currentFileURL: URL?
    @State private var currentFileName: String?
    @State private var showWorkspaceFolderPicker = false
    @State private var showCreateFileDialog = false
    @State private var newFileNameInput = ""
    @State private var isLoading = false
    @State private var isGenerating = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?
    @State private var streamTick: Int = 0
    @State private var lastStreamTickTime: Double = 0
    @State private var debouncedAutosaveTask: Task<Void, Never>?
    @State private var workspaceFiles: [URL] = []
    @State private var pendingDeleteChatId: UUID?
    @State private var pendingDeleteFileURL: URL?

    private enum VibeFocusField: Hashable {
        case chat
        case editor
    }

    @FocusState private var focusedField: VibeFocusField?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var hasFileSession: Bool {
        !(currentFileName ?? "").isEmpty
    }

    private var hasWorkspaceFolder: Bool {
        workspaceFolderURL != nil
    }

    private var sendButtonIconName: String {
        isGenerating ? "xmark" : "paperplane.fill"
    }

    private var isSendButtonDisabled: Bool {
        (!isGenerating && chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) || !hasFileSession
    }

    private var activeMessages: [VibeChatMessage] {
        guard let idx = chatSessions.firstIndex(where: { $0.id == activeChatSessionId }) else {
            return chatSessions.first?.messages ?? []
        }
        return chatSessions[idx].messages
    }

    private var contextWindowCap: Double {
        let selectedCap = Double(max(1, selectedFeatureModel(named: selectedModelName)?.contextWindowSize ?? 0))
        let loadedCap = Double(max(1, llm.loadedContextWindow ?? 0))
        let configuredCap = Double(max(1, llm.contextWindow))
        return max(selectedCap, loadedCap, configuredCap, 1)
    }

    private var contextBudgetForRing: Double {
        // Keep the ring useful for normal chats even on very large-context models.
        let generationBudget = Double(max(1, Int(maxTokens)))
        return min(contextWindowCap, generationBudget)
    }

    private var approximateContextTokensUsed: Double {
        let activeTextChars = activeMessages.reduce(0) { $0 + $1.text.count }
        let codeChars = generatedCode.count
        let inputChars = chatInput.count
        let totalChars = activeTextChars + codeChars + inputChars
        let estimatedTokens = Double(totalChars) / 4.0
        return max(0, estimatedTokens)
    }

    private var contextUsageFractionRaw: Double {
        guard contextBudgetForRing > 0 else { return 0 }
        return min(max(approximateContextTokensUsed / contextBudgetForRing, 0), 1)
    }

    private var contextUsageFractionDisplay: Double {
        if approximateContextTokensUsed <= 0 {
            return 0
        }
        return min(max(contextUsageFractionRaw, 0.02), 1)
    }

    private var contextUsageLabel: String {
        if approximateContextTokensUsed > 0 {
            return "\(max(1, Int((contextUsageFractionRaw * 100).rounded())))%"
        }
        return "0%"
    }

    private var isContextBudgetExceededForSession: Bool {
        contextUsageFractionRaw >= 0.995
    }

    var body: some View {
        Group {
            if !isCurrentModelLoaded {
                VStack(spacing: 12) {
                    Image(systemName: "curlybraces.square")
                        .font(.system(size: 48, weight: .semibold))
                        .foregroundStyle(.secondary)
                    Text(settings.localized("scam_detector_load_model"))
                        .font(.title3.weight(.bold))
                    Text(settings.localized("scam_detector_load_model_desc"))
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button {
                        showSettings = true
                    } label: {
                        HStack {
                            Spacer()
                            Text(settings.localized("feature_settings_title"))
                            Spacer()
                        }
                        .frame(height: 50)
                        .contentShape(Rectangle())
                    }
                    .frame(maxWidth: 260)
                    .liquidGlassPrimaryButton(cornerRadius: 12)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                if !hasWorkspaceFolder {
                    VStack(spacing: 12) {
                        Image(systemName: "folder.fill")
                            .font(.system(size: 48, weight: .semibold))
                            .foregroundStyle(.secondary)

                        Text(settings.localized("vibe_coder_select_folder_title"))
                            .font(.title3.weight(.bold))
                            .multilineTextAlignment(.center)

                        Text(settings.localized("vibe_coder_select_folder_desc"))
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.7))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)

                        Button {
                            showWorkspaceFolderPicker = true
                        } label: {
                            HStack {
                                Spacer()
                                Text(settings.localized("vibe_coder_open_folder"))
                                Spacer()
                            }
                            .frame(height: 50)
                            .contentShape(Rectangle())
                        }
                        .frame(maxWidth: 260)
                        .liquidGlassPrimaryButton(cornerRadius: 12)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 12) {
                        GeometryReader { _ in
                            let isLandscape = UIScreen.main.bounds.width > UIScreen.main.bounds.height

                            let chatPanel = AnyView(
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack {
                                        Text(settings.localized("vibe_coder_ai_chat"))
                                            .font(.headline)
                                        Spacer()

                                        ZStack {
                                            Circle()
                                                .stroke(Color.white.opacity(0.18), lineWidth: 2)
                                            Circle()
                                                .trim(from: 0, to: contextUsageFractionDisplay)
                                                .stroke(
                                                    contextUsageFractionRaw < 0.90 ? Color.cyan : Color.orange,
                                                    style: StrokeStyle(lineWidth: 2.5, lineCap: .round)
                                                )
                                                .rotationEffect(.degrees(-90))

                                            Text(contextUsageFractionRaw < 0.995 ? contextUsageLabel : "!")
                                                .font(.system(size: 8, weight: .bold, design: .rounded))
                                        }
                                        .frame(width: 28, height: 28)
                                        .accessibilityLabel("Context usage \(contextUsageLabel)")

                                        Button {
                                            clearActiveChat()
                                        } label: {
                                            Image(systemName: "trash")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(activeMessages.isEmpty || isGenerating)

                                        Button {
                                            createNewChatSession()
                                        } label: {
                                            Image(systemName: "plus")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(isGenerating)
                                    }
                                    .padding(.horizontal)

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 8) {
                                            ForEach(chatSessions) { session in
                                                VibeChatSessionChip(
                                                    title: session.title,
                                                    isSelected: session.id == activeChatSessionId,
                                                    canDelete: chatSessions.count > 1 && !isGenerating,
                                                    onSelect: { activeChatSessionId = session.id },
                                                    onDelete: { pendingDeleteChatId = session.id }
                                                )
                                            }
                                        }
                                        .padding(.horizontal)
                                    }

                                    ScrollViewReader { proxy in
                                        ScrollView {
                                            VStack(alignment: .leading, spacing: 8) {
                                                ForEach(activeMessages) { message in
                                                    VStack(alignment: .leading, spacing: 4) {
                                                        Text(message.role == "user" ? settings.localized("vibe_coder_message_you") : settings.localized("vibe_coder_message_ai"))
                                                            .font(.caption.weight(.semibold))
                                                            .foregroundStyle(.white.opacity(0.65))
                                                        Text(message.text)
                                                            .textSelection(.enabled)
                                                            .frame(maxWidth: .infinity, alignment: .leading)
                                                            .padding(8)
                                                            .background(.ultraThinMaterial)
                                                            .clipShape(RoundedRectangle(cornerRadius: 10))
                                                    }
                                                    .id(message.id)
                                                }
                                            }
                                            .padding(.horizontal)
                                            .padding(.top, 6)
                                        }
                                        .frame(minHeight: 160)
                                        .onChange(of: activeMessages.count) { _, _ in
                                            if let last = activeMessages.last {
                                                withAnimation(.linear(duration: 0.08)) {
                                                    proxy.scrollTo(last.id, anchor: .bottom)
                                                }
                                            }
                                        }
                                        .onChange(of: streamTick) { _, _ in
                                            if let last = activeMessages.last {
                                                proxy.scrollTo(last.id, anchor: .bottom)
                                            }
                                        }
                                    }

                                    HStack(spacing: 10) {
                                        TextField(
                                            hasFileSession ? settings.localized("vibe_coder_ask_ai_edit") : settings.localized("vibe_coder_create_open_file_hint"),
                                            text: $chatInput,
                                            axis: .vertical
                                        )
                                        .lineLimit(1...5)
                                        .focused($focusedField, equals: .chat)
                                        .disabled(!hasFileSession || isGenerating)
                                        .padding(.vertical, 10)
                                        .padding(.horizontal, 12)
                                        .background(Color.white.opacity(0.05))
                                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                                        Button {
                                            if isGenerating {
                                                stopGeneration()
                                            } else {
                                                sendChat()
                                            }
                                        } label: {
                                            Image(systemName: sendButtonIconName)
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 44, height: 44)
                                        }
                                        .foregroundStyle(.white)
                                        .background(
                                            RoundedRectangle(cornerRadius: 12)
                                                .fill(Color.white.opacity(0.08))
                                        )
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                        )
                                        .disabled(isSendButtonDisabled)
                                    }
                                    .padding(.horizontal)
                                    .padding(.bottom, 8)
                                }
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(Color.white.opacity(0.14), lineWidth: 1)
                                )
                                .padding(.horizontal)
                            )

                            let editorPanel = AnyView(
                                VStack(alignment: .leading, spacing: 8) {
                                    HStack {
                                        Text(currentFileName ?? settings.localized("vibe_coder_open_or_create_file"))
                                            .font(.subheadline.weight(.semibold))
                                            .foregroundStyle(.white.opacity(0.72))
                                        Spacer()

                                        Button {
                                            showWorkspaceFolderPicker = true
                                        } label: {
                                            Image(systemName: "folder")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }

                                        Button {
                                            showCreateFileDialog = true
                                        } label: {
                                            Image(systemName: "plus")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }

                                        Button {
                                            #if canImport(UIKit)
                                            UIPasteboard.general.string = generatedCode
                                            #endif
                                        } label: {
                                            Image(systemName: "doc.on.doc")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                                        Button {
                                            saveCurrentFile()
                                        } label: {
                                            Image(systemName: "square.and.arrow.down")
                                                .font(.system(size: 16, weight: .semibold))
                                                .frame(width: 36, height: 36)
                                        }
                                        .disabled(!hasFileSession)

                                        if isHTMLFile {
                                            Button {
                                                openHTMLPreviewInSafari()
                                            } label: {
                                                Image(systemName: "safari")
                                                    .font(.system(size: 16, weight: .semibold))
                                                    .frame(width: 36, height: 36)
                                            }
                                            .disabled(generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                                        }
                                    }

                                    ScrollView(.horizontal, showsIndicators: false) {
                                        HStack(spacing: 8) {
                                            ForEach(workspaceFiles, id: \.path) { url in
                                                VibeFileChip(
                                                    title: url.lastPathComponent,
                                                    isSelected: url == currentFileURL,
                                                    canDelete: !isGenerating,
                                                    onSelect: { openFile(url) },
                                                    onDelete: { pendingDeleteFileURL = url }
                                                )
                                            }
                                        }
                                    }

                                    TextEditor(text: $generatedCode)
                                        .font(.system(.body, design: .monospaced))
                                        .focused($focusedField, equals: .editor)
                                        .frame(minHeight: 220)
                                        .padding(8)
                                        .scrollContentBackground(.hidden)
                                        .background(Color.white.opacity(0.02))
                                        .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                .padding(.horizontal)
                                .padding(.vertical, 10)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(Color.white.opacity(0.14), lineWidth: 1)
                                )
                                .padding(.horizontal)
                            )

                            Group {
                                if isLandscape {
                                    HStack(spacing: 12) {
                                        editorPanel
                                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                                        chatPanel
                                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                                    }
                                } else {
                                    VStack(spacing: 12) {
                                        chatPanel
                                        editorPanel
                                    }
                                }
                            }
                        }

                        if let errorMessage {
                            Text(errorMessage)
                                .foregroundColor(.red)
                                .font(.caption)
                                .padding(.horizontal)
                        }
                    }
                }
            }
        }
        .navigationTitle(settings.localized("vibe_coder_title"))
        .navigationBarTitleDisplayMode(.inline)
        .apolloScreenBackground()
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button(settings.localized("done")) {
                    focusedField = nil
                    dismissKeyboard()
                }
            }
        }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    stopGeneration()
                    llm.unloadModel()
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
            }
        }
        .sheet(isPresented: $showSettings) {
            FeatureModelSettingsSheet(
                selectedModelName: $selectedModelName,
                maxTokens: $maxTokens,
                enableThinking: $enableThinking,
                enableVision: .constant(false),
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                supportsVisionToggle: false,
                writingMode: nil,
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            restoreChatSessionsFromStorage()

            Task {
                await syncRunAnywhereModelDiscovery()
                let available = downloadableFeatureModels()
                if selectedModelName.isEmpty || !available.contains(where: { $0.name == selectedModelName }) {
                    selectedModelName = available.first?.name ?? ""
                }
            }

            if workspaceFolderURL == nil {
                restoreWorkspaceFolderFromBookmark()
            } else {
                refreshWorkspaceFiles()
            }

            if workspaceFolderURL != nil {
                restoreLastOpenedFileIfPossible()
            }

            if !chatSessions.contains(where: { $0.id == activeChatSessionId }) {
                activeChatSessionId = chatSessions.first?.id ?? activeChatSessionId
            }
        }
        .onChange(of: chatSessions) { _, _ in
            persistChatSessionsToStorage()
        }
        .onChange(of: activeChatSessionId) { _, _ in
            persistChatSessionsToStorage()
        }
        .onChange(of: workspaceFolderURL) { _, newValue in
            if newValue != nil {
                restoreLastOpenedFileIfPossible()
            }
        }
        .onChange(of: currentFileURL) { _, _ in
            persistCurrentFileSelection()
        }
        .onChange(of: generatedCode) { _, _ in
            guard hasFileSession, !isGenerating else { return }
            debouncedAutosaveTask?.cancel()
            debouncedAutosaveTask = Task {
                try? await Task.sleep(nanoseconds: 450_000_000)
                if Task.isCancelled { return }
                await MainActor.run {
                    saveCurrentFile(silent: true)
                }
            }
        }
        .fileImporter(isPresented: $showWorkspaceFolderPicker, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url):
                setWorkspaceFolder(url)
            case .failure(let error):
                errorMessage = error.localizedDescription
            }
        }
        .alert(settings.localized("vibe_coder_create_file_title"), isPresented: $showCreateFileDialog) {
            TextField(settings.localized("vibe_coder_file_name_placeholder"), text: $newFileNameInput)
            Button(settings.localized("cancel"), role: .cancel) {}
            Button(settings.localized("vibe_coder_create")) {
                createNewFile()
            }
        }
        .alert(settings.localized("vibe_coder_delete_chat_title"), isPresented: Binding(
            get: { pendingDeleteChatId != nil },
            set: { if !$0 { pendingDeleteChatId = nil } }
        )) {
            Button(settings.localized("cancel"), role: .cancel) { pendingDeleteChatId = nil }
            Button(settings.localized("delete"), role: .destructive) {
                if let id = pendingDeleteChatId {
                    deleteChatSession(id)
                }
            }
        } message: {
            Text(settings.localized("vibe_coder_delete_chat_message"))
        }
        .alert(settings.localized("vibe_coder_delete_file_title"), isPresented: Binding(
            get: { pendingDeleteFileURL != nil },
            set: { if !$0 { pendingDeleteFileURL = nil } }
        )) {
            Button(settings.localized("cancel"), role: .cancel) { pendingDeleteFileURL = nil }
            Button(settings.localized("delete"), role: .destructive) {
                if let fileURL = pendingDeleteFileURL {
                    deleteFile(fileURL)
                }
            }
        } message: {
            Text(String(format: settings.localized("vibe_coder_delete_file_message"), pendingDeleteFileURL?.lastPathComponent ?? ""))
        }
        .onDisappear {
            stopGeneration()
            debouncedAutosaveTask?.cancel()
            debouncedAutosaveTask = nil
            llm.unloadModel()
            #if canImport(Network)
            FeatureLocalHTMLPreviewServer.shared.stop()
            #endif
        }
    }

    private func normalizedExtension(_ fileName: String?) -> String? {
        guard let raw = fileName?.lowercased() else { return nil }
        let parts = raw.split(separator: ".")
        if parts.count < 2 { return nil }
        if parts.last == "txt", parts.count >= 3 {
            return String(parts[parts.count - 2])
        }
        return String(parts.last!)
    }

    private var isHTMLFile: Bool {
        let ext = normalizedExtension(currentFileName)
        return ext == "html" || ext == "htm"
    }

    private func languagePromptConfig() -> (languageName: String, targetRule: String, fenceLanguage: String)? {
        switch normalizedExtension(currentFileName) {
        case "html", "htm", "css":
            return ("Web App (HTML/CSS/JS)", "Build a single self-contained HTML file with embedded CSS and JavaScript.", "html")
        case "py": return ("Python", "Build a runnable Python script using only standard library.", "python")
        case "js": return ("JavaScript", "Build a runnable JavaScript program (no TypeScript).", "javascript")
        case "ts": return ("TypeScript", "Build a runnable TypeScript program with clear types.", "typescript")
        case "c": return ("C", "Build a runnable C program with int main().", "c")
        case "php": return ("PHP", "Build a runnable PHP script.", "php")
        case "rb": return ("Ruby", "Build a runnable Ruby script.", "ruby")
        case "swift": return ("Swift", "Build a runnable Swift program.", "swift")
        case "dart": return ("Dart", "Build a runnable Dart program.", "dart")
        case "lua": return ("Lua", "Build a runnable Lua script.", "lua")
        case "sh", "bash", "zsh": return ("Shell", "Build a runnable POSIX shell script.", "sh")
        case "sql": return ("SQL", "Build valid SQL statements with clear schema assumptions.", "sql")
        case "java": return ("Java", "Build a runnable Java program with a main method.", "java")
        case "kt": return ("Kotlin", "Build a runnable Kotlin console program with a main function.", "kotlin")
        case "cs": return ("C#", "Build a runnable C# console app entry point.", "csharp")
        case "cpp", "cc", "cxx": return ("C++", "Build a runnable modern C++ program (C++17 style).", "cpp")
        case "go": return ("Go", "Build a runnable Go program with package main and func main().", "go")
        case "rs": return ("Rust", "Build a runnable Rust program with fn main().", "rust")
        default: return nil
        }
    }

    private func buildFileAwareEditPrompt(_ userPrompt: String) -> String {
        guard let config = languagePromptConfig() else { return userPrompt }
        let fileName = currentFileName ?? "untitled"
        let codeSection = generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "FILE_IS_EMPTY"
            : String(generatedCode.prefix(20_000))

        return """
        You are an expert coding assistant working on a real file.
        FILE: \(fileName)
        TARGET LANGUAGE: \(config.languageName)
        TARGET RULE: \(config.targetRule)

        USER REQUEST:
        \(userPrompt)

        CURRENT FILE CONTENT:
        ```\(config.fenceLanguage)
        \(codeSection)
        ```

        INSTRUCTIONS:
        - Produce the FULL updated file content.
        - Do not return partial snippets or patch hunks.
        - Wrap the final full file between markers:
          <<<FULL_FILE_START>>>
          [full file content]
          <<<FULL_FILE_END>>>
        - Respect the FILE extension/language exactly.
        - If file is empty, create a complete starter implementation for this request.
        - Do not output explanations.
        - Output only one fenced code block using ```\(config.fenceLanguage).
        """
    }

    private func extractGeneratedCode(_ response: String) -> String {
        if let markerRange = response.range(of: #"<<<FULL_FILE_START>>>[\s\S]*?<<<FULL_FILE_END>>>"#, options: .regularExpression) {
            var extracted = String(response[markerRange])
            extracted = extracted.replacingOccurrences(of: "<<<FULL_FILE_START>>>", with: "")
            extracted = extracted.replacingOccurrences(of: "<<<FULL_FILE_END>>>", with: "")
            return sanitizeExtractedCode(extracted)
        }

        if let codeRange = response.range(of: #"```[a-zA-Z0-9_-]*[\s\S]*?```"#, options: .regularExpression) {
            let fenced = String(response[codeRange])
                .replacingOccurrences(of: #"^```[a-zA-Z0-9_-]*\s*"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\s*```$"#, with: "", options: .regularExpression)
            return sanitizeExtractedCode(fenced)
        }

        return sanitizeExtractedCode(response)
    }

    private func sanitizeExtractedCode(_ raw: String) -> String {
        raw
            .replacingOccurrences(of: "<<<FULL_FILE_START>>>", with: "")
            .replacingOccurrences(of: "<<<FULL_FILE_END>>>", with: "")
            .replacingOccurrences(of: "```", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func openFile(_ url: URL, announceInChat: Bool = true) {
        do {
            #if canImport(UIKit)
            let scopeURL = workspaceFolderURL ?? url
            let didStart = scopeURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    scopeURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let text = try String(contentsOf: url, encoding: .utf8)
            currentFileURL = url
            currentFileName = url.lastPathComponent
            generatedCode = text
            if announceInChat {
                appendMessage(to: activeChatSessionId, role: "assistant", text: "Opened \(url.lastPathComponent)")
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func createNewFile() {
        let name = newFileNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard name.contains("."), !name.isEmpty else {
            errorMessage = settings.localized("vibe_coder_file_name_error")
            return
        }
        guard let folderURL = workspaceFolderURL else {
            errorMessage = settings.localized("vibe_coder_select_folder_title")
            return
        }

        let fileURL = folderURL.appendingPathComponent(name)
        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            if !FileManager.default.fileExists(atPath: fileURL.path) {
                try "".write(to: fileURL, atomically: true, encoding: .utf8)
            }

            currentFileURL = fileURL
            currentFileName = fileURL.lastPathComponent
            generatedCode = ""
            refreshWorkspaceFiles()
            appendMessage(to: activeChatSessionId, role: "assistant", text: "Started new file: \(name)")
        } catch {
            errorMessage = error.localizedDescription
        }

        newFileNameInput = ""
    }

    private func saveCurrentFile(silent: Bool = false) {
        guard hasFileSession else {
            errorMessage = settings.localized("vibe_coder_no_file_selected_error")
            return
        }

        guard let workspaceFolderURL else {
            errorMessage = settings.localized("vibe_coder_select_folder_title")
            return
        }

        let folderURL = workspaceFolderURL

        let fileURL: URL
        if let currentFileURL {
            fileURL = currentFileURL
        } else {
            fileURL = folderURL.appendingPathComponent(currentFileName ?? "main.txt")
            currentFileURL = fileURL
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            try generatedCode.write(to: fileURL, atomically: true, encoding: .utf8)
            currentFileURL = fileURL
            currentFileName = fileURL.lastPathComponent
            if !silent {
                refreshWorkspaceFiles()
            }
            if !silent {
                appendMessage(to: activeChatSessionId, role: "assistant", text: "Saved \(currentFileName ?? fileURL.lastPathComponent)")
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func restoreWorkspaceFolderFromBookmark() {
        guard !folderBookmark.isEmpty else { return }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: folderBookmark, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale) else {
            return
        }
        if stale {
            // Keep using it for now; user can re-pick later.
        }
        workspaceFolderURL = url
    }

    private func setWorkspaceFolder(_ url: URL) {
        workspaceFolderURL = url
        currentFileURL = nil
        currentFileName = nil
        generatedCode = ""
        do {
            #if canImport(UIKit)
            let didStart = url.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    url.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let bookmark = try url.bookmarkData(options: [.minimalBookmark], includingResourceValuesForKeys: nil, relativeTo: nil)
            folderBookmark = bookmark
        } catch {
            // Non-fatal; folder selection still works for this session.
        }

        refreshWorkspaceFiles()
        restoreLastOpenedFileIfPossible()
    }

    private func openHTMLPreviewInSafari() {
        let html = generatedCode
        guard !html.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        #if canImport(Network) && canImport(UIKit)
        Task { @MainActor in
            do {
                let url = try await FeatureLocalHTMLPreviewServer.shared.start(html: html)
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        #elseif canImport(UIKit)
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("LLMHub-Preview-\(UUID().uuidString)")
            .appendingPathExtension("html")
        do {
            try html.write(to: fileURL, atomically: true, encoding: .utf8)
            UIApplication.shared.open(fileURL, options: [:], completionHandler: nil)
        } catch {
            errorMessage = error.localizedDescription
        }
        #endif
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("vibe_coder_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        let modelContextCap = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
        let effectiveContext = min(max(1, Int(maxTokens)), modelContextCap)
        llm.maxTokens = min(Int(maxTokens), effectiveContext)
        llm.contextWindow = effectiveContext
        llm.enableVision = false
        llm.enableAudio = false
        llm.enableThinking = enableThinking

        do {
            if force || llm.currentlyLoadedModel != model.name {
                try await llm.loadModel(model)
            }
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    private func sendChat() {
        dismissKeyboard()

        let trimmedPrompt = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return }
        guard !isGenerating else { return }

        if isContextBudgetExceededForSession {
            createNewChatSession()
            appendMessage(
                to: activeChatSessionId,
                role: "assistant",
                text: "Started a new chat because context window was full."
            )
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            guard hasFileSession else {
                errorMessage = settings.localized("vibe_coder_no_file_selected_error")
                return
            }
            guard languagePromptConfig() != nil else {
                errorMessage = "Unsupported or unknown file extension. Use a code file like .py, .js, .ts, .java, .kt, .go, .rs, .cpp, .cs, .html"
                return
            }

            let sessionId = activeChatSessionId
            appendMessage(to: sessionId, role: "user", text: trimmedPrompt)
            let assistantId = appendMessage(to: sessionId, role: "assistant", text: "")
            await MainActor.run { chatInput = "" }

            isGenerating = true
            do {
                let prompt = buildFileAwareEditPrompt(trimmedPrompt)
                try await llm.generate(prompt: prompt) { text, _, _ in
                    Task { @MainActor in
                        updateMessageText(sessionId: sessionId, messageId: assistantId, text: text)

                        let now = CFAbsoluteTimeGetCurrent()
                        if now - lastStreamTickTime > 0.06 {
                            lastStreamTickTime = now
                            streamTick += 1
                        }
                    }
                }

                if let finalText = messageText(sessionId: sessionId, messageId: assistantId) {
                    let extractedCode = extractGeneratedCode(finalText)
                    if !extractedCode.isEmpty {
                        await MainActor.run {
                            generatedCode = extractedCode
                            saveCurrentFile(silent: true)
                        }
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
            generationTask = nil
        }
    }

    @discardableResult
    private func appendMessage(to sessionId: UUID, role: String, text: String, id: UUID = UUID()) -> UUID {
        guard let idx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return id }
        chatSessions[idx].messages.append(VibeChatMessage(id: id, role: role, text: text))
        return id
    }

    private func updateMessageText(sessionId: UUID, messageId: UUID, text: String) {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return }
        guard let mIdx = chatSessions[sIdx].messages.firstIndex(where: { $0.id == messageId }) else { return }
        chatSessions[sIdx].messages[mIdx].text = sanitizeModelOutputText(text)
    }

    private func messageText(sessionId: UUID, messageId: UUID) -> String? {
        guard let sIdx = chatSessions.firstIndex(where: { $0.id == sessionId }) else { return nil }
        guard let msg = chatSessions[sIdx].messages.first(where: { $0.id == messageId }) else { return nil }
        return msg.text
    }

    private func createNewChatSession() {
        let title = "Chat \(chatSessions.count + 1)"
        let session = VibeChatSession(title: title)
        chatSessions.append(session)
        activeChatSessionId = session.id
    }

    private func clearActiveChat() {
        guard let idx = chatSessions.firstIndex(where: { $0.id == activeChatSessionId }) else { return }
        chatSessions[idx].messages.removeAll()
    }

    private func deleteChatSession(_ id: UUID) {
        guard chatSessions.count > 1 else {
            pendingDeleteChatId = nil
            return
        }
        chatSessions.removeAll { $0.id == id }
        if activeChatSessionId == id {
            activeChatSessionId = chatSessions.first?.id ?? activeChatSessionId
        }
        pendingDeleteChatId = nil
    }

    private func refreshWorkspaceFiles() {
        guard let folderURL = workspaceFolderURL else {
            workspaceFiles = []
            return
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            let urls = try FileManager.default.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )

            workspaceFiles = urls
                .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false }
                .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
        } catch {
            workspaceFiles = []
        }
    }

    private func deleteFile(_ url: URL) {
        guard let folderURL = workspaceFolderURL else {
            pendingDeleteFileURL = nil
            return
        }

        do {
            #if canImport(UIKit)
            let didStart = folderURL.startAccessingSecurityScopedResource()
            defer {
                if didStart {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }
            #endif

            try FileManager.default.removeItem(at: url)
            if currentFileURL == url {
                currentFileURL = nil
                currentFileName = nil
                generatedCode = ""
            }
            refreshWorkspaceFiles()
        } catch {
            errorMessage = error.localizedDescription
        }

        pendingDeleteFileURL = nil
    }

    private func persistChatSessionsToStorage() {
        do {
            chatSessionsData = try JSONEncoder().encode(chatSessions)
        } catch {
            // Keep previous valid encoded value if encoding fails.
        }
        activeChatSessionIdRaw = activeChatSessionId.uuidString
    }

    private func restoreChatSessionsFromStorage() {
        if !chatSessionsData.isEmpty,
           let decoded = try? JSONDecoder().decode([VibeChatSession].self, from: chatSessionsData),
           !decoded.isEmpty {
            chatSessions = decoded
        } else if chatSessions.isEmpty {
            chatSessions = [VibeChatSession(title: "Chat 1")]
        }

        if let restoredId = UUID(uuidString: activeChatSessionIdRaw),
           chatSessions.contains(where: { $0.id == restoredId }) {
            activeChatSessionId = restoredId
        } else {
            activeChatSessionId = chatSessions.first?.id ?? UUID()
        }
    }

    private func persistCurrentFileSelection() {
        guard let folderURL = workspaceFolderURL,
              let fileURL = currentFileURL,
              fileURL.path.hasPrefix(folderURL.path) else {
            currentFileRelativePath = ""
            return
        }

        var relativePath = fileURL.path
        relativePath.removeFirst(folderURL.path.count)
        currentFileRelativePath = relativePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private func restoreLastOpenedFileIfPossible() {
        guard let folderURL = workspaceFolderURL,
              !currentFileRelativePath.isEmpty,
              currentFileURL == nil else {
            return
        }

        let candidateURL = folderURL.appendingPathComponent(currentFileRelativePath)
        if FileManager.default.fileExists(atPath: candidateURL.path) {
            openFile(candidateURL, announceInChat: false)
        }
    }
}
