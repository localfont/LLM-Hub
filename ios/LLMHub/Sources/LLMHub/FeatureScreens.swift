import Foundation
import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import RunAnywhere

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

private struct FeatureModelSettingsSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Binding var selectedModelName: String
    @Binding var maxTokens: Double
    @Binding var enableThinking: Bool
    @Binding var enableVision: Bool
    @Binding var isLoading: Bool
    @Binding var errorMessage: String?
    let supportsVisionToggle: Bool
    let onLoad: () async -> Void
    let onUnload: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var models: [AIModel] = []
    @State private var isRefreshingModels = false

    private var selectedModel: AIModel? {
        models.first(where: { $0.name == selectedModelName })
            ?? selectedFeatureModel(named: selectedModelName)
    }

    private var maxContextCap: Double {
        let advertised = selectedModel?.contextWindowSize ?? 4096
        return Double(max(1, advertised))
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("select_model"))
                            .font(.headline)

                        if isRefreshingModels {
                            HStack(spacing: 8) {
                                ProgressView()
                                Text(settings.localized("loading"))
                                    .foregroundStyle(.secondary)
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
                            Spacer()
                            Text("\(Int(maxTokens))")
                                .monospacedDigit()
                        }
                        Slider(value: $maxTokens, in: 1...maxContextCap, step: 1)
                            .onChange(of: maxTokens) { _, newValue in
                                maxTokens = min(max(1, newValue), maxContextCap)
                            }

                        Toggle(settings.localized("enable_thinking"), isOn: $enableThinking)
                        if supportsVisionToggle {
                            Toggle(settings.localized("scam_detector_enable_vision"), isOn: $enableVision)
                        }
                    }
                    .padding()
                    .background(Color(uiColor: .secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 14))

                    HStack(spacing: 10) {
                        Button {
                            Task { await onLoad() }
                        } label: {
                            if isLoading {
                                ProgressView()
                            } else {
                                Text(settings.localized("load_model"))
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isLoading || selectedModelName.isEmpty || isRefreshingModels)

                        Button(settings.localized("unload_model"), role: .destructive) {
                            onUnload()
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoading)
                    }

                    if let errorMessage, !errorMessage.isEmpty {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                }
                .padding()
            }
            .navigationTitle(settings.localized("feature_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(settings.localized("done")) { dismiss() }
                }
            }
            .task {
                await refreshModelsIfNeeded()
            }
        }
    }

    private func refreshModelsIfNeeded() async {
        if !models.isEmpty { return }
        isRefreshingModels = true
        let loaded = downloadableFeatureModels()
        models = loaded
        if selectedModelName.isEmpty || !loaded.contains(where: { $0.name == selectedModelName }) {
            selectedModelName = loaded.first?.name ?? ""
        }
        let cap = Double(max(1, selectedModel?.contextWindowSize ?? 4096))
        maxTokens = min(max(1, maxTokens), cap)
        isRefreshingModels = false
    }
}

struct WritingAidScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var selectedModelName: String = ""
    @State private var maxTokens: Double = 1024
    @State private var enableThinking: Bool = true
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var selectedMode: WritingAidMode = .friendly
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
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button(settings.localized("feature_settings_title")) {
                        showSettings = true
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(settings.localized("writing_aid_input_label"))
                            .font(.headline)
                        TextEditor(text: $inputText)
                            .frame(minHeight: 140)
                            .padding(8)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Label(settings.localized("paste"), systemImage: "doc.on.clipboard")
                            }
                            .buttonStyle(.bordered)
                            Spacer()
                        }
                    }
                    .padding(.horizontal)

                    Picker("", selection: $selectedMode) {
                        ForEach(WritingAidMode.allCases, id: \.rawValue) { mode in
                            Text(settings.localized(mode.rawValue)).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)

                    HStack(spacing: 12) {
                        Button {
                            toggleProcess()
                        } label: {
                            Text(isProcessing ? settings.localized("processing_tap_to_cancel") : settings.localized("writing_aid_process"))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isLoading || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                        Button(settings.localized("copy")) {
                            #if canImport(UIKit)
                            UIPasteboard.general.string = outputText
                            #endif
                        }
                        .buttonStyle(.bordered)
                        .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(.horizontal)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("writing_aid_result"))
                                .font(.headline)
                            Text(outputText.isEmpty ? "-" : outputText)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Color(uiColor: .secondarySystemBackground))
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
                }
            }
        }
        .navigationTitle(settings.localized("writing_aid_title"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: onNavigateBack) { Image(systemName: "arrow.left") }
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
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            if selectedModelName.isEmpty {
                selectedModelName = downloadableFeatureModels().first?.name ?? ""
            }
        }
        .onDisappear {
            generationTask?.cancel()
        }
    }

    private func writingPrompt() -> String {
        let content = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch selectedMode {
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

        llm.maxTokens = Int(maxTokens)
        llm.contextWindow = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
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
                        outputText = text
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
    @State private var selectedModelName: String = ""
    @State private var maxTokens: Double = 1024
    @State private var enableThinking: Bool = true
    @State private var enableVision: Bool = true
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
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button(settings.localized("feature_settings_title")) {
                        showSettings = true
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("scam_detector_input_label"))
                            .font(.headline)
                        TextEditor(text: $inputText)
                            .frame(minHeight: 120)
                            .padding(8)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        HStack {
                            Button {
                                #if canImport(UIKit)
                                if let clip = UIPasteboard.general.string, !clip.isEmpty {
                                    inputText += clip
                                }
                                #endif
                            } label: {
                                Label(settings.localized("paste"), systemImage: "doc.on.clipboard")
                            }
                            .buttonStyle(.bordered)
                            Spacer()
                        }

                        if enableVision {
                            let visionLabel = settings.localized("scam_detector_enable_vision")
                            PhotosPicker(selection: $selectedImageItem, matching: .images) {
                                Label(visionLabel, systemImage: "photo")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)

                            if let selectedImageURL,
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
                    }
                    .padding(.horizontal)

                    if isFetchingURL {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text(settings.localized("scam_detector_fetching_url"))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                            Spacer()
                        }
                        .padding(.horizontal)
                    }

                    HStack(spacing: 12) {
                        Button {
                            toggleAnalyze()
                        } label: {
                            Text(isAnalyzing ? settings.localized("scam_detector_analyzing") : settings.localized("scam_detector_analyze"))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isLoading)

                        Button(settings.localized("copy")) {
                            #if canImport(UIKit)
                            UIPasteboard.general.string = outputText
                            #endif
                        }
                        .buttonStyle(.bordered)
                        .disabled(outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(.horizontal)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(settings.localized("scam_detector_result"))
                                .font(.headline)
                            Text(outputText.isEmpty ? "-" : outputText)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Color(uiColor: .secondarySystemBackground))
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
                }
            }
        }
        .navigationTitle(settings.localized("scam_detector_title"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: onNavigateBack) { Image(systemName: "arrow.left") }
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
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            if selectedModelName.isEmpty {
                selectedModelName = downloadableFeatureModels().first?.name ?? ""
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
        .onDisappear {
            generationTask?.cancel()
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

        llm.maxTokens = Int(maxTokens)
        llm.contextWindow = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
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
                try await llm.generate(prompt: prompt, imageURL: selectedImageURL) { text, _, _ in
                    Task { @MainActor in
                        outputText = text
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

private struct VibeChatMessage: Identifiable {
    let id = UUID()
    let role: String
    var text: String
}

struct VibeCoderScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var selectedModelName: String = ""
    @State private var maxTokens: Double = 2048
    @State private var enableThinking: Bool = true
    @State private var promptText: String = ""
    @State private var generatedCode: String = ""
    @State private var chatMessages: [VibeChatMessage] = []
    @State private var currentFileURL: URL?
    @State private var currentFileName: String?
    @State private var showOpenFilePicker = false
    @State private var showCreateFileDialog = false
    @State private var newFileNameInput = ""
    @State private var isLoading = false
    @State private var isGenerating = false
    @State private var showSettings = false
    @State private var errorMessage: String?
    @State private var generationTask: Task<Void, Never>?

    let onNavigateBack: () -> Void

    @ObservedObject private var llm = LLMBackend.shared

    private var isCurrentModelLoaded: Bool {
        llm.isLoaded && llm.currentlyLoadedModel == selectedModelName
    }

    private var hasFileSession: Bool {
        !(currentFileName ?? "").isEmpty
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
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                    Button(settings.localized("feature_settings_title")) {
                        showSettings = true
                    }
                    .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 10) {
                    HStack(spacing: 8) {
                        Button {
                            showOpenFilePicker = true
                        } label: {
                            Label(settings.localized("vibe_coder_open_folder"), systemImage: "folder")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            showCreateFileDialog = true
                        } label: {
                            Label(settings.localized("vibe_coder_new_file"), systemImage: "plus")
                        }
                        .buttonStyle(.bordered)

                        Button {
                            saveCurrentFile()
                        } label: {
                            Label(settings.localized("vibe_coder_save_file"), systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.bordered)
                        .disabled(!hasFileSession)
                    }
                    .padding(.horizontal)

                    VStack(alignment: .leading, spacing: 6) {
                        Text(currentFileName ?? settings.localized("vibe_coder_open_or_create_file"))
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)
                        TextEditor(text: $generatedCode)
                            .font(.system(.body, design: .monospaced))
                            .frame(minHeight: 220)
                            .padding(8)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .padding(.horizontal)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(chatMessages) { message in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(message.role == "user" ? settings.localized("vibe_coder_message_you") : settings.localized("vibe_coder_message_ai"))
                                        .font(.caption.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                    Text(message.text)
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(8)
                                        .background(Color(uiColor: .secondarySystemBackground))
                                        .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            }
                        }
                        .padding(.horizontal)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("vibe_coder_prompt_label"))
                            .font(.headline)
                        TextEditor(text: $promptText)
                            .frame(minHeight: 90)
                            .padding(8)
                            .background(Color(uiColor: .secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .padding(.horizontal)

                    HStack(spacing: 12) {
                        Button {
                            toggleGenerate()
                        } label: {
                            Text(isGenerating ? settings.localized("vibe_coder_stop_generation") : settings.localized("vibe_coder_generate"))
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isLoading || promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                        Button(settings.localized("vibe_coder_copy_code")) {
                            #if canImport(UIKit)
                            UIPasteboard.general.string = generatedCode
                            #endif
                        }
                        .buttonStyle(.bordered)
                        .disabled(generatedCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    .padding(.horizontal)

                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }
                }
            }
        }
        .navigationTitle(settings.localized("vibe_coder_title"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: onNavigateBack) { Image(systemName: "arrow.left") }
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
                onLoad: { await ensureModelLoaded(force: true) },
                onUnload: { llm.unloadModel() }
            )
        }
        .onAppear {
            if selectedModelName.isEmpty {
                selectedModelName = downloadableFeatureModels().first?.name ?? ""
            }
        }
        .fileImporter(isPresented: $showOpenFilePicker, allowedContentTypes: [.plainText, .sourceCode]) { result in
            switch result {
            case .success(let url):
                openFile(url)
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
        .onDisappear {
            generationTask?.cancel()
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

    private func openFile(_ url: URL) {
        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            currentFileURL = url
            currentFileName = url.lastPathComponent
            generatedCode = text
            chatMessages.append(.init(role: "assistant", text: "Opened \(url.lastPathComponent)"))
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
        currentFileName = name
        currentFileURL = nil
        generatedCode = ""
        chatMessages.append(.init(role: "assistant", text: "Started new file: \(name)"))
        newFileNameInput = ""
    }

    private func saveCurrentFile() {
        guard hasFileSession else {
            errorMessage = settings.localized("vibe_coder_no_file_selected_error")
            return
        }

        let fileURL: URL
        if let currentFileURL {
            fileURL = currentFileURL
        } else {
            let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first ?? FileManager.default.temporaryDirectory
            fileURL = base.appendingPathComponent(currentFileName ?? "main.txt")
            currentFileURL = fileURL
        }

        do {
            try generatedCode.write(to: fileURL, atomically: true, encoding: .utf8)
            chatMessages.append(.init(role: "assistant", text: "Saved \(currentFileName ?? fileURL.lastPathComponent)"))
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func ensureModelLoaded(force: Bool) async {
        guard let model = selectedFeatureModel(named: selectedModelName) else {
            errorMessage = settings.localized("vibe_coder_no_model")
            return
        }
        isLoading = true
        defer { isLoading = false }

        llm.maxTokens = Int(maxTokens)
        llm.contextWindow = model.contextWindowSize > 0 ? model.contextWindowSize : 4096
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

    private func toggleGenerate() {
        if isGenerating {
            generationTask?.cancel()
            generationTask = nil
            isGenerating = false
            return
        }

        generationTask = Task {
            await ensureModelLoaded(force: false)
            guard llm.isLoaded else { return }

            let trimmedPrompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard hasFileSession else {
                errorMessage = settings.localized("vibe_coder_no_file_selected_error")
                return
            }
            guard languagePromptConfig() != nil else {
                errorMessage = "Unsupported or unknown file extension. Use a code file like .py, .js, .ts, .java, .kt, .go, .rs, .cpp, .cs, .html"
                return
            }

            chatMessages.append(.init(role: "user", text: trimmedPrompt))
            chatMessages.append(.init(role: "assistant", text: ""))
            let assistantIndex = chatMessages.count - 1

            isGenerating = true
            do {
                let prompt = buildFileAwareEditPrompt(trimmedPrompt)
                try await llm.generate(prompt: prompt) { text, _, _ in
                    Task { @MainActor in
                        if assistantIndex >= 0 && assistantIndex < chatMessages.count {
                            chatMessages[assistantIndex].text = text
                        }
                    }
                }

                if assistantIndex >= 0 && assistantIndex < chatMessages.count {
                    let extractedCode = extractGeneratedCode(chatMessages[assistantIndex].text)
                    if !extractedCode.isEmpty {
                        generatedCode = extractedCode
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            isGenerating = false
            generationTask = nil
        }
    }
}
