import SwiftUI
import RunAnywhere

struct ChatSettingsSheet: View {
    @ObservedObject var vm: ChatViewModel
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss
    @State private var draftContextWindow: Double = 2048
    @State private var draftMaxTokens: Double = 1024
    @State private var draftTopK: Double = 64
    @State private var draftTopP: Double = 0.95
    @State private var draftTemperature: Double = 1.0
    
    var body: some View {
        NavigationView {
            ZStack {
                ApolloLiquidBackground()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Model Selection Header
                        VStack(spacing: 16) {
                            HStack(alignment: .center, spacing: 14) {
                                Image(systemName: "cpu")
                                    .font(.system(size: 26))
                                    .foregroundColor(.blue.opacity(0.85))
                                    .frame(width: 32)
                                
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(settings.localized("select_model_title"))
                                        .font(.headline)
                                        .foregroundColor(.white)
                                    Text(settings.localized("select_model"))
                                        .font(.caption)
                                        .foregroundColor(.white.opacity(0.68))
                                }
                                .padding(.vertical, 4)
                                
                                Spacer()
                                
                                Picker("", selection: $vm.selectedModelName) {
                                    if vm.selectedModelName == settings.localized("no_model_selected") {
                                        Text(settings.localized("no_model_selected")).tag(settings.localized("no_model_selected"))
                                    }
                                    ForEach(downloadedModels) { model in
                                        Text(model.name).tag(model.name)
                                    }
                                }
                                .pickerStyle(.menu)
                                .accentColor(.blue.opacity(0.88))
                                .labelsHidden()
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            
                            if currentModel != nil {
                                Divider()
                                HStack {
                                    Label(settings.localized("currently_loaded"), systemImage: "bolt.circle.fill")
                                        .font(.subheadline)
                                        .foregroundColor(.white.opacity(0.72))
                                    Spacer()
                                    if vm.selectedModelName == vm.loadedModelName {
                                        Text("✅ " + settings.localized("ready_to_chat"))
                                            .font(.caption.bold())
                                            .foregroundColor(.green)
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)

                        // Model Configurations
                        VStack(alignment: .leading, spacing: 20) {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.blue.opacity(0.86))
                                Text(settings.localized("model_configs_title"))
                                    .font(.headline)
                                    .foregroundColor(.white)
                            }
                            .padding(.bottom, 8)
                            
                            ConfigSlider(title: settings.localized("context_window_size"), value: $draftContextWindow, range: 1...modelMaxContextWindow, format: "%.0f", subtitle: "max \(Int(modelMaxContextWindow))", step: contextWindowStep, onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("max_tokens"), value: $draftMaxTokens, range: 1...draftMaxTokensCap, format: "%.0f", subtitle: "<= context", onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("top_k"), value: $draftTopK, range: 1...256, format: "%.0f", onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("top_p"), value: $draftTopP, range: 0...1, format: "%.2f", onCommit: applyDraftToViewModel)
                            ConfigSlider(title: settings.localized("temperature"), value: $draftTemperature, range: 0...2, format: "%.2f", onCommit: applyDraftToViewModel)

                            HStack {
                                Spacer()
                                Button(settings.localized("reset_to_defaults")) {
                                    resetAllConfigsToDefaults()
                                }
                                .font(.subheadline.weight(.semibold))
                                .foregroundColor(.blue.opacity(0.92))
                            }
                            
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.14), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)

                        // Modality Toggle Tiles
                        if let model = currentModel, (model.supportsVision || model.supportsAudio || model.supportsThinking) {
                            VStack(alignment: .leading, spacing: 16) {
                                HStack {
                                    Image(systemName: "sparkles")
                                        .foregroundColor(.blue.opacity(0.86))
                                    Text(settings.localized("modality_options"))
                                        .font(.headline)
                                        .foregroundColor(.white)
                                }
                                .padding(.bottom, 4)
                                
                                if model.supportsVision {
                                    ToggleTile(title: settings.localized("enable_vision"), isOn: $vm.enableVision, icon: "eye.fill")
                                }
                                if model.supportsAudio {
                                    ToggleTile(title: settings.localized("enable_audio"), isOn: $vm.enableAudio, icon: "mic.fill")
                                }
                                if model.supportsThinking {
                                    ToggleTile(title: settings.localized("enable_thinking"), isOn: $vm.enableThinking, icon: "brain")
                                }
                            }
                            .padding()
                            .background(.ultraThinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.white.opacity(0.14), lineWidth: 1)
                            )
                            .shadow(color: .black.opacity(0.3), radius: 12, x: 0, y: 8)
                        }
                        
                        // Action Buttons
                        VStack(spacing: 12) {
                            Button(action: {
                                Task {
                                    await vm.loadModelIfNecessary(force: true)
                                    dismiss()
                                }
                            }) {
                                HStack {
                                    if vm.isBackendLoading {
                                        ProgressView().padding(.trailing, 8)
                                    } else {
                                        Image(systemName: "arrow.clockwise.circle.fill")
                                            .padding(.trailing, 4)
                                    }
                                    Text(settings.localized("load_model"))
                                        .fontWeight(.bold)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .foregroundColor(.white)
                                .contentShape(Rectangle())
                            }
                            .liquidGlassPrimaryButton(cornerRadius: 14)
                            .disabled(vm.isBackendLoading || vm.selectedModelName == settings.localized("no_model_selected"))
                            
                            if vm.loadedModelName != nil {
                                Button(action: {
                                    vm.unloadModel()
                                    dismiss()
                                }) {
                                    Text(settings.localized("unload_model"))
                                        .fontWeight(.medium)
                                        .foregroundColor(.red.opacity(0.94))
                                        .frame(maxWidth: .infinity)
                                        .padding()
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 14))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 14)
                                                .stroke(Color.red.opacity(0.34), lineWidth: 1)
                                        )
                                }
                            }
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
            .onAppear {
                syncDraftFromViewModel()
            }
            .onChange(of: vm.selectedModelName) { _ in
                syncDraftFromViewModel()
            }
            .onChange(of: draftContextWindow) { _, newValue in
                let cap = min(max(1, newValue), modelMaxContextWindow)
                if draftMaxTokens > cap {
                    draftMaxTokens = cap
                }
            }
        }
    }
    
    private var downloadedModels: [AIModel] {
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
    
    private var currentModel: AIModel? {
        ModelData.models.first(where: { $0.name == vm.selectedModelName })
    }

    private var modelMaxContextWindow: Double {
        guard let currentModel else { return 2048 }
        let advertised = currentModel.contextWindowSize > 0 ? currentModel.contextWindowSize : 2048
        return Double(max(1, advertised))
    }

    private var draftMaxTokensCap: Double {
        min(max(1, draftContextWindow), modelMaxContextWindow)
    }

    private var contextWindowStep: Double {
        let maxWindow = max(1, Int(modelMaxContextWindow))
        return Double(max(1, maxWindow / 1024))
    }

    private func syncDraftFromViewModel() {
        draftContextWindow = min(max(1, vm.contextWindow), modelMaxContextWindow)
        draftMaxTokens = min(max(1, vm.maxTokens), min(max(1, draftContextWindow), modelMaxContextWindow))
        draftTopK = min(max(1, vm.topK), 256)
        draftTopP = min(max(0, vm.topP), 1)
        draftTemperature = min(max(0, vm.temperature), 2)
    }

    private func applyDraftToViewModel() {
        let clampedContext = min(max(1, draftContextWindow), modelMaxContextWindow)
        let clampedMaxTokens = min(max(1, draftMaxTokens), min(max(1, clampedContext), modelMaxContextWindow))
        let clampedTopK = min(max(1, draftTopK), 256)
        let clampedTopP = min(max(0, draftTopP), 1)
        let clampedTemperature = min(max(0, draftTemperature), 2)

        draftContextWindow = clampedContext
        draftMaxTokens = clampedMaxTokens
        draftTopK = clampedTopK
        draftTopP = clampedTopP
        draftTemperature = clampedTemperature

        vm.contextWindow = clampedContext
        vm.maxTokens = clampedMaxTokens
        vm.topK = clampedTopK
        vm.topP = clampedTopP
        vm.temperature = clampedTemperature
    }

    private func resetAllConfigsToDefaults() {
        draftContextWindow = min(2048, modelMaxContextWindow)
        draftMaxTokens = min(4096, min(max(1, draftContextWindow), modelMaxContextWindow))
        draftTopK = 64
        draftTopP = 0.95
        draftTemperature = 1.0
        applyDraftToViewModel()
    }

    private var draftedModels: [AIModel] {
        downloadedModels
    }
}

struct ConfigSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    var subtitle: String? = nil
    var step: Double? = nil
    let onCommit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.white)
                if let subtitle = subtitle {
                    Text("(\(subtitle))")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.58))
                }
                Spacer()
                Text(String(format: format, value))
                    .font(.system(.subheadline, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.white.opacity(0.92))
            }
            Group {
                if let step {
                    Slider(value: $value, in: range, step: step) { editing in
                        if !editing {
                            onCommit()
                        }
                    }
                } else {
                    Slider(value: $value, in: range) { editing in
                        if !editing {
                            onCommit()
                        }
                    }
                }
            }
            .tint(.white.opacity(0.92))
        }
    }
}

struct ToggleTile: View {
    let title: String
    @Binding var isOn: Bool
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue.opacity(0.75))
                .frame(width: 24)
            Text(title)
                .font(.subheadline)
                .foregroundColor(.white)
            Spacer()
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(.white.opacity(0.9))
        }
    }
}

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
