import SwiftUI

// MARK: - Settings Screen (mirroring Android SettingsScreen.kt)
struct SettingsScreen: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.openURL) var openURL

    var onNavigateBack: () -> Void
    var onNavigateToModels: () -> Void

    @State private var showLanguageDialog = false
    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            List {
                // MARK: Models Section
                Section {
                    SettingsRow(
                        icon: "square.and.arrow.down.fill",
                        iconColor: .indigo,
                        titleKey: "download_models",
                        subtitleKey: "browse_download_models"
                    ) {
                        onNavigateToModels()
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "models", icon: "cpu")
                }

                // MARK: Appearance Section
                Section {
                    SettingsRow(
                        icon: "globe",
                        iconColor: .blue,
                        titleKey: "language",
                        subtitleString: settings.localized(settings.selectedLanguage.displayNameKey)
                    ) {
                        showLanguageDialog = true
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "appearance", icon: "paintbrush")
                }

                // MARK: Information Section
                Section {
                    SettingsRow(
                        icon: "info.circle.fill",
                        iconColor: .blue,
                        titleKey: "about",
                        subtitleKey: "app_information_contact"
                    ) { }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)

                    SettingsRow(
                        icon: "doc.text.fill",
                        iconColor: .gray,
                        titleKey: "terms_of_service",
                        subtitleKey: "legal_terms_conditions"
                    ) { }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "information", icon: "info.circle")
                }

                // MARK: Source Code Section
                Section {
                    SettingsRow(
                        icon: "chevron.left.forwardslash.chevron.right",
                        iconColor: .gray,
                        titleKey: "github_repository",
                        subtitleKey: "view_source_contribute"
                    ) {
                        if let url = URL(string: "https://github.com/timmyy123/LLM-Hub") {
                            openURL(url)
                        }
                    }
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                    .listRowBackground(Color.clear)
                } header: {
                    SectionHeader(titleKey: "source_code_section", icon: "curlybraces")
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .navigationTitle(settings.localized("feature_settings_title"))
        .navigationBarTitleDisplayMode(.large)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text(settings.localized("back"))
                    }
                }
            }
        }
        // Language Dialog
        .sheet(isPresented: $showLanguageDialog) {
            LanguagePickerSheet()
                .environmentObject(settings)
        }
    }
}

// MARK: - Language Picker Sheet
struct LanguagePickerSheet: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.dismiss) var dismiss

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            List {
                ForEach(AppLanguage.allCases) { lang in
                    Button {
                        settings.selectedLanguage = lang
                        dismiss()
                    } label: {
                        HStack {
                            Text(settings.localized(lang.displayNameKey))
                                .foregroundColor(.white)
                                .environment(\.layoutDirection, lang.isRTL ? .rightToLeft : .leftToRight)
                            Spacer()
                            if settings.selectedLanguage == lang {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.blue.opacity(0.9))
                            }
                        }
                    }
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets(top: 6, leading: 14, bottom: 6, trailing: 14))
                }
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
        }
        .navigationTitle(settings.localized("select_language"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button(settings.localized("done")) { dismiss() }
            }
        }
    }
}


// MARK: - Reusable Components

struct SectionHeader: View {
    @EnvironmentObject var settings: AppSettings
    let titleKey: String
    let icon: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(settings.localized(titleKey))
        }

        .font(.footnote.bold())
        .foregroundColor(.white.opacity(0.74))
        .textCase(nil)
    }
}

struct SettingsRow: View {
    @EnvironmentObject var settings: AppSettings
    let icon: String
    let iconColor: Color
    let titleKey: String
    var subtitleKey: String? = nil
    var subtitleString: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 14) {
                RoundedRectangle(cornerRadius: 8)
                    .fill(iconColor.gradient)
                    .frame(width: 32, height: 32)
                    .overlay {
                        Image(systemName: icon)
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    }

                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized(titleKey))
                        .font(.subheadline)
                        .foregroundColor(.white)
                    if let sk = subtitleKey {
                        Text(settings.localized(sk))
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.65))
                            .lineLimit(1)
                    } else if let ss = subtitleString {
                        Text(verbatim: ss)
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.65))
                            .lineLimit(1)
                    }
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.caption.bold())
                    .foregroundColor(.white.opacity(0.55))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.16), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct SettingsToggleRow: View {
    let icon: String
    let iconColor: Color
    let title: String
    let subtitle: String
    @Binding var isOn: Bool

    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 8)
                .fill(iconColor.gradient)
                .frame(width: 32, height: 32)
                .overlay {
                    Image(systemName: icon)
                        .font(.system(size: 16))
                        .foregroundColor(.white)
                }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.primary)
                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            Toggle("", isOn: $isOn)
                .labelsHidden()
        }
    }
}
