import SwiftUI

struct ApolloLiquidBackground: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(hex: "0b0f19"), Color(hex: "131a2a")],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            Circle()
                .fill(Color.white.opacity(0.18))
                .frame(width: 260, height: 260)
                .blur(radius: 80)
                .offset(x: 130, y: -260)

            Circle()
                .fill(Color.blue.opacity(0.20))
                .frame(width: 220, height: 220)
                .blur(radius: 90)
                .offset(x: -140, y: -220)
        }
    }
}

struct ApolloIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(.white.opacity(configuration.isPressed ? 0.75 : 0.95))
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
    }
}

private struct ApolloScreenBackgroundModifier: ViewModifier {
    func body(content: Content) -> some View {
        ZStack {
            ApolloLiquidBackground()
            content
        }
    }
}

extension View {
    func apolloScreenBackground() -> some View {
        modifier(ApolloScreenBackgroundModifier())
    }
}
