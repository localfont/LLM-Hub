import Foundation
#if canImport(Network)
import Network

public final class LocalHTMLPreviewServer: @unchecked Sendable {
    public static let shared = LocalHTMLPreviewServer()
    
    private var listener: NWListener?
    private var html: String = ""
    
    public func stop() {
        listener?.cancel()
        listener = nil
        html = ""
    }
    
    public func start(html: String) async throws -> URL {
        stop()
        self.html = html
        
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
                        continuation.resume(throwing: NSError(domain: "LocalHTMLPreviewServer", code: 1))
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
            throw NSError(domain: "LocalHTMLPreviewServer", code: 3)
        }
        return url
    }
    
    private static func handle(connection: NWConnection, html: String) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, _, _ in
            let request = String(data: data ?? Data(), encoding: .utf8) ?? ""
            let firstLine = request.split(separator: "\n").first.map(String.init) ?? ""
            let isFavicon = firstLine.contains("/favicon")
            
            if isFavicon {
                let response = "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"
                connection.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in
                    connection.cancel()
                })
                return
            }
            
            let bodyData = html.data(using: .utf8) ?? Data()
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: \(bodyData.count)\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
            var response = Data()
            response.append(header.data(using: .utf8) ?? Data())
            response.append(bodyData)
            
            connection.send(content: response, completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }
}
#endif
