import Foundation

private extension URLError.Code {
    var isTransientDownloadFailure: Bool {
        switch self {
        case .networkConnectionLost,
            .notConnectedToInternet,
            .timedOut,
            .cannotConnectToHost,
            .cannotFindHost,
            .dnsLookupFailed,
            .resourceUnavailable,
            .internationalRoamingOff,
            .callIsActive,
            .dataNotAllowed:
            return true
        default:
            return false
        }
    }
}

public struct DownloadUpdate: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let speedBytesPerSecond: Double
}

public actor ModelDownloader {
    public static let shared = ModelDownloader()
    
    private let urlSession: URLSession
    private let completionThresholdRatio: Double = 0.98
    
    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 3600 // 1 hour for large models
        self.urlSession = URLSession(configuration: config)
    }

    private func remoteFileSize(fileURL: URL, hfToken: String?) async -> Int64? {
        var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
        request.httpMethod = "HEAD"
        if let token = hfToken, !token.isEmpty {
            request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (_, response) = try await urlSession.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                (200...299).contains(httpResponse.statusCode)
            else {
                return nil
            }
            if let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length"),
                let size = Int64(contentLength),
                size > 0
            {
                return size
            }
        } catch {
            // If HEAD fails, fall back to downloading file again.
        }
        return nil
    }

    private func localFileSize(at url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
            let fileSize = attrs[.size] as? Int64
        else {
            return 0
        }
        return max(0, fileSize)
    }
    
    public func downloadModel(
        _ model: AIModel,
        hfToken: String?,
        destinationDir: URL,
        onProgress: @Sendable @escaping (DownloadUpdate) -> Void
    ) async throws {
        let totalSize = model.sizeBytes
        var downloadedBytesPerFile: [String: Int64] = [:]
        let startTime = Date()
        
        // Ensure clean destination
        if !FileManager.default.fileExists(atPath: destinationDir.path) {
            try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)
        }
        
        for fileName in model.files {
            // Encode repoId and fileName separately to avoid corrupting URL structure
            // Use urlPathAllowed but carefully since repoId has a slash
            let encodedRepoId = model.repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? model.repoId
            let encodedFileName = fileName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? fileName
            let urlString = "https://huggingface.co/\(encodedRepoId)/resolve/main/\(encodedFileName)"
            
            guard let fileURL = URL(string: urlString) else { continue }
            
            let destinationFileURL = destinationDir.appendingPathComponent(fileName)
            
            // Check if file exists and is already downloaded fully.
            // We only skip when local size matches remote Content-Length.
            if FileManager.default.fileExists(atPath: destinationFileURL.path) {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destinationFileURL.path),
                   let fileSize = attrs[.size] as? Int64,
                   fileSize > 0 {
                    let expectedSize = await remoteFileSize(fileURL: fileURL, hfToken: hfToken)
                    if expectedSize == nil || expectedSize == fileSize {
                        downloadedBytesPerFile[fileName] = fileSize
                        let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                        let elapsed = Date().timeIntervalSince(startTime)
                        let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                        onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                        continue
                    }
                }
            }
            
            let maxRetries = 6
            var attempt = 0
            var finishedFile = false

            while !finishedFile {
                do {
                    var existingBytes = localFileSize(at: destinationFileURL)
                    var request = URLRequest(url: fileURL, cachePolicy: .reloadIgnoringLocalCacheData)
                    if let token = hfToken, !token.isEmpty {
                        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                    }
                    if existingBytes > 0 {
                        request.addValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                    }

                    let (bytes, response) = try await urlSession.bytes(for: request)
                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw NSError(domain: "ModelDownloader", code: -1, userInfo: [NSLocalizedDescriptionKey: "No Response"])
                    }

                    // Critical 404/403 Handling
                    if !(200...299).contains(httpResponse.statusCode) {
                        // Ignore missing optional files (like chat_template.jinja in some older MLX repos)
                        if httpResponse.statusCode == 404 && fileName == "chat_template.jinja" {
                            finishedFile = true
                            break
                        }

                        let reason = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                        throw NSError(domain: "ModelDownloader", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode): \(reason)"])
                    }

                    // Efficient Buffered Write with resume support.
                    // 206 means server accepted Range and we should append.
                    // 200 means full content, so restart file from zero.
                    if !FileManager.default.fileExists(atPath: destinationFileURL.path) {
                        FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
                    }
                    if existingBytes > 0 && httpResponse.statusCode == 200 {
                        try? FileManager.default.removeItem(at: destinationFileURL)
                        FileManager.default.createFile(atPath: destinationFileURL.path, contents: nil)
                        existingBytes = 0
                    }
                    let fileHandle = try FileHandle(forWritingTo: destinationFileURL)
                    defer { try? fileHandle.close() }
                    if existingBytes > 0 {
                        try fileHandle.seekToEnd()
                    } else {
                        try fileHandle.truncate(atOffset: 0)
                    }

                    var byteCountPerFile: Int64 = existingBytes
                    var buffer = Data()
                    let chunkSize = 64 * 1024 // 64KB buffer

                    for try await byte in bytes {
                        buffer.append(byte)
                        byteCountPerFile += 1

                        if buffer.count >= chunkSize {
                            try fileHandle.write(contentsOf: buffer)
                            buffer.removeAll(keepingCapacity: true)

                            // Periodic Progress Update
                            downloadedBytesPerFile[fileName] = byteCountPerFile
                            let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                            let elapsed = Date().timeIntervalSince(startTime)
                            let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                            onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                        }
                    }

                    if !buffer.isEmpty {
                        try fileHandle.write(contentsOf: buffer)
                        buffer.removeAll()
                    }

                    downloadedBytesPerFile[fileName] = byteCountPerFile
                    let currentTotal = downloadedBytesPerFile.values.reduce(0, +)
                    let elapsed = Date().timeIntervalSince(startTime)
                    let speed = elapsed > 0 ? Double(currentTotal) / elapsed : 0
                    onProgress(DownloadUpdate(bytesDownloaded: currentTotal, totalBytes: totalSize, speedBytesPerSecond: speed))
                    finishedFile = true
                } catch let error as URLError where error.code.isTransientDownloadFailure && attempt < maxRetries {
                    attempt += 1
                    let delaySeconds = min(pow(2.0, Double(attempt - 1)), 30.0)
                    try await Task.sleep(for: .seconds(delaySeconds))
                }
            }
        }

        let finalBytes = downloadedBytesPerFile.values.reduce(0, +)
        let minimumExpectedBytes = Int64(Double(totalSize) * completionThresholdRatio)
        if finalBytes < minimumExpectedBytes {
            throw NSError(
                domain: "ModelDownloader",
                code: -2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Incomplete download: \(finalBytes) / \(totalSize) bytes"
                ]
            )
        }
    }
}
