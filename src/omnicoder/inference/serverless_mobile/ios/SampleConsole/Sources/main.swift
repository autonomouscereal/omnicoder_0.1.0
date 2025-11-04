import Foundation
import CoreML

@main
struct OmniCoderDecodeConsole {
    static func main() throws {
        // Load MLModel from bundled resources
        guard let url = Bundle.module.url(forResource: "omnicoder_decode_step", withExtension: "mlmodel") else {
            print("[error] omnicoder_decode_step.mlmodel not found in Resources")
            return
        }
        let compiled = try MLModel.compileModel(at: url)
        let model = try MLModel(contentsOf: compiled)

        // Discover layer count by inspecting model description (assumes naming convention)
        let inputKeys = Array(model.modelDescription.inputDescriptionsByName.keys)
        let kKeys = inputKeys.filter { $0.hasPrefix("k_lat_") }
        let vKeys = inputKeys.filter { $0.hasPrefix("v_lat_") }
        let L = min(kKeys.count, vKeys.count)

        // Tiny ASCII tokenizer for demo
        func encode(_ text: String) throws -> [Int64] {
            var ids: [Int64] = [1] // BOS
            for ch in text.unicodeScalars { ids.append(Int64(ch.value % 32000)) }
            return ids
        }
        func decode(_ id: Int64) -> String { String(UnicodeScalar(UInt8((id % 95) + 32))) }

        // Discover (H,DL) from first k_lat_0 input if static dims are present; else default
        var heads: Int64 = 8
        var dl: Int64 = 16
        if let anyK = model.modelDescription.inputDescriptionsByName["k_lat_0"]?.multiArrayConstraint?.shape {
            if anyK.count >= 4, let h = anyK[1] as? Int64, let d = anyK[3] as? Int64 { heads = h; dl = d }
        }

        // Streaming loop
        let prompt = "Hello from Swift"
        let ids = try encode(prompt)
        var lastId = ids.last ?? 1
        var textOut = prompt
        for _ in 0..<32 {
            var feats: [String: MLFeatureValue] = [:]
            let inputIds = try MLMultiArray(shape: [1,1], dataType: .int64)
            inputIds[0] = NSNumber(value: lastId)
            feats["input_ids"] = MLFeatureValue(multiArray: inputIds)
            // Supply zero-length past for first steps; could maintain rolling K/V by keeping outputs
            for i in 0..<L {
                let k = try MLMultiArray(shape: [1, heads, 0, dl], dataType: .float32)
                let v = try MLMultiArray(shape: [1, heads, 0, dl], dataType: .float32)
                feats["k_lat_\(i)"] = MLFeatureValue(multiArray: k)
                feats["v_lat_\(i)"] = MLFeatureValue(multiArray: v)
            }
            let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats))
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else { break }
            // Argmax over vocab
            var bestIdx: Int = 0
            var bestVal: Float = -Float.greatestFiniteMagnitude
            let count = logits.count
            for i in 0..<count {
                let v = logits[i].floatValue
                if v > bestVal { bestVal = v; bestIdx = i }
            }
            lastId = Int64(bestIdx)
            textOut.append(decode(lastId))
        }
        print(textOut)
    }
}


