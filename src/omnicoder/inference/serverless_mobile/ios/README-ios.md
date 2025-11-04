# iOS (Core ML MLProgram / MLC-LLM)

Two supported paths for the text LLM decode-step graph:

1) Core ML (MLProgram, stateful decode-step)
   - Export the decode-step model:
     - `python -m omnicoder.export.coreml_decode_export --out weights/text/omnicoder_decode_step.mlmodel --preset mobile_4gb`
   - Integration steps:
     - Add the `.mlmodel` to Xcode; set compute units to `.cpuAndNeuralEngine`.
     - Bind inputs per token: `input_ids` (1x1 int64) plus `k_lat_i`/`v_lat_i` caches for each layer.
     - Keep the recurrent KV outputs `nk_lat_i`/`nv_lat_i` and feed them back on the next step.
     - Control memory: prefer fp16 activations; keep sequence length budget modest; stream tokens.
     - Example Swift pseudocode:
       ```swift
       let model = try OmniCoderDecodeStep(configuration: MLModelConfiguration())
       var kLat: [MLShapedArray<Float32>] = Array(repeating: MLShapedArray<Float32>(repeating: 0.0, shape: [1,H,0,DL]), count: L)
       var vLat: [MLShapedArray<Float32>] = Array(repeating: MLShapedArray<Float32>(repeating: 0.0, shape: [1,H,0,DL]), count: L)
       var token: Int64 = startToken
       for _ in 0..<maxNewTokens {
           let inputIds = try MLMultiArray(shape: [1,1], dataType: .int64)
           inputIds[0] = NSNumber(value: token)
           var inputs: [String: MLFeatureValue] = ["input_ids": MLFeatureValue(multiArray: inputIds)]
           for i in 0..<L { inputs["k_lat_\(i)"] = MLFeatureValue(mlShapedArray: kLat[i]) }
           for i in 0..<L { inputs["v_lat_\(i)"] = MLFeatureValue(mlShapedArray: vLat[i]) }
           let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
           // logits
           let logits = out.featureValue(for: "logits")!.multiArrayValue!
           // argmax/softmax sample to get next token
           token = sampleNextToken(from: logits)
           // update caches
           for i in 0..<L { kLat[i] = out.featureValue(for: "nk_lat_\(i)")!.shapedArrayValue as! MLShapedArray<Float32> }
           for i in 0..<L { vLat[i] = out.featureValue(for: "nv_lat_\(i)")!.shapedArrayValue as! MLShapedArray<Float32> }
       }
       ```
   - Quantization:
     - Prefer int8/int4 weight formats where supported; otherwise keep weights external in lower-precision variants.
     - KV-cache quantization/paging: if a KVQ sidecar (e.g., `omnicoder_decode_step.kvq.json`) exists, store K/V per group (u8/NF4) and dequantize per step. If a paging sidecar (`*.kv_paging.json`) exists, truncate `nk_lat_*`/`nv_lat_*` tails to the page length to bound memory.

2) MLC-LLM (TVM runtime)
   - Compile ONNX decode-step to an MLC package:
     - `python -m omnicoder.export.mlc_compile --onnx weights/text/omnicoder_decode_step.onnx --out_dir weights/text/mlc --target iphone --max_seq_len 4096`
   - Integrate compiled artifacts and runtime into your iOS app.

Notes
- iOS often limits app peak to ~2–3 GB. Use the `mobile_packager_summary.json` to budget and keep decoders lazy-loaded.
- Multimodal decoders (image/video) should be separate bundles loaded on demand, preferably in fp16 and quantized where possible.

### Sample Swift console project (SwiftPM)

For a minimal host-side validation of a decode-step `.mlmodel`, a SwiftPM console stub is included at `serverless_mobile/ios/SampleConsole`:

1) Place your exported `omnicoder_decode_step.mlmodel` under `Sources/Resources/`.
2) Build and run:

```bash
cd src/omnicoder/inference/serverless_mobile/ios/SampleConsole
swift build -c release
swift run App
```

It will compile the model at runtime, execute a single step with zero-length KV caches, and print the first-step latency and logits size.
