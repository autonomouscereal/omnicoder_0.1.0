# Android (ExecuTorch / NNAPI / ORT-mobile / llama.cpp JNI)

This repo supports multiple deployment paths for the text LLM decode-step graph. Pick one:

1) ExecuTorch (recommended for PyTorch-based stacks)
   - Export the stateful decode-step program:
     - One-command packager (will attempt .pte and fall back to .pt):
       - `python -m omnicoder.export.mobile_packager --preset mobile_4gb --out_dir weights/text --seq_len_budget 4096 --export_executorch`
     - Or direct exporter: `python -m omnicoder.export.executorch_export --out weights/text/omnicoder_decode_step.pte`
   - Android app integration:
     - Add ExecuTorch AAR to your app and initialize runtime.
     - Load `omnicoder_decode_step.pte` via assets.
     - Use NNAPI delegate where available for acceleration.
     - Maintain KV-cache tensors between steps: feed `k_lat_i`/`v_lat_i` back into the next call; read `nk_lat_i`/`nv_lat_i` outputs.
   - Quantization:
     - For end-to-end int4/int8, quantize weights prior to export (AWQ/GPTQ for LLM; activation quant on-device varies by operator).

2) ONNX Runtime Mobile (ORT-mobile)
   - Export ONNX decode-step graph: `python -m omnicoder.export.onnx_export --output weights/text/omnicoder_decode_step.onnx --seq_len 1 --mobile_preset mobile_4gb --decode_step`
    - In Android app, use ORT-mobile AAR with NNAPI/CPU EP. Example per-token streaming with NNAPI EP:
      ```java
      OrtEnvironment env = OrtEnvironment.getEnvironment();
      SessionOptions opts = new SessionOptions();
      Map<String, String> epOptions = new HashMap<>();
      epOptions.put("nnapi_accelerator_name", "NnapiAccelerator.qnn"); // .gpu/.npu as available
      opts.addNnapi(epOptions);
      OrtSession sess = env.createSession(modelPath, opts);

      // Allocate recurrent KV state
      OnnxTensor inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{startToken}), new long[]{1,1});
      OnnxTensor[] pastK = new OnnxTensor[L];
      OnnxTensor[] pastV = new OnnxTensor[L];
      for (int i=0;i<L;i++) {
          pastK[i] = OnnxTensor.createTensor(env, FloatBuffer.allocate(H*0*DL), new long[]{1,H,0,DL});
          pastV[i] = OnnxTensor.createTensor(env, FloatBuffer.allocate(H*0*DL), new long[]{1,H,0,DL});
      }
      Map<String, OnnxTensor> inputs = new HashMap<>();
      inputs.put("input_ids", inputIds);
      for (int i=0;i<L;i++) inputs.put("k_lat_"+i, pastK[i]);
      for (int i=0;i<L;i++) inputs.put("v_lat_"+i, pastV[i]);
      OrtSession.Result res = sess.run(inputs);
      float[] logits = ((OnnxTensor)res.get(0)).getFloatBuffer().array();
      for (int i=0;i<L;i++) pastK[i] = (OnnxTensor)res.get(1+i);
      for (int i=0;i<L;i++) pastV[i] = (OnnxTensor)res.get(1+L+i);
      ```
   - Stream generation:
     - Inspect model inputs: `input_ids`, `k_lat_0..L-1`, `v_lat_0..L-1`.
     - Call session per token; update caches with `nk_lat_*`/`nv_lat_*` outputs.
   - Optional: dynamic int8 PTQ via packager `--quantize_onnx`.
   - KV-cache quantization and paging:
     - If a KVQ sidecar (e.g., `omnicoder_decode_step.kvq.json`) exists, store K/V per group (u8/NF4) and dequantize per step for compute. Calibrations are emitted as sidecars.
     - Use paging sidecar (`*.kv_paging.json`) to cap in-RAM KV by retaining only recent pages; your app should truncate `nk_lat_*`/`nv_lat_*` tails accordingly.

3) llama.cpp (GGUF) via JNI (if you switch to a llama.cpp-supported LLM)
   - Quantize a supported LLM to GGUF (e.g., Q4_K_M).
   - Build `llama.cpp` for Android (NDK) and include JNI bridge.
   - Stream tokens using the library; KV-cache is internal.

Notes
- Keep peak memory < 2–4 GB: use `mobile_packager_summary.json` to budget KV+weights.
- Lazy-load multimodal decoders (image/video/audio) only when needed.
- Test on representative devices; tune `seq_len_budget` to your UI needs.
