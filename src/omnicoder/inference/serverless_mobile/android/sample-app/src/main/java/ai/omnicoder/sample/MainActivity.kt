package ai.omnicoder.sample

import android.app.Activity
import android.os.Bundle
import android.widget.TextView
import ai.onnxruntime.*
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.text = runOnce()
        setContentView(tv)
    }

    private fun runOnce(): String {
        try {
            val env = OrtEnvironment.getEnvironment()
            val opts = SessionOptions()
            // Enable NNAPI if present (best-effort)
            try {
                val epOpts = HashMap<String, String>()
                epOpts["nnapi_accelerator_name"] = "NnapiAccelerator.qnn"
                opts.addNnapi(epOpts)
            } catch (_: Exception) {}
            // Copy ONNX from assets to files dir on first run
            val modelFile = java.io.File(filesDir, "omnicoder_decode_step.onnx")
            if (!modelFile.exists()) {
                assets.open("omnicoder_decode_step.onnx").use { src ->
                    modelFile.outputStream().use { dst -> src.copyTo(dst) }
                }
            }
            val modelPath = modelFile.absolutePath
            val sess = env.createSession(modelPath, opts)

            // Discover required K/V inputs and their static (H, DL) dims
            val inputInfo = sess.inputInfo
            val kNames = mutableListOf<String>()
            val vNames = mutableListOf<String>()
            var heads = 8L
            var dl = 16L
            for ((name, info) in inputInfo) {
                if (name.startsWith("k_lat_")) {
                    kNames.add(name)
                    val shape = (info.info as TensorInfo).shape
                    // shape: (B, H, T_past, DL)
                    if (shape.size >= 4) {
                        heads = kotlin.math.max(1L, shape[1])
                        dl = kotlin.math.max(1L, shape[3])
                    }
                } else if (name.startsWith("v_lat_")) {
                    vNames.add(name)
                }
            }
            kNames.sort()
            vNames.sort()
            val numLayers = kotlin.math.min(kNames.size, vNames.size)

            // Minimal tokenizer: map ASCII chars to ids (mod 32000)
            fun encode(text: String): LongArray {
                val out = LongArray(text.length + 1)
                // BOS token = 1
                out[0] = 1
                for (i in text.indices) {
                    out[i + 1] = (text[i].code % 32000).toLong()
                }
                return out
            }
            fun decodeToken(id: Long): Char {
                val c = (id % 95 + 32).toInt() // printable ASCII range
                return c.toChar()
            }

            val prompt = "Hello from Android"
            val ids = encode(prompt)
            var lastId = ids.last()
            val maxGen = 32
            val sb = StringBuilder(prompt)

            // Allocate zero-length past for first step
            fun zeroKv(env: OrtEnvironment): Pair<Map<String, OnnxTensor>, MutableMap<String, OnnxTensor>> {
                val inMap = HashMap<String, OnnxTensor>()
                val alloc = { shape: LongArray ->
                    OnnxTensor.createTensor(env, FloatArray(0), shape)
                }
                // T_past=0 tensors
                val kShape = longArrayOf(1L, heads, 0L, dl)
                val vShape = longArrayOf(1L, heads, 0L, dl)
                for (i in 0 until numLayers) {
                    inMap[kNames[i]] = alloc(kShape)
                    inMap[vNames[i]] = alloc(vShape)
                }
                return Pair(inMap, HashMap())
            }

            var (kvInputs, kvNext) = zeroKv(env)
            for (_step in 0 until maxGen) {
                val inputs = HashMap<String, OnnxTensor>()
                inputs.putAll(kvInputs)
                // input_ids
                val inputIds = OnnxTensor.createTensor(env, longArrayOf(lastId), longArrayOf(1L, 1L))
                inputs["input_ids"] = inputIds
                // Run
                val res = sess.run(inputs)
                // Collect logits
                val logits = (res.get("logits") as OnnxTensor).floatBuffer
                // Argmax over last dimension
                var bestIdx = 0
                var bestVal = Float.NEGATIVE_INFINITY
                // Assume vocab=32000 or derive from tensor length
                val vocab = logits.capacity()
                for (i in 0 until vocab) {
                    val v = logits.get(i)
                    if (v > bestVal) { bestVal = v; bestIdx = i }
                }
                // Collect new K/V for next step
                val newKv = HashMap<String, OnnxTensor>()
                for (i in 0 until numLayers) {
                    val nk = res.get("nk_lat_${i}") as OnnxTensor
                    val nv = res.get("nv_lat_${i}") as OnnxTensor
                    newKv[kNames[i]] = nk
                    newKv[vNames[i]] = nv
                }
                kvInputs = newKv
                lastId = bestIdx.toLong()
                sb.append(decodeToken(lastId))
                res.close()
                inputIds.close()
            }
            return sb.toString()
        } catch (e: Exception) {
            return "ERR: ${e.message}"
        }
    }
}


