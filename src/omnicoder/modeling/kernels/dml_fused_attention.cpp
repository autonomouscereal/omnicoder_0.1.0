// Omnicoder DirectML fused attention and INT4 matmul
// Device-agnostic ATen implementations registered under CompositeImplicitAutograd.
// When tensors live on DirectML (torch-directml), ATen ops dispatch to DML kernels.

#include <torch/script.h>
#include <ATen/ATen.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <string>

// Note: We intentionally do not bind to DirectML here. We rely on PyTorch's DirectML backend
// to execute ATen ops on the DML device when tensors are on that device.

namespace {
// Simple per-shape mask cache to reduce repeated host->device transfers on DML
static std::mutex g_mask_mtx;
static std::unordered_map<std::string, at::Tensor> g_mask_cache;

static std::string key_for(const at::Tensor& t, const at::Device& dev) {
  // key: rows x cols @ device
  auto sizes = t.sizes();
  int64_t rows = sizes.size() > 0 ? sizes[0] : 0;
  int64_t cols = sizes.size() > 1 ? sizes[1] : 0;
  return std::to_string(rows) + "x" + std::to_string(cols) + "@" + std::to_string((int)dev.index());
}
}

torch::Tensor dml_mla(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor attn_mask, bool is_causal) {
  c10::optional<torch::Tensor> mask = c10::nullopt;
  if (attn_mask.defined() && attn_mask.numel() > 0) {
    // Move/copy mask to q's device once and cache by (rows,cols,device)
    try {
      auto dev = q.device();
      std::string key = key_for(attn_mask, dev);
      {
        std::lock_guard<std::mutex> _g(g_mask_mtx);
        auto it = g_mask_cache.find(key);
        if (it != g_mask_cache.end() && it->second.device() == dev) {
          mask = it->second;
        } else {
          auto m = attn_mask.to(dev, /*non_blocking=*/true).contiguous();
          g_mask_cache[key] = m;
          mask = m;
        }
      }
    } catch (...) {
      mask = attn_mask;
    }
  }
  // Use ATen SDPA; when tensors are on DML, this will execute on DirectML backend.
  return torch::nn::functional::scaled_dot_product_attention(
      q, k, v, /*attn_mask=*/mask, /*dropout_p=*/0.0, /*is_causal=*/is_causal);
}

TORCH_LIBRARY(omnicoder_dml, m) {
  m.def("mla(Tensor q, Tensor k, Tensor v, Tensor attn_mask, bool is_causal) -> Tensor");
  m.def("matmul_int4(Tensor x, Tensor packed_w, Tensor scale, Tensor zero) -> Tensor");
}

// CompositeImplicitAutograd makes the implementation device-agnostic while using ATen ops under the hood.
TORCH_LIBRARY_IMPL(omnicoder_dml, CompositeImplicitAutograd, m) {
  m.impl("mla", dml_mla);
  m.impl("matmul_int4", [](torch::Tensor x, torch::Tensor packed_w, torch::Tensor scale, torch::Tensor zero) {
    // Unpack low/high nibbles along last-dim, dequantize to float32 on the same device as inputs
    auto device = x.device();
    auto pw = packed_w.to(device, /*non_blocking=*/true);
    auto sc = scale.to(device, /*non_blocking=*/true).to(torch::kFloat32);
    auto zc = zero.to(device, /*non_blocking=*/true).to(torch::kFloat32);
    auto low = (pw & 0x0Fu).to(torch::kInt8);
    auto high = ((pw >> 4) & 0x0Fu).to(torch::kInt8);
    // Respect nibble order via env OMNICODER_INT4_NIBBLE_ORDER (low_first|high_first)
    bool high_first = false;
    if (const char* env = std::getenv("OMNICODER_INT4_NIBBLE_ORDER")) {
      if (std::strcmp(env, "high_first") == 0 || std::strcmp(env, "HIGH_FIRST") == 0) {
        high_first = true;
      }
    }
    torch::Tensor nibbles;
    if (high_first) {
      nibbles = torch::stack({high, low}, -1).reshape({pw.size(0), -1}).to(torch::kFloat32);
    } else {
      nibbles = torch::stack({low, high}, -1).reshape({pw.size(0), -1}).to(torch::kFloat32);
    }
    auto w = (nibbles - zc) * sc;  // (out, in_aligned)
    auto xf = x.to(device, /*non_blocking=*/true).to(torch::kFloat32);
    // Trim columns to match input features if packed weight was aligned to a larger multiple
    const auto in_features = xf.size(-1);
    if (w.size(1) > in_features) {
      w = w.narrow(/*dim=*/1, /*start=*/0, /*length=*/in_features);
    }
    return xf.matmul(w.t());  // (B, out)
  });
}


