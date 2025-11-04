import argparse
from pathlib import Path

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore

from omnicoder.modeling.transformer_moe import OmniTransformer


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch dynamic quantization (Linear layers) for OmniTransformer")
    ap.add_argument("--ckpt", type=str, default="", help="Optional: path to state_dict .pt to load before quantizing")
    ap.add_argument("--out", type=str, default="weights/omnicoder_int8.pt", help="Output path for quantized state_dict")
    args = ap.parse_args()

    model = OmniTransformer()
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()

    # Dynamic quantization for Linear layers (CPU inference)
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _safe_save(qmodel.state_dict(), args.out)
    print(f"Saved dynamically quantized (int8) state dict to {args.out}")


if __name__ == "__main__":
    main()


