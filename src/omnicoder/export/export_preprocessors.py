from __future__ import annotations

import argparse
from pathlib import Path

import torch

from omnicoder.modeling.multimodal.aligner import PreAligner


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PreAligner modality heads as ONNX preprocessors")
    ap.add_argument("--prealign_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ck = torch.load(args.prealign_ckpt, map_location='cpu', weights_only=True)
    ed = int(ck.get("embed_dim", 256))
    aligner = PreAligner(embed_dim=ed, text_dim=ed, image_dim=768, audio_dim=512, video_dim=768)
    aligner.load_state_dict(ck["aligner"])  # type: ignore[index]
    aligner.eval().to(args.device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Create small wrapper modules to export each head separately
    class Head(torch.nn.Module):
        def __init__(self, seq: torch.nn.Sequential):
            super().__init__()
            self.seq = seq
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return torch.nn.functional.normalize(self.seq(x), dim=-1)

    heads = {
        "text": Head(aligner.text_proj),
        "image": Head(aligner.image_proj),
        "audio": Head(aligner.audio_proj),
        "video": Head(aligner.video_proj),
    }
    d_in = {"text": ed, "image": 768, "audio": 512, "video": 768}

    for name, mod in heads.items():
        onnx_path = out / f"pre_{name}_proj.onnx"
        dummy = torch.randn(1, d_in[name], device=args.device)
        torch.onnx.export(
            mod.to(args.device),
            (dummy,),
            onnx_path.as_posix(),
            input_names=[f"{name}_in"],
            output_names=[f"{name}_emb"],
            opset_version=int(args.opset),
            dynamic_axes={f"{name}_in": {0: "batch"}, f"{name}_emb": {0: "batch"}},
        )
        print(f"Exported {name} head to {onnx_path}")


if __name__ == "__main__":
    main()


