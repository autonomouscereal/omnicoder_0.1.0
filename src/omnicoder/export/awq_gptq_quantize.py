import argparse
import sys
from pathlib import Path


def _quantize_awq(hf_model_id: str, out_dir: Path) -> bool:
    try:
        from awq import AutoAWQForCausalLM, AWQConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        print("AutoAWQ not installed. Install with: pip install autoawq transformers")
        return False
    try:
        tok = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
        model = AutoAWQForCausalLM.from_pretrained(hf_model_id, low_cpu_mem_usage=True)
        awq_config = AWQConfig(bits=4, group_size=128)
        model.quantize(tokenizer=tok, quant_config=awq_config)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_quantized(out_dir)
        print(f"Saved AWQ int4 model to {out_dir}")
        return True
    except Exception as e:
        print(f"AWQ quantization failed: {e}")
        return False


def _quantize_gptq(hf_model_id: str, out_dir: Path) -> bool:
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        print("AutoGPTQ not installed. Install with: pip install auto-gptq transformers")
        return False
    try:
        tok = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
        quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
        model = AutoGPTQForCausalLM.from_pretrained(hf_model_id, quantize_config)
        # Simple calibration on tokenizer vocab subset (toy). For best results, pass a dataset.
        calib_texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
        enc = [tok.encode(t, add_special_tokens=False)[:128] for t in calib_texts]
        model.quantize(encodings=enc)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_quantized(out_dir)
        print(f"Saved GPTQ int4 model to {out_dir}")
        return True
    except Exception as e:
        print(f"GPTQ quantization failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="AWQ/GPTQ quantization helper")
    ap.add_argument('--hf_model', type=str, help='HuggingFace model id (preferred)')
    ap.add_argument('--out', type=str, required=True, help='Output directory for quantized model')
    ap.add_argument('--method', type=str, default='awq', choices=['awq','gptq'])
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not args.hf_model:
        print("Please provide --hf_model (e.g., meta-llama/Llama-2-7b-hf). For local .pt/.bin checkpoints, integrate your loader.")
        sys.exit(1)

    if args.method == 'awq':
        ok = _quantize_awq(args.hf_model, out_dir)
    else:
        ok = _quantize_gptq(args.hf_model, out_dir)
    if not ok:
        sys.exit(2)


if __name__ == '__main__':
    main()


