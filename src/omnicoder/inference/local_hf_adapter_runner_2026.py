from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


class RunnerError(RuntimeError):
    pass


def missing_deps() -> list[str]:
    return [name for name in ("torch", "transformers", "peft") if importlib.util.find_spec(name) is None]


def prompt_from_request(request: dict[str, Any]) -> str:
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    task = request.get("task")
    if task not in (None, "", [], {}):
        return json.dumps(task, ensure_ascii=True, sort_keys=True)
    raise RunnerError("request did not include prompt or task")


def load_request(stdin_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"stdin was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerError("stdin JSON must be an object")
    return payload


def run_model(args: argparse.Namespace, request: dict[str, Any]) -> dict[str, Any]:
    deps = missing_deps()
    if deps:
        raise RunnerError(f"missing local HF adapter runner dependencies: {', '.join(deps)}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "auto": "auto"}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": args.device_map}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        adapter = Path(args.adapter)
        if not adapter.exists():
            raise RunnerError(f"adapter path does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    prompt = prompt_from_request(request)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_input_tokens))
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(request.get("max_output_tokens") or args.max_new_tokens),
            temperature=float(request.get("temperature") or args.temperature),
            do_sample=float(request.get("temperature") or args.temperature) > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {"prediction": text.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint-runner adapter for local HF/PEFT artifacts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--device-map", "--device_map", dest="device_map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--max-input-tokens", "--max_input_tokens", dest="max_input_tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = load_request(sys.stdin.read())
        print(json.dumps(run_model(args, request), ensure_ascii=True, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=True, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
