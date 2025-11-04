from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.training.simple_tokenizer import get_text_tokenizer
from omnicoder.config import get_mobile_preset


class QADataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: List[str]) -> None:
        self.rows: List[Tuple[str, str]] = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            for ln in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                if not ln.strip():
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                q = str(obj.get('prompt') or obj.get('question') or '').strip()
                a = str(obj.get('answer') or obj.get('label') or '').strip()
                if q and a:
                    self.rows.append((q, a))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q, a = self.rows[idx]
        tok = get_text_tokenizer(prefer_hf=True)
        prompt = f"Q: {q}\nA:"
        inp = torch.tensor(tok.encode(prompt), dtype=torch.long)
        tgt = torch.tensor(tok.encode(a), dtype=torch.long)
        return inp, tgt


def build_model(preset_name: str, seq_len_hint: int) -> OmniTransformer:
    preset = get_mobile_preset(preset_name)
    return OmniTransformer(
        vocab_size=preset.vocab_size,
        n_layers=preset.n_layers,
        d_model=preset.d_model,
        n_heads=preset.n_heads,
        mlp_dim=preset.mlp_dim,
        n_experts=preset.moe_experts,
        top_k=preset.moe_top_k,
        max_seq_len=max(preset.max_seq_len, seq_len_hint),
        use_rope=True,
        kv_latent_dim=preset.kv_latent_dim,
        multi_query=preset.multi_query,
        multi_token=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Text QA finetune over benchmark JSONLs")
    ap.add_argument("--jsonl_list", type=str, nargs='+', required=True, help="One or more JSONL files with {prompt/question, answer}")
    ap.add_argument("--mobile_preset", type=str, default="mobile_4gb")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", type=str, default="weights/omnicoder_qa_finetune.pt")
    args = ap.parse_args()

    ds = QADataset(args.jsonl_list)
    if len(ds) == 0:
        raise SystemExit("qa_finetune: dataset empty")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = build_model(args.mobile_preset, seq_len_hint=2048)
    try:
        from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
        loaded = load_best_or_latest(model, args.out)
        if loaded is not None:
            print(f"[resume] loaded {loaded}")
    except Exception:
        pass
    model.to(args.device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    tok = get_text_tokenizer(prefer_hf=True)
    step = 0
    for inp_ids, tgt_ids in dl:
        inp_ids = inp_ids.to(args.device)
        tgt_ids = tgt_ids.to(args.device)
        out = model(inp_ids)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        # Align labels to next-token of prompt+answer length
        bsz, t, vocab = logits.shape
        ans_len = tgt_ids.size(1)
        ans_logits = logits[:, t - ans_len - 1 : t - 1, :]
        loss = ce(ans_logits.reshape(-1, vocab), tgt_ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 20 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f}")
        if step >= args.steps:
            break
    try:
        from omnicoder.utils.checkpoint import save_with_sidecar  # type: ignore
    except Exception:
        save_with_sidecar = None  # type: ignore
    meta = {'train_args': {'steps': int(args.steps)}}
    if callable(save_with_sidecar):
        final = save_with_sidecar(args.out, model.state_dict(), meta=meta)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _safe_save({k: (v.detach().to('cpu') if torch.is_tensor(v) else v) for k, v in model.state_dict().items()}, args.out)
        final = args.out
    # Micro-eval: compute QA CE on one sample and update best if improved
    try:
        from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        if len(ds) > 0:
            _inp, _tgt = ds[0]
            _inp = _inp.unsqueeze(0).to(args.device)
            _tgt = _tgt.unsqueeze(0).to(args.device)
            with torch.inference_mode():
                _out = model(_inp)
                _logits = _out if isinstance(_out, torch.Tensor) else _out[0]
            _bsz, _t, _v = _logits.shape
            _alen = _tgt.size(1)
            _ans_logits = _logits[:, _t - _alen - 1 : _t - 1, :]
            _ce = nn.CrossEntropyLoss()(_ans_logits.reshape(-1, _v), _tgt.reshape(-1)).item()
            maybe_save_best(args.out, model, 'qa_ce', float(_ce), higher_is_better=False)
    except Exception:
        pass
    print(f"Saved QA finetune checkpoint to {final}")


if __name__ == "__main__":
    main()


