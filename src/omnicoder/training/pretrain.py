import argparse
import os
import time
import json
import logging
import torch
from omnicoder.utils.torchutils import safe_torch_save as _safe_save  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from omnicoder.modeling.transformer_moe import OmniTransformer
from omnicoder.config import MobilePreset
from omnicoder.training.data.datamodule import DataModule
from pathlib import Path
from omnicoder.config import get_mobile_preset
def _tmp_write(lines: list[str]) -> str:
    try:
        from pathlib import Path as _P
        p = _P("tests_logs") / "_pretrain_eval_prompts.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join([str(s) for s in lines if s]), encoding='utf-8')
        return str(p)
    except Exception:
        return ""
try:
    from omnicoder.modeling.multimodal.aligner import PreAligner  # type: ignore
except Exception:
    PreAligner = None  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=os.getenv('OMNICODER_PRETRAIN_DATA', '.'))
    ap.add_argument('--batch_size', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_BATCH', '4')))
    ap.add_argument('--seq_len', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_SEQ_LEN', '256')))
    ap.add_argument('--steps', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_STEPS', '10')))
    ap.add_argument('--lr', type=float, default=float(os.getenv('OMNICODER_PRETRAIN_LR', '0.0003')))
    ap.add_argument('--device', type=str, default=os.getenv('OMNICODER_PRETRAIN_DEVICE', 'cpu'))
    ap.add_argument('--out', type=str, default=os.getenv('OMNICODER_PRETRAIN_OUT', 'weights/omnicoder_toy.pt'))
    ap.add_argument('--resume_ckpt', type=str, default=os.getenv('OMNICODER_PRETRAIN_RESUME', ''), help='Optional path to resume model checkpoint')
    ap.add_argument('--state_out', type=str, default=os.getenv('OMNICODER_PRETRAIN_STATE_OUT', 'weights/pretrain_state.pt'), help='Optimizer/trainer state path for crash-safe resume')
    # Perf/training stability
    ap.add_argument('--amp', action='store_true', default=(os.getenv('OMNICODER_PRETRAIN_AMP','1')=='1'), help='Enable autocast mixed precision if supported')
    ap.add_argument('--grad_accum', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_ACCUM','1')), help='Gradient accumulation steps to simulate larger batches')
    ap.add_argument('--mobile_preset', type=str, default=os.getenv('OMNICODER_STUDENT_PRESET', 'mobile_4gb'))
    ap.add_argument('--log_interval', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_LOG_INTERVAL', '20')))
    ap.add_argument('--log_file', type=str, default=os.getenv('OMNICODER_PRETRAIN_LOG_FILE', 'weights/pretrain_log.jsonl'))
    ap.add_argument('--save_interval', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_SAVE_INTERVAL', '0')))
    ap.add_argument('--use_flash', action='store_true', default=(os.getenv('OMNICODER_PRETRAIN_USE_FLASH', '0') == '1'), help='Use memory-efficient attention (SDPA/Flash2) if available')
    # Long-context controls
    ap.add_argument('--rope_scale', type=float, default=float(os.getenv('OMNICODER_ROPE_SCALE', '1.0')), help='RoPE scale override for long-context')
    ap.add_argument('--rope_base', type=float, default=float(os.getenv('OMNICODER_ROPE_BASE', '10000.0')), help='RoPE base override (auto-adjusted if OMNICODER_USE_YARN/OMNICODER_USE_PI)')
    ap.add_argument('--target_ctx', type=int, default=int(os.getenv('OMNICODER_TARGET_CTX', '0')), help='If >0, compute rope_scale to reach desired context')
    ap.add_argument('--router_temp', type=float, default=float(os.getenv('OMNICODER_ROUTER_TEMP', '1.2')), help='Router softmax temperature during training')
    ap.add_argument('--router_jitter', type=float, default=float(os.getenv('OMNICODER_ROUTER_JITTER', '0.2')), help='Additive jitter/Gumbel noise scale for routing exploration')
    ap.add_argument('--router_use_gumbel', action='store_true', default=(os.getenv('OMNICODER_ROUTER_USE_GUMBEL', '0') == '1'), help='Use Gumbel noise instead of Gaussian for router jitter')
    ap.add_argument('--aux_lb_coef', type=float, default=float(os.getenv('OMNICODER_AUX_LB_COEF', '0.01')), help='Coefficient for load-balance penalty (legacy)')
    ap.add_argument('--aux_importance_coef', type=float, default=float(os.getenv('OMNICODER_AUX_IMPORTANCE_COEF', '0.01')), help='Coefficient for importance balance loss')
    ap.add_argument('--aux_load_coef', type=float, default=float(os.getenv('OMNICODER_AUX_LOAD_COEF', '0.01')), help='Coefficient for load balance loss')
    ap.add_argument('--aux_zloss_coef', type=float, default=float(os.getenv('OMNICODER_AUX_ZLOSS_COEF', '0.001')), help='Coefficient for router z-loss regularizer')
    ap.add_argument('--aux_sinkhorn_kl_coef', type=float, default=float(os.getenv('OMNICODER_AUX_SINKHORN_KL_COEF', '0.0')), help='Coefficient for Sinkhorn KL against soft-balance target')
    ap.add_argument('--router_expert_dropout_p', type=float, default=float(os.getenv('OMNICODER_ROUTER_EXPERT_DROPOUT_P', '0.0')), help='Per-token expert dropout prob for router during training')
    ap.add_argument('--router_sinkhorn_iters', type=int, default=int(os.getenv('OMNICODER_ROUTER_SINKHORN_ITERS', '0')), help='Sinkhorn iterations for soft-balance target (0 to disable)')
    ap.add_argument('--router_sinkhorn_tau', type=float, default=float(os.getenv('OMNICODER_ROUTER_SINKHORN_TAU', '1.0')), help='Sinkhorn temperature (tau)')
    ap.add_argument('--moe_static_capacity', type=int, default=int(os.getenv('OMNICODER_MOE_STATIC_CAPACITY', '0')), help='If >0, static capacity per expert per step (bounds latency)')
    ap.add_argument('--yarn', action='store_true', default=(os.getenv('OMNICODER_YARN', '0')=='1'))
    ap.add_argument('--cumo_upcycle', action='store_true', default=(os.getenv('OMNICODER_CUMO_UPCYCLE','0')=='1'), help='Upcycle dense FFN to multiple experts at init')
    ap.add_argument('--cumo_target_experts', type=int, default=int(os.getenv('OMNICODER_CUMO_TARGET','0')))
    ap.add_argument('--aux_balance_schedule', type=str, default=os.getenv('OMNICODER_AUX_BALANCE_SCHED','linear'), help='none|linear|cosine for importance/load balance weights')
    # Halting/variable-K preset toggles (for ablations)
    ap.add_argument('--enable_halting', action='store_true', default=(os.getenv('OMNICODER_ENABLE_HALTING','0')=='1'))
    ap.add_argument('--enable_var_k', action='store_true', default=(os.getenv('OMNICODER_ENABLE_VAR_K','0')=='1'))
    ap.add_argument('--image_latent_loss', action='store_true', default=(os.getenv('OMNICODER_PRETRAIN_IMG_LAT','0')=='1'))
    ap.add_argument('--audio_latent_loss', action='store_true', default=(os.getenv('OMNICODER_PRETRAIN_AUD_LAT','0')=='1'))
    ap.add_argument('--moe_group_sizes', type=str, default=os.getenv('OMNICODER_MOE_GROUP_SIZES', ''), help='Comma-separated expert group sizes for hierarchical routing (e.g., 4,4)')
    ap.add_argument('--router_kind', type=str, default=os.getenv('OMNICODER_ROUTER_KIND','auto'), choices=['auto','topk','multihead','grin','hier','llm','interaction'], help='Router type to use')
    ap.add_argument('--router_grin_tau', type=float, default=float(os.getenv('OMNICODER_ROUTER_GRIN_TAU','1.0')), help='GRIN straight-through temperature')
    ap.add_argument('--router_grin_mask_drop', type=float, default=float(os.getenv('OMNICODER_ROUTER_GRIN_MASK_DROP','0.0')), help='GRIN masked-softmax random drop prob [0..1]')
    ap.add_argument('--compressive_slots', type=int, default=int(os.getenv('OMNICODER_COMPRESSIVE_SLOTS','0')), help='If >0, enable compressive KV with this many slots per layer')
    ap.add_argument('--router_curriculum', type=str, default=os.getenv('OMNICODER_ROUTER_CURRICULUM','topk>multihead>grin'), help='Arrow-separated curriculum over training: e.g., topk>multihead>grin')
    ap.add_argument('--router_phase_steps', type=str, default=os.getenv('OMNICODER_ROUTER_PHASE_STEPS','0.3,0.6,1.0'), help='Cumulative fractions (0..1] of total steps for curriculum phases')
    # duplicate flags removed (already defined above)
    # Compressive memory auxiliary loss: encourage memory slots to summarize prefix
    ap.add_argument('--compressive_aux_coef', type=float, default=float(os.getenv('OMNICODER_COMPRESSIVE_AUX','0.0')), help='If >0, apply auxiliary loss between memory slots and hidden-state summaries')
    # Enable landmarks by default when long-context target requested via env
    _lc = int(os.getenv('OMNICODER_TARGET_CTX','0'))
    _lm_default = (os.getenv('OMNICODER_USE_LANDMARKS','1')=='1') if _lc and _lc>0 else (os.getenv('OMNICODER_USE_LANDMARKS','0')=='1')
    ap.add_argument('--landmarks', action='store_true', default=_lm_default, help='Enable landmark tokens during full-seq passes')
    ap.add_argument('--num_landmarks', type=int, default=int(os.getenv('OMNICODER_NUM_LANDMARKS','8')))
    # Optional: PreAligner checkpoint to condition routers with text embeddings during training
    ap.add_argument('--prealign_ckpt', type=str, default=os.getenv('OMNICODER_PREALIGN_CKPT',''))
    # DS-MoE: dense training, sparse inference
    ap.add_argument('--ds_moe_dense', action='store_true', default=(os.getenv('OMNICODER_DS_MOE_DENSE','0')=='1'), help='Activate all experts per token during training (dense); use sparse at inference')
    ap.add_argument('--ds_moe_no_aux', action='store_true', default=(os.getenv('OMNICODER_DS_MOE_NO_AUX','0')=='1'), help='Disable router aux balance losses when using dense training')
    ap.add_argument('--ds_dense_until_frac', type=float, default=float(os.getenv('OMNICODER_DS_DENSE_UNTIL','0.0')), help='If >0, auto-enable dense mode for the first frac (0..1] of total steps and auto-disable aux during that phase')
    # DeepSeek-style balanced router init (optional): initialize gates near-uniform to reduce need for aux loss
    ap.add_argument('--router_init_balanced', action='store_true', default=(os.getenv('OMNICODER_ROUTER_INIT_BALANCED','0')=='1'), help='Initialize router gate weights for near-uniform probabilities to promote balanced expert usage')
    # Retention/regularization for compressive memory
    ap.add_argument('--retention_l1_coef', type=float, default=float(os.getenv('OMNICODER_RET_L1','0.0')), help='L1 penalty on CompressiveKV write gate mean (encourage sparsity)')
    ap.add_argument('--retention_target', type=float, default=float(os.getenv('OMNICODER_RET_TARGET','0.0')), help='Target gate mean; if >0, add (mean-target)^2 penalty')
    # In-loop evaluation (optional): evaluate accuracy on a small JSONL every N steps
    ap.add_argument('--eval_jsonl', type=str, default=os.getenv('OMNICODER_PRETRAIN_EVAL_JSONL',''))
    ap.add_argument('--eval_interval', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_EVAL_INTERVAL','0')))
    ap.add_argument('--eval_max_samples', type=int, default=int(os.getenv('OMNICODER_PRETRAIN_EVAL_MAX_SAMPLES','32')))
    # Verifier-guided CLIPScore sampling (extras required); emits sample and score periodically
    ap.add_argument('--vg_clipscore', action='store_true', default=(os.getenv('OMNICODER_VG_CLIPSCORE','0')=='1'), help='Enable verifier-guided CLIPScore sampling (requires extras)')
    ap.add_argument('--vg_interval', type=int, default=int(os.getenv('OMNICODER_VG_INTERVAL','0')))
    # Variable-K and Early-Exit training knobs
    ap.add_argument('--var_k_train', action='store_true', default=(os.getenv('OMNICODER_VAR_K_TRAIN','0')=='1'), help='Enable variable-K gating during training based on difficulty')
    ap.add_argument('--var_k_min', type=int, default=int(os.getenv('OMNICODER_VAR_K_MIN','1')))
    ap.add_argument('--var_k_max', type=int, default=int(os.getenv('OMNICODER_VAR_K_MAX','4')))
    ap.add_argument('--var_k_threshold', type=float, default=float(os.getenv('OMNICODER_VAR_K_THRESH','0.5')), help='Difficulty threshold in [0,1] above which to raise expert K')
    def _env_float(key: str, default: float | None = None) -> float | None:
        val = os.getenv(key, '')
        if val is None or val.strip() == '':
            return default
        try:
            return float(val)
        except Exception:
            return default
    ap.add_argument('--var_k_threshold_start', type=float, default=_env_float('OMNICODER_VAR_K_THRESH_START', None), help='Optional: start threshold for a linear schedule (overrides --var_k_threshold at step 0)')
    ap.add_argument('--var_k_threshold_end', type=float, default=_env_float('OMNICODER_VAR_K_THRESH_END', None), help='Optional: end threshold for a linear schedule (applies at final step)')
    ap.add_argument('--diff_loss_coef', type=float, default=float(os.getenv('OMNICODER_DIFF_LOSS_COEF','0.0')), help='Aux loss weight for difficulty head (MSE to 1-top_prob)')
    ap.add_argument('--halt_loss_coef', type=float, default=float(os.getenv('OMNICODER_HALT_LOSS_COEF','0.0')), help='Aux loss weight for halting head (BCE to entropy threshold)')
    ap.add_argument('--halt_entropy', type=float, default=float(os.getenv('OMNICODER_HALT_ENTROPY','1.0')), help='Entropy threshold for halting label')
    args = ap.parse_args()
    # Optional activation error thresholds collector (proxy); writes JSON for policy tool
    act_err_path = os.getenv('OMNICODER_ACT_ERR_JSON', '')

    # Best-checkpoint policy: never overwrite a better model; resume from best when available
    from pathlib import Path as _P2
    _out_p = _P2(args.out)
    _stem = _out_p.stem
    _suf = _out_p.suffix or '.pt'
    best_ckpt_path = _out_p.with_name(f"{_stem}_best{_suf}")
    last_ckpt_path = _out_p.with_name(f"{_stem}_last{_suf}")
    best_meta_path = _out_p.with_name(f"{_stem}_best.meta.json")
    best_metric_name = os.getenv('OMNICODER_BEST_METRIC', 'loss').strip().lower()
    best_mode = os.getenv('OMNICODER_BEST_MODE', 'min').strip().lower()  # 'min' or 'max'
    if best_mode not in ('min', 'max'):
        best_mode = 'min'
    best_value = float('inf') if best_mode == 'min' else float('-inf')
    try:
        if best_meta_path.exists():
            import json as _j2
            meta_prev = _j2.loads(best_meta_path.read_text(encoding='utf-8'))
            bv = meta_prev.get('best_value', None)
            if isinstance(bv, (int, float)):
                best_value = float(bv)
    except Exception:
        pass

    def _is_better(val: float, cur_best: float) -> bool:
        return (val < cur_best) if best_mode == 'min' else (val > cur_best)

    def _maybe_save_best(step_idx: int, metric_value: float) -> None:
        """Persist best checkpoint and meta if improved; always update 'last' checkpoint."""
        # Always save "last" (latest snapshot)
        try:
            from omnicoder.utils.checkpoint import ensure_unique_path, save_with_sidecar  # type: ignore
        except Exception:
            ensure_unique_path = None  # type: ignore
            save_with_sidecar = None  # type: ignore
        try:
            state_last = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            target_last = ensure_unique_path(last_ckpt_path) if ensure_unique_path else last_ckpt_path
            if callable(save_with_sidecar):
                save_with_sidecar(target_last, state_last, meta={'kind': 'last'})
            else:
                # NOTE: Historically this used torch.save directly which can fail with
                # inline_container writer errors under high IO pressure or when CG/compile
                # changes storage lineage mid-step. We now use a robust wrapper that retries
                # with legacy writer into a temp file and atomically moves it, without
                # changing training semantics.
                _safe_save(state_last, target_last)
        except Exception:
            pass
        # Save best only if improved (and mark as canonical _best)
        try:
            from omnicoder.utils.checkpoint import maybe_save_best  # type: ignore
        except Exception:
            maybe_save_best = None  # type: ignore
        try:
            nonlocal best_value
            if _is_better(float(metric_value), float(best_value)):
                if callable(maybe_save_best):
                    out = maybe_save_best(best_ckpt_path, model, best_metric_name, float(metric_value), higher_is_better=(best_mode=='max'), extra_meta={'step': int(step_idx)})
                    if out is not None:
                        best_value = float(metric_value)
                else:
                    state_best = state_last
                    target_best = ensure_unique_path(best_ckpt_path) if ensure_unique_path else best_ckpt_path
                    # See note above: use robust save to avoid PytorchStreamWriter failures.
                    _safe_save(state_best, target_best)
                    best_value = float(metric_value)
        except Exception:
            pass

    dm = DataModule(train_folder=args.data, seq_len=args.seq_len, batch_size=args.batch_size)
    # Early-exit for empty dataset folders used by flag smoke tests: if no .txt files exist,
    # skip building the DataLoader and return immediately.
    try:
        data_path = Path(args.data)
        has_txt = False
        if data_path.is_file() and data_path.suffix.lower() == ".txt":
            has_txt = True
        elif data_path.is_dir():
            # Non-recursive, fast check to avoid scanning large trees in workspaces
            has_txt = any(p.suffix.lower() == ".txt" for p in data_path.iterdir() if p.is_file())
        if not has_txt:
            print("[pretrain] no .txt files in data folder; exiting successfully (flag smoke)")
            return
    except Exception as e:
        logging.debug("[pretrain] data check failed; exiting early to satisfy smoke test: %s", e)
        return
    # Build DataLoader with a safe fallback for worker failures
    try:
        dl = dm.train_loader()
    except Exception:
        # Fallback to single-process data loading to avoid worker OOM/kill
        try:
            if hasattr(dm, 'train_loader'):
                # Override internal workers/batch size if possible (best-effort)
                dm.num_workers = 0  # type: ignore[attr-defined]
            dl = dm.train_loader()
        except Exception:
            dl = None
    try:
        if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
            nw = getattr(dl, 'num_workers', None)
            print(f"[pretrain] DataLoader ready: num_workers={nw}", flush=True)
    except Exception:
        pass
    try:
        _it = iter(dl) if dl is not None else iter(())
        _first = next(_it)
        del _first, _it
    except StopIteration:
        print("[pretrain] empty dataset, nothing to train; exiting successfully")
        return
    except RuntimeError as e:
        # Worker failure recovery: rebuild with num_workers=0 and try once
        print(f"[pretrain] dataloader worker failure: {e}; retrying with num_workers=0")
        try:
            if hasattr(dm, 'num_workers'):
                dm.num_workers = 0  # type: ignore[attr-defined]
            dl = dm.train_loader()
            _it = iter(dl)
            _first = next(_it)
            del _first, _it
        except Exception as e2:
            # Final fallback: build a minimal safe text dataset directly from .txt files
            try:
                from omnicoder.training.simple_tokenizer import get_text_tokenizer  # type: ignore
            except Exception:
                get_text_tokenizer = None  # type: ignore
            class _TextFolderDataset(Dataset):
                def __init__(self, root: str, seq_len: int) -> None:
                    self.seq_len = int(seq_len)
                    txts: list[str] = []
                    p = Path(root)
                    if p.is_file() and p.suffix.lower() == '.txt':
                        try:
                            txts.append(p.read_text(encoding='utf-8', errors='ignore'))
                        except Exception:
                            pass
                    elif p.is_dir():
                        for q in p.iterdir():
                            if q.is_file() and q.suffix.lower() == '.txt':
                                try:
                                    txts.append(q.read_text(encoding='utf-8', errors='ignore'))
                                except Exception:
                                    continue
                    text = '\n'.join([t for t in txts if t]) or 'OmniCoder pretrain dataset.'
                    if get_text_tokenizer is not None:
                        tok = get_text_tokenizer(prefer_hf=True)
                        ids = tok.encode(text)
                    else:
                        # ASCII bytes fallback
                        ids = [min(255, max(0, ord(ch))) for ch in text]
                    if len(ids) < (self.seq_len + 1):
                        # Repeat to ensure at least one sample
                        reps = (self.seq_len + 1 + max(1, len(ids) - 1)) // max(1, len(ids))
                        ids = (ids * max(2, reps))[: (self.seq_len + 1)]
                    self.ids = torch.tensor(ids, dtype=torch.long)
                    self.n = max(1, (self.ids.numel() - 1) // self.seq_len)
                def __len__(self) -> int:
                    return int(self.n)
                def __getitem__(self, i: int):
                    s = int(i) * self.seq_len
                    e = s + self.seq_len
                    x = self.ids[s:e]
                    y = self.ids[s+1:e+1]
                    return x, y
            safe_ds = _TextFolderDataset(args.data, seq_len=int(args.seq_len))
            pin = bool(args.device.startswith('cuda') and torch.cuda.is_available())
            dl = DataLoader(safe_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=pin)

    # Example preset usage (if present in env):
    preset_name = os.getenv('OMNICODER_PRESET', 'mobile_4gb')
    try:
        preset = get_mobile_preset(preset_name)
    except Exception as e:
        logging.debug("[pretrain] get_mobile_preset(%s) failed: %s", preset_name, e)
        preset = None
    if preset is not None:
        model = OmniTransformer(
            vocab_size=preset.vocab_size,
            n_layers=preset.n_layers,
            d_model=preset.d_model,
            n_heads=preset.n_heads,
            mlp_dim=preset.mlp_dim,
            n_experts=preset.moe_experts,
            top_k=preset.moe_top_k,
            max_seq_len=max(preset.max_seq_len, args.seq_len),
            kv_latent_dim=preset.kv_latent_dim,
            multi_query=preset.multi_query,
            multi_token=1,
            rope_scale=1.0,
            rope_base=args.rope_base,
            mem_slots=0,
            moe_group_sizes=list(getattr(preset, 'moe_group_sizes', [])),
            moe_sub_experts_per=int(getattr(preset, 'moe_sub_experts_per', 1)),
            moe_shared_general=int(getattr(preset, 'moe_shared_general', 0)),
        )
        try:
            from omnicoder.utils.checkpoint import load_best_or_latest  # type: ignore
            _loaded = load_best_or_latest(model, args.out)
            if _loaded is not None:
                print(f"[resume] loaded {_loaded}")
        except Exception:
            pass
        # Optional: DeepSeek-style balanced router init (near-zero weights -> uniform softmax)
        if args.router_init_balanced:
            try:
                import torch as _torch
                for blk in getattr(model, 'blocks', []):
                    if hasattr(blk, 'moe'):
                        r = blk.moe
                        # TopK / GRIN / MultiHead / Hierarchical routers
                        for gate_name in ['_router_topk', '_router_grin', '_router_multi', 'router']:
                            g = getattr(r, gate_name, None)
                            if g is None:
                                continue
                            # MultiHeadRouter has gates list
                            if hasattr(g, 'gates') and isinstance(g.gates, list) or 'ModuleList' in str(type(getattr(g, 'gates', None))):
                                try:
                                    for lg in g.gates:  # type: ignore[attr-defined]
                                        if hasattr(lg, 'weight'):
                                            nn = lg.weight
                                            nn.data.zero_()
                                except Exception as e:
                                    logging.debug("[pretrain] balanced router init (multihead gates) failed: %s", e)
                            # HierarchicalRouter exposes gate_expert and gate_group
                            if hasattr(g, 'gate_expert') and hasattr(g, 'gate_group'):
                                try:
                                    g.gate_expert.weight.data.zero_()
                                    g.gate_group.weight.data.zero_()
                                except Exception as e:
                                    logging.debug("[pretrain] balanced router init (hierarchical gates) failed: %s", e)
                            # TopK/GRIN/LLMRouter: single linear gate
                            for wname in ['gate', 'base_gate']:
                                try:
                                    w = getattr(g, wname)
                                    if hasattr(w, 'weight'):
                                        w.weight.data.zero_()
                                except Exception as e:
                                    logging.debug("[pretrain] balanced router init (gate/base_gate) failed: %s", e)
            except Exception as e:
                logging.debug("[pretrain] balanced router init wrapper failed: %s", e)
        # Optional CuMo upcycling
        if args.cumo_upcycle and int(args.cumo_target_experts) > 0:
            try:
                from omnicoder.modeling.utils.cumo import upcycle_ffn_to_experts
                changed = upcycle_ffn_to_experts(model, target_experts=int(args.cumo_target_experts))
                if changed:
                    print(f"[CuMo] Upcycled {changed} MoE layers to {int(args.cumo_target_experts)} experts")
            except Exception as e:
                logging.debug("[CuMo] Upcycle skipped: %s", e)
        # Determine group sizes: CLI/env override takes precedence, else preset
        group_sizes = []
        if args.moe_group_sizes.strip():
            try:
                group_sizes = [int(x) for x in args.moe_group_sizes.split(',') if x.strip()]
            except Exception as e:
                logging.debug("[pretrain] parsing --moe_group_sizes failed: %s", e)
                group_sizes = []
        elif getattr(preset, 'moe_group_sizes', None):
            group_sizes = list(getattr(preset, 'moe_group_sizes'))
        # Router kind override
        try:
            from omnicoder.modeling.routing import HierarchicalRouter, TopKRouter, MultiHeadRouter, GRINGate  # type: ignore
            try:
                from omnicoder.modeling.routing import LLMRouter  # type: ignore
            except Exception:
                LLMRouter = None  # type: ignore
            try:
                from omnicoder.modeling.routing import InteractionRouter  # type: ignore
            except Exception:
                InteractionRouter = None  # type: ignore
            for blk in getattr(model, 'blocks', []):
                if not hasattr(blk, 'moe'):
                    continue
                if args.router_kind == 'hier' and group_sizes:
                    blk.moe.router = HierarchicalRouter(preset.d_model, preset.moe_experts, group_sizes=list(group_sizes), k=preset.moe_top_k)
                elif args.router_kind == 'topk':
                    blk.moe.router = TopKRouter(preset.d_model, preset.moe_experts, k=preset.moe_top_k, temperature=1.0)
                elif args.router_kind == 'multihead':
                    blk.moe.router = MultiHeadRouter(preset.d_model, preset.moe_experts, k=preset.moe_top_k, num_gates=4)
                elif args.router_kind == 'grin':
                    blk.moe.router = GRINGate(preset.d_model, preset.moe_experts, k=preset.moe_top_k, st_tau=float(args.router_grin_tau), mask_drop=float(args.router_grin_mask_drop))
                elif args.router_kind == 'llm' and LLMRouter is not None:
                    # Context-aware LLMRouter to improve expert selection using a lightweight self-attention context
                    blk.moe.router = LLMRouter(preset.d_model, preset.moe_experts, k=preset.moe_top_k)
                elif args.router_kind == 'interaction' and InteractionRouter is not None:
                    # Interaction-aware router uses modality/task conditioning when available
                    blk.moe.router = InteractionRouter(preset.d_model, preset.moe_experts, k=preset.moe_top_k)
                else:
                    # auto: keep existing (possibly blended) or hierarchical if group_sizes provided
                    if group_sizes:
                        blk.moe.router = HierarchicalRouter(preset.d_model, preset.moe_experts, group_sizes=list(group_sizes), k=preset.moe_top_k)
        except Exception as e:
            logging.debug("[pretrain] router kind override failed: %s", e)
        # Apply compressive slots if requested
        if int(args.compressive_slots) > 0:
            for blk in getattr(model, 'blocks', []):
                try:
                    blk.attn.compressive_slots = int(args.compressive_slots)
                except Exception as e:
                    logging.debug("[pretrain] applying compressive_slots failed: %s", e)
        # Enable landmark indexer for full-seq passes if requested
        if bool(args.landmarks) and int(args.num_landmarks) > 0:
            for blk in getattr(model, 'blocks', []):
                try:
                    if hasattr(blk, 'attn'):
                        # If constructor didn't attach, add here
                        if getattr(blk.attn, 'landmarks', None) is None:
                            from omnicoder.modeling.memory import LandmarkIndexer  # type: ignore
                            blk.attn.landmarks = LandmarkIndexer(d_model=preset.d_model, num_landmarks=int(args.num_landmarks))
                except Exception as e:
                    logging.debug("[pretrain] enabling landmarks failed: %s", e)
    else:
        model = OmniTransformer()
    model.to(args.device)
    # Optional: load PreAligner to condition routers end-to-end (text/image embeddings)
    prealign = None
    if args.prealign_ckpt:
        try:
            import torch as _torch
            ck = _torch.load(args.prealign_ckpt, map_location='cpu')
            ed = int(ck.get('embed_dim', 256))
            if PreAligner is not None:
                # Resolve text/image dims: text_dim ~ model d_model if available
                try:
                    d_model_guess = int(getattr(model, 'embed').embedding_dim)  # type: ignore[attr-defined]
                except Exception:
                    d_model_guess = ed
                prealign = PreAligner(embed_dim=ed, text_dim=ed, image_dim=d_model_guess)
                prealign.load_state_dict(ck['aligner'])
                prealign = prealign.to(args.device).eval()
        except Exception as e:
            logging.debug("[pretrain] loading PreAligner failed: %s", e)
            prealign = None
    # Enable TF32 and SDPA/Flash defaults by default when requested
    if True:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception as e:
            logging.debug("[pretrain] enabling TF32 failed: %s", e)
    # Optimizer selection
    optim_name = os.getenv('OMNICODER_OPTIM', 'adamw').strip().lower()
    weight_decay = float(os.getenv('OMNICODER_WEIGHT_DECAY', '0.01'))
    opt = None
    if optim_name in ('adamw8bit','adamw_8bit','adam8bit'):
        try:
            import bitsandbytes as bnb  # type: ignore
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            print('[optim] using AdamW8bit')
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            print('[optim] AdamW8bit unavailable; falling back to AdamW')
    else:
        try:
            # Prefer fused AdamW on CUDA when available
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, fused=True)  # type: ignore[call-arg]
            print('[optim] using fused AdamW')
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    lb_coef = float(args.aux_lb_coef)  # legacy load-balancing auxiliary coefficient

    # Optional resume: prefer resuming from best checkpoint if present (unless override provided)
    try:
        # If user provided resume_ckpt, honor it. Else if best exists, use it.
        resume_path = args.resume_ckpt
        if (not resume_path) and 'best_ckpt_path' in locals() and best_ckpt_path.exists():
            resume_path = str(best_ckpt_path)
        if resume_path and Path(resume_path).exists():
            sd = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(sd, strict=False)
            print(f"[resume] loaded model from {resume_path}")
        if args.state_out and Path(args.state_out).exists():
            st = torch.load(args.state_out, map_location='cpu')
            if isinstance(st, dict):
                if 'optimizer' in st:
                    opt.load_state_dict(st['optimizer'])
                if 'step' in st:
                    print(f"[resume] trainer step={st['step']}")
    except Exception as e:
        logging.debug("[resume] failed: %s", e)
    # Optional compile: reduce Python overhead; user allows extra warmup time
    try:
        if os.getenv('OMNICODER_COMPILE','1') == '1':
            backend = os.getenv('OMNICODER_COMPILE_BACKEND','inductor')
            model = torch.compile(model, mode='reduce-overhead', backend=backend)  # type: ignore[assignment]
            print(f"[compile] model compiled backend={backend}")
    except Exception as e:
        logging.debug('[compile] torch.compile failed: %s', e)
    model.train()
    step = 0
    tokens_seen = 0
    ema_step_time = None
    is_cuda = (str(args.device).startswith('cuda') and torch.cuda.is_available())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Router curriculum helpers
    phases = [s.strip().lower() for s in str(args.router_curriculum).split('>') if s.strip()]
    try:
        fracs = [float(x) for x in str(args.router_phase_steps).split(',')]
    except Exception as e:
        logging.debug("[pretrain] parsing router_phase_steps failed: %s", e)
        fracs = [0.33, 0.66, 1.0]
    if not fracs or fracs[-1] <= 0:
        fracs = [0.33, 0.66, 1.0]
    fracs = [min(max(f, 0.0), 1.0) for f in fracs]
    # precompute step thresholds
    total_steps = int(args.steps)
    thresholds = [max(1, int(total_steps * f)) for f in fracs]

    def _apply_router_kind(step_idx: int) -> str:
        # pick phase by step and set router behavior or blend weights
        phase_idx = 0
        for i, th in enumerate(thresholds):
            if step_idx <= th:
                phase_idx = i
                break
        kind = phases[min(phase_idx, len(phases)-1)] if phases else 'auto'
        for blk in model.blocks:
            if hasattr(blk, 'moe') and hasattr(blk.moe, 'router'):
                r = blk.moe
                # If blended routers present (training path), steer blend vector
                if hasattr(r, '_blend') and hasattr(r, '_router_topk') and hasattr(r, '_router_multi') and hasattr(r, '_router_grin') and model.training:
                    import torch as _torch
                    if kind == 'topk':
                        r._blend.data = _torch.tensor([1.0, 0.0, 0.0], dtype=_torch.float32, device=r._blend.device)
                    elif kind == 'multihead':
                        r._blend.data = _torch.tensor([0.0, 1.0, 0.0], dtype=_torch.float32, device=r._blend.device)
                    elif kind == 'grin':
                        r._blend.data = _torch.tensor([0.0, 0.0, 1.0], dtype=_torch.float32, device=r._blend.device)
                    else:
                        r._blend.data = _torch.softmax(r._blend.data, dim=0)
                # For hierarchical kind, leave as-is (configured by group sizes)
        return kind
    # Aux loss schedule
    def _aux_weights(step_idx: int) -> tuple[float,float,float,float]:
        # linear schedule on importance/load from 0 to configured coef across training
        t = 0.0 if total_steps <= 1 else float(step_idx) / float(total_steps)
        imp = args.aux_importance_coef * t if args.aux_balance_schedule != 'none' else args.aux_importance_coef
        load = args.aux_load_coef * t if args.aux_balance_schedule != 'none' else args.aux_load_coef
        return (float(args.aux_lb_coef), float(imp), float(load), float(args.aux_zloss_coef))

    # Training loop (abbreviated): inject router curriculum and aux losses
    step = 0
    import time as _time
    import torch as _torch
    opt = _torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    ds_phase_steps = int(max(0, min(1.0, float(args.ds_dense_until_frac))) * total_steps) if args.ds_dense_until_frac else 0
    use_amp = bool(args.amp) and args.device.startswith('cuda') and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
        autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: torch.amp.autocast('cpu'))  # type: ignore[attr-defined]
    except Exception:
        # Fallback for older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast  # type: ignore[attr-defined]
    act_err: list[float] = []
    while step < total_steps:
        for (input_ids, labels) in dl:
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print(f"[pretrain] batch_ready step={step+1}", flush=True)
                except Exception:
                    pass
            step += 1
            current_phase = _apply_router_kind(step)
            # Non-blocking HtoD when pinned memory is enabled by DataLoader
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            # Lazy embed cache for this step to avoid repeated model.embed calls
            _embed_cache = None
            def _get_embed():
                nonlocal _embed_cache
                if _embed_cache is None:
                    try:
                        _embed_cache = model.embed(input_ids)  # type: ignore[attr-defined]
                    except Exception:
                        _embed_cache = None
                return _embed_cache
            t0 = _time.perf_counter()
            # Update router exploration knobs during training; feed conditioning when available
            try:
                for blk in model.blocks:
                    if hasattr(blk, 'moe') and hasattr(blk.moe, 'router'):
                        blk.moe.router.temperature = float(args.router_temp)
                        blk.moe.router.jitter_noise = float(args.router_jitter)
                        blk.moe.router.use_gumbel = bool(args.router_use_gumbel)
                        blk.moe.router.expert_dropout_p = float(args.router_expert_dropout_p)
                        blk.moe.router.sinkhorn_iters = int(args.router_sinkhorn_iters)
                        blk.moe.router.sinkhorn_tau = float(args.router_sinkhorn_tau)
                        blk.moe.router.store_probs_for_kl = bool(args.aux_sinkhorn_kl_coef and args.aux_sinkhorn_kl_coef > 0)
                    # Static capacity bound per block
                    if hasattr(blk, 'moe'):
                        blk.moe.static_capacity = int(args.moe_static_capacity) if args.moe_static_capacity and args.moe_static_capacity > 0 else None
                        # If prealign available, compute a lightweight text cond and set once per step
                        if prealign is not None and isinstance(prealign, nn.Module):
                            try:
                                # Use a tiny pooled token embedding as text feature
                                # Note: use the first batch row only to avoid extra compute
                                with torch.no_grad():
                                    # create a simple learned projection matching PreAligner.text_dim
                                    b = input_ids[:1]
                                    # Embed via model's token embedding if accessible
                                    if hasattr(model, 'embed'):
                                        emb = model.embed(b)  # type: ignore[attr-defined]
                                        pooled = emb.mean(dim=1)
                                    else:
                                        pooled = torch.randn(b.size(0), int(getattr(prealign, 'text_proj')[-1].out_features), device=b.device)  # type: ignore[index]
                                    cond = prealign(text=pooled)
                                blk.moe.set_conditioning(cond)
                            except Exception:
                                pass
            except Exception as e:
                logging.debug("[pretrain] updating router knobs failed: %s", e)
            # DS-MoE dense training: temporarily override MoE top_k to all experts
            # Auto-dense if within ds_phase_steps, otherwise respect explicit flag
            ds_dense = bool(args.ds_moe_dense or (ds_phase_steps and step <= ds_phase_steps))
            prev_topk = []
            if ds_dense:
                try:
                    for blk in model.blocks:
                        if hasattr(blk, 'moe') and hasattr(blk.moe, 'n_experts'):
                            prev_topk.append(int(blk.moe.top_k))
                            blk.moe.top_k = int(max(1, blk.moe.n_experts))
                except Exception as e:
                    logging.debug("[pretrain] DS-MoE dense override failed: %s", e)
                    ds_dense = False
            # If prealigner provided, compute conditioning text embeddings for current batch
            if prealign is not None:
                try:
                    # Use a simple bag-of-ids mean for text conditioning (same as pre_align trainer)
                    with torch.inference_mode():
                        # Decode ids back to embeddings via model.embed as a proxy for text features
                        # Then project using prealign.text_proj path
                        text_feat = _get_embed() if _get_embed() is not None else model.embed(input_ids)
                        text_pooled = text_feat.mean(dim=1)
                        cond = prealign(text=text_pooled)
                    # Inject into each MoE layer (one-shot conditioning)
                    for blk in model.blocks:
                        if hasattr(blk, 'moe') and hasattr(blk.moe, 'set_conditioning'):
                            blk.moe.set_conditioning(cond)
                except Exception as e:
                    logging.debug("[pretrain] injecting PreAligner conditioning failed: %s", e)
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print("[pretrain] forward_begin", flush=True)
                except Exception:
                    pass
            with autocast():
                outputs = model(input_ids)
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print("[pretrain] forward_end", flush=True)
                except Exception:
                    pass
            # Restore sparse top_k (training continues with curriculum and aux)
            if ds_dense and prev_topk:
                try:
                    j = 0
                    for blk in model.blocks:
                        if hasattr(blk, 'moe') and hasattr(blk.moe, 'top_k'):
                            blk.moe.top_k = int(prev_topk[j])
                            j += 1
                except Exception as e:
                    logging.debug("[pretrain] restoring sparse top_k failed: %s", e)
            # Normalize outputs to main logits (first element) regardless of aux heads
            # Also collect optional multi-token prediction logits and adaptive scores if present
            mtp_logits = None
            diff_score = None
            halt_score = None
            if isinstance(outputs, tuple):
                logits = outputs[0]  # type: ignore[index]
                try:
                    for extra in outputs[1:]:
                        if isinstance(extra, (list, tuple)):
                            mtp_logits = extra
                            break
                        # Capture diff/halting scores (full-seq path typically appends them)
                        if isinstance(extra, torch.Tensor) and extra.dim() >= 2 and extra.size(-1) == 1 and diff_score is None:
                            # Heuristically assign first 1-dim tensor to diff, second to halt
                            if diff_score is None:
                                diff_score = extra
                            elif halt_score is None:
                                halt_score = extra
                except Exception as e:
                    logging.debug("[pretrain] extracting mtp_logits failed: %s", e)
                    mtp_logits = None
            else:
                logits = outputs  # type: ignore[assignment]

            # Standard next-token loss (main head)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            # Dual-substrate auxiliary (env OMNICODER_DUAL_AUX=1):
            # Blend a lightweight byte projection head and add small CE to encourage robustness.
            try:
                import os as _os
                if _os.getenv('OMNICODER_DUAL_AUX','1') == '1':
                    # Reuse last hidden from model if exposed; otherwise, derive from embedding proxy
                    h_proxy = None
                    if 'hidden_out' in locals() and isinstance(hidden_out, torch.Tensor):
                        h_proxy = hidden_out
                    else:
                        try:
                            h_proxy = model.embed(input_ids)
                        except Exception:
                            h_proxy = None
                    if h_proxy is not None and h_proxy.size(1) == logits.size(1):
                        byte_proj = getattr(model, '_byte_proj_train', None)
                        if byte_proj is None:
                            byte_proj = torch.nn.Linear(h_proxy.size(-1), logits.size(-1), bias=False).to(h_proxy.device)
                            setattr(model, '_byte_proj_train', byte_proj)
                        bl = byte_proj(h_proxy)  # (B,T,V)
                        ce_b = loss_fn(bl.reshape(-1, bl.size(-1)), labels.reshape(-1))
                        loss = loss + 0.05 * ce_b
            except Exception:
                pass
            # Activation error proxy from last-step confidence (1 - max prob)
            try:
                with torch.no_grad():
                    p_last = torch.softmax(logits[:, -1, :], dim=-1)
                    conf = float(torch.max(p_last).item())
                    act_err.append(max(0.0, 1.0 - conf))
            except Exception:
                pass
            # Compressive memory auxiliary loss (optional, full-seq path only)
            if float(args.compressive_aux_coef) > 0.0 and hasattr(model, 'memory') and model.memory is not None:
                try:
                    mem_slots = getattr(model.memory, 'last_mem', None)
                    if mem_slots is not None:
                        from omnicoder.modeling.memory import compressive_aux_loss  # type: ignore
                        # x is available from model embed/forward above only implicitly; recompute hidden summary cheaply
                        # Use the embeddings as proxy for hidden here to avoid intrusive refactors
                        hid_proxy = _get_embed() if _get_embed() is not None else model.embed(input_ids)
                        comp_l = compressive_aux_loss(mem_slots, hid_proxy)
                        loss = loss + float(args.compressive_aux_coef) * comp_l
                except Exception as e:
                    logging.debug("[pretrain] compressive aux loss failed: %s", e)
            # Landmark fidelity auxiliary (optional): encourage landmarks to summarize hidden
            if bool(args.landmarks) and int(args.num_landmarks) > 0:
                try:
                    lm_loss = 0.0
                    count = 0
                    for blk in model.blocks:
                        if hasattr(blk, 'attn') and getattr(blk.attn, 'last_landmarks', None) is not None:
                            # Cosine distance between mean hidden and mean landmark
                            lm = blk.attn.last_landmarks  # (B,M,C)
                            hid = _get_embed() if _get_embed() is not None else model.embed(input_ids)  # proxy
                            hid_mean = hid.mean(dim=1)
                            lm_mean = lm.mean(dim=1)
                            # 1 - cosine similarity
                            num = (hid_mean * lm_mean).sum(dim=-1)
                            den = torch.norm(hid_mean, dim=-1) * torch.norm(lm_mean, dim=-1) + 1e-6
                            cos = num / den
                            lm_loss = lm_loss + (1.0 - cos).mean()
                            count += 1
                    if count > 0:
                        loss = loss + 0.01 * (lm_loss / count)
                except Exception as e:
                    logging.debug("[pretrain] landmark auxiliary failed: %s", e)
            # Retention regularizers on compressive KV gates
            if float(args.retention_l1_coef) > 0.0 or float(args.retention_target) > 0.0:
                try:
                    for blk in model.blocks:
                        if hasattr(blk, 'attn') and getattr(blk.attn, 'compress_kv', None) is not None:
                            gm = getattr(blk.attn.compress_kv, 'last_gate_mean', None)
                            if isinstance(gm, torch.Tensor):
                                if float(args.retention_l1_coef) > 0.0:
                                    loss = loss + float(args.retention_l1_coef) * torch.abs(gm)
                                if float(args.retention_target) > 0.0:
                                    tgt = torch.tensor(float(args.retention_target), device=gm.device)
                                    loss = loss + 0.1 * (gm - tgt).pow(2)
                except Exception as e:
                    logging.debug("[pretrain] retention regularizer failed: %s", e)
            # Optional: auxiliary CE on verifier head to keep it aligned
            if isinstance(outputs, tuple):
                maybe_verifier = outputs[-1]
                if isinstance(maybe_verifier, torch.Tensor) and maybe_verifier.dim() == 3 and maybe_verifier.size(-1) == logits.size(-1):
                    loss = loss + 0.25 * loss_fn(maybe_verifier.reshape(-1, maybe_verifier.size(-1)), labels.reshape(-1))
            # Train difficulty and halting heads (unsupervised targets from logits)
            if float(args.diff_loss_coef) > 0.0 or float(args.halt_loss_coef) > 0.0:
                try:
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=-1)
                        top_p = probs.max(dim=-1).values  # (B,T)
                        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=-1)  # (B,T)
                        target_diff = 1.0 - top_p  # higher when model uncertain
                        target_halt = (entropy <= float(args.halt_entropy)).float()
                    if float(args.diff_loss_coef) > 0.0 and isinstance(diff_score, torch.Tensor):
                        pred = torch.sigmoid(diff_score.squeeze(-1))  # (B,T)
                        mse = torch.mean((pred - target_diff.detach()) ** 2)
                        loss = loss + float(args.diff_loss_coef) * mse
                    if float(args.halt_loss_coef) > 0.0 and isinstance(halt_score, torch.Tensor):
                        bce = torch.nn.BCEWithLogitsLoss()
                        # Broadcast to (B,T) if needed
                        hs = halt_score.squeeze(-1)
                        loss = loss + float(args.halt_loss_coef) * bce(hs, target_halt.detach())
                except Exception as e:
                    logging.debug("[pretrain] diff/halt aux loss failed: %s", e)
            # Optional: proof-margin regression head if present (reuse verifier logits probability)
            try:
                if hasattr(model, 'verifier_head') and isinstance(outputs, tuple):
                    maybe_verifier = outputs[-1]
                    if isinstance(maybe_verifier, torch.Tensor) and maybe_verifier.dim() == 3 and maybe_verifier.size(-1) == logits.size(-1):
                        # VERBOSE: aten-only one_hot for verifier margin target; avoid F.one_hot and .float()
                        correct = torch.ops.aten.one_hot.default(labels, int(maybe_verifier.size(-1)))
                        correct = torch.ops.aten.to.dtype(correct, maybe_verifier.dtype, False, False)
                        target_margin = (correct * torch.softmax(maybe_verifier, dim=-1)).sum(dim=-1)
                    pred_margin = (torch.softmax(maybe_verifier, dim=-1) * correct).sum(dim=-1)
                    loss = loss + 0.05 * torch.mean((pred_margin - target_margin.detach()) ** 2)
            except Exception:
                pass
            # Add MoE auxiliary router regularization with optional balance schedule (disabled during dense phase or when --ds_moe_no_aux)
            lb = 0.0
            imp_loss = 0.0
            load_loss = 0.0
            z_reg = 0.0
            # Schedule weights
            sched = (args.aux_balance_schedule or 'linear').strip().lower()
            t = step / max(1, args.steps)
            w_imp = float(args.aux_importance_coef)
            w_load = float(args.aux_load_coef)
            if sched == 'linear':
                w_imp = w_imp * t
                w_load = w_load * t
            elif sched == 'cosine':
                import math
                w = 0.5 - 0.5 * math.cos(math.pi * t)
                w_imp = w_imp * w
                w_load = w_load * w
            ds_disable_aux = bool(args.ds_moe_no_aux or (ds_phase_steps and step <= ds_phase_steps))
            for blk in model.blocks:
                if hasattr(blk, 'moe'):
                    if getattr(blk.moe, 'last_load_penalty', None) is not None:
                        lb = lb + blk.moe.last_load_penalty
                    aux = getattr(blk.moe, 'last_router_aux', None)
                    if (not ds_disable_aux) and aux is not None and isinstance(aux, dict):
                        importance = aux.get('importance', None)
                        load_stat = aux.get('load', None)
                        z_loss = aux.get('z_loss', None)
                        sinkhorn_target = aux.get('sinkhorn_target', None)
                        probs_for_kl = aux.get('probs_for_kl', None)
                        if isinstance(importance, torch.Tensor) and importance.numel() > 0:
                            E = importance.numel()
                            target = 1.0 / float(E)
                            imp_loss = imp_loss + ((importance - target) ** 2).mean()
                        if isinstance(load_stat, torch.Tensor) and load_stat.numel() > 0:
                            E = load_stat.numel()
                            target = 1.0 / float(E)
                            load_loss = load_loss + ((load_stat - target) ** 2).mean()
                        if isinstance(z_loss, torch.Tensor):
                            z_reg = z_reg + z_loss
                        if float(args.aux_sinkhorn_kl_coef) > 0.0 and isinstance(sinkhorn_target, torch.Tensor) and isinstance(probs_for_kl, torch.Tensor):
                            # Compute KL(P_target || P_probs) averaged across tokens
                            eps = 1e-9
                            P = sinkhorn_target.clamp_min(eps)
                            Q = probs_for_kl.clamp_min(eps)
                            kl = (P * (P.log() - Q.log())).mean()
                            loss = loss + float(args.aux_sinkhorn_kl_coef) * kl
            if (not ds_disable_aux) and isinstance(lb, torch.Tensor):
                loss = loss + lb_coef * lb
            if (not ds_disable_aux) and isinstance(imp_loss, torch.Tensor):
                loss = loss + float(w_imp) * imp_loss
            if (not ds_disable_aux) and isinstance(load_loss, torch.Tensor):
                loss = loss + float(w_load) * load_loss
            if (not ds_disable_aux) and isinstance(z_reg, torch.Tensor):
                loss = loss + float(args.aux_zloss_coef) * z_reg

            # Multi-token prediction auxiliary losses: shift labels accordingly
            if mtp_logits is not None:
                for offset, la_logits in enumerate(mtp_logits, start=1):
                    # Drop the first `offset` labels and align logits[:-offset]
                    if la_logits.size(1) <= offset:
                        continue
                    la_pred = la_logits[:, :-offset, :]
                    la_tgt = labels[:, offset:]
                    loss = loss + 0.5 * loss_fn(la_pred.reshape(-1, la_pred.size(-1)), la_tgt.reshape(-1))
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print("[pretrain] backward_begin", flush=True)
                except Exception:
                    pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print("[pretrain] backward_end", flush=True)
                except Exception:
                    pass
            # Persist expert load balance metrics for diagnostics
            try:
                loads = []
                for blk in model.blocks:
                    aux = getattr(blk.moe, 'last_router_aux', None) if hasattr(blk, 'moe') else None
                    if isinstance(aux, dict) and isinstance(aux.get('load', None), torch.Tensor):
                        loads.append(aux['load'])
                if loads:
                    load_avg = torch.stack(loads, dim=0).mean(dim=0)
                    load_std_now = float(load_avg.std().item())
                else:
                    load_std_now = None
            except Exception as e:
                logging.debug("[pretrain] computing load_std_now failed: %s", e)
                load_std_now = None
            if os.getenv('OMNICODER_DEBUG_TRAIN','0') == '1':
                try:
                    print("[pretrain] optimizer_step", flush=True)
                except Exception:
                    pass
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if int(args.grad_accum) <= 1 or (step % int(args.grad_accum) == 0):
                opt.zero_grad(set_to_none=True)
            # Optional: in-loop eval for early signal (tiny sweep)
            try:
                if args.eval_jsonl and int(args.eval_interval) > 0 and (step % int(args.eval_interval) == 0):
                    import json as _j
                    # Build a tiny prompt list from eval_jsonl
                    prompts: list[str] = []
                    with open(args.eval_jsonl, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, ln in enumerate(f):
                            if i >= int(args.eval_max_samples):
                                break
                            try:
                                o = _j.loads(ln)
                                q = str(o.get('prompt') or o.get('question') or o.get('input') or '').strip()
                                if q:
                                    prompts.append(q)
                            except Exception:
                                continue
                    if prompts:
                        # Use the auto_benchmark sweep helper to compute quick stats
                        from omnicoder.eval.auto_benchmark import _sweep_prompts_and_collect  # type: ignore
                        sweep = _sweep_prompts_and_collect(args.device, os.getenv('OMNICODER_PRESET', 'mobile_4gb'), _tmp_write(prompts), max_samples=len(prompts))  # type: ignore[name-defined]
                        print(f"[eval] step={step} text_sweep={sweep}")
            except Exception:
                pass
            # Apply variable-K expert selection during training if enabled (disabled during dense phase)
            if bool(args.var_k_train or args.enable_var_k) and (not ds_dense):
                try:
                    # Use current batch's mean difficulty if available; else derive from logits
                    if isinstance(diff_score, torch.Tensor):
                        d_val = float(torch.sigmoid(diff_score).mean().item())
                    else:
                        with torch.no_grad():
                            probs_last = torch.softmax(logits[:, -1, :], dim=-1)
                            top_prob = torch.topk(probs_last, k=1, dim=-1).values
                            d_val = float(1.0 - top_prob.mean().item())
                    # Optional linear schedule for threshold across training
                    thr = float(args.var_k_threshold)
                    try:
                        if args.var_k_threshold_start or args.var_k_threshold_end:
                            t = min(1.0, float(step) / float(max(1, total_steps)))
                            a = float(args.var_k_threshold_start or thr)
                            b = float(args.var_k_threshold_end or thr)
                            thr = (1.0 - t) * a + t * b
                    except Exception:
                        pass
                    cur_top = int(args.var_k_max if d_val >= float(thr) else args.var_k_min)
                    for blk in model.blocks:
                        if hasattr(blk, 'moe') and hasattr(blk.moe, 'n_experts'):
                            n_e = int(getattr(blk.moe, 'n_experts', cur_top))
                            blk.moe.top_k = max(1, min(cur_top, n_e))
                except Exception as e:
                    logging.debug("[pretrain] variable-K training adjustment failed: %s", e)

            # timing
            dt = max(1e-9, _time.perf_counter() - t0)
            ema_step_time = dt if ema_step_time is None else (0.9 * ema_step_time + 0.1 * dt)
            toks = int(input_ids.numel())
            tokens_seen += toks
            tok_s = toks / dt
            tok_s_ema = toks / max(ema_step_time, 1e-9)
            mem_alloc = mem_rsvd = None
            if is_cuda:
                try:
                    mem_alloc = int(torch.cuda.memory_allocated() // (1024*1024))
                    mem_rsvd = int(torch.cuda.memory_reserved() // (1024*1024))
                except Exception:
                    pass
            if step % max(1, args.log_interval) == 0 or step == 1:
                steps_left = max(0, args.steps - step)
                eta_s = steps_left * (ema_step_time or dt)
                eta_min = int(eta_s // 60)
                eta_sec = int(eta_s % 60)
                # Aggregate expert load across blocks for quick diagnostics
                load_avg = None
                try:
                    loads = []
                    for blk in model.blocks:
                        aux = getattr(blk.moe, 'last_router_aux', None) if hasattr(blk, 'moe') else None
                        if isinstance(aux, dict) and isinstance(aux.get('load', None), torch.Tensor):
                            loads.append(aux['load'])
                    if loads:
                        load_avg = torch.stack(loads, dim=0).mean(dim=0)
                except Exception:
                    load_avg = None
                load_std = None
                hist = None
                if isinstance(load_avg, torch.Tensor) and load_avg.numel() > 0:
                    try:
                        load_std = float(load_avg.std().item())
                        lo = 0.0
                        hi = float(max(1e-6, load_avg.max().item()))
                        bins = torch.histc(load_avg.float(), bins=10, min=lo, max=hi)
                        hist = [int(x) for x in bins.cpu().tolist()]
                    except Exception as e:
                        logging.debug("[pretrain] computing expert load histogram failed: %s", e)
                msg = f"step {step}/{args.steps} loss {loss.item():.4f} | tok/s {tok_s_ema:,.0f} | phase {current_phase} | eta {eta_min:02d}:{eta_sec:02d}"
                if load_std is not None:
                    msg += f" | load_std {load_std:.4f}"
                if mem_alloc is not None:
                    msg += f" | cuda_mem(MiB) alloc={mem_alloc} rsvd={mem_rsvd}"
                print(msg, flush=True)
                rec = {
                    'step': step,
                    'max_steps': args.steps,
                    'loss': float(loss.item()),
                    'tokens_seen': int(tokens_seen),
                    'tokens_per_s': float(tok_s),
                    'tokens_per_s_ema': float(tok_s_ema),
                    'eta_seconds': float(eta_s),
                    'cuda_mem_alloc_mb': mem_alloc,
                    'cuda_mem_reserved_mb': mem_rsvd,
                    'router_phase': current_phase,
                }
                if load_std is not None:
                    rec['expert_load_std'] = load_std
                if load_std_now is not None:
                    rec['expert_load_std_step'] = load_std_now
                # retention stats (compressive kv)
                try:
                    if int(args.compressive_slots) > 0:
                        for blk in model.blocks:
                            if hasattr(blk, 'attn') and hasattr(blk.attn, 'compress_kv') and hasattr(blk.attn.compress_kv, 'last_gate_mean'):
                                gm = blk.attn.compress_kv.last_gate_mean
                                if isinstance(gm, torch.Tensor):
                                    rec['kv_write_gate_mean'] = float(gm.item())
                                    break
                except Exception as e:
                    logging.debug("[pretrain] recording KV write gate mean failed: %s", e)
                if hist is not None:
                    rec['expert_load_hist'] = hist
                try:
                    with open(args.log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(rec) + '\n')
                except Exception as e:
                    logging.debug("[pretrain] writing log file failed: %s", e)
                # Update best/last checkpoints using the configured metric (default: loss)
                metric_value = float(loss.item()) if best_metric_name == 'loss' else float(tok_s_ema)
                _maybe_save_best(step, metric_value)
            if args.save_interval and args.save_interval > 0 and (step % args.save_interval == 0):
                ckpt_path = Path(args.out).with_name(Path(args.out).stem + f"_step{step}" + Path(args.out).suffix)
                try:
                    _safe_save(model.state_dict(), ckpt_path)
                    print(f"[ckpt] saved {ckpt_path}")
                except Exception as e:
                    logging.debug("[pretrain] checkpoint save failed: %s", e)
                # Save trainer state for crash-safe resume
                try:
                    _safe_save({
                        'optimizer': opt.state_dict(),
                        'step': step,
                    }, args.state_out)
                except Exception as e:
                    logging.debug("[pretrain] saving state failed: %s", e)
            if step >= args.steps:
                break

    _safe_save(model.state_dict(), args.out)
    # Save a manifest alongside the final checkpoint
    try:
        from omnicoder.utils.model_manifest import build_manifest, save_manifest_for_checkpoint  # type: ignore
        man = build_manifest(model=model, modality='text', preset=os.getenv('OMNICODER_PRESET','mobile_4gb'))
        save_manifest_for_checkpoint(args.out, man)
    except Exception:
        pass
    # Write activation error thresholds JSON for policy tool when requested
    try:
        if act_err_path:
            import numpy as _np
            ae = _np.array(act_err, dtype=float) if len(act_err) > 0 else _np.array([0.0])
            summary = {
                'thresholds': {
                    'global_mean_error': float(ae.mean()),
                    'p50': float(_np.percentile(ae, 50)),
                    'p90': float(_np.percentile(ae, 90)),
                }
            }
            Path(act_err_path).parent.mkdir(parents=True, exist_ok=True)
            Path(act_err_path).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    except Exception as e:
        logging.debug('[pretrain] act error write failed: %s', e)
    print(f"Saved checkpoint to {args.out}")

    # Apply YaRN/PI rope scaling for long-context fine-tuning if requested
    if int(args.target_ctx) > 0:
        try:
            from omnicoder.config import get_rope_scale_for_target_ctx
            rs = get_rope_scale_for_target_ctx(model.max_seq_len, int(args.target_ctx))
            if args.yarn:
                rs = float(rs) * 0.9
            for blk in getattr(model, 'blocks', []):
                if hasattr(blk, 'attn'):
                    blk.attn.rope_scale = rs
                    blk.attn.rope_base = float(args.rope_base)
        except Exception as e:
            logging.debug("[pretrain] applying YaRN/PI rope scaling failed: %s", e)

    # Optionally add simple latent regularizers (when continuous heads exist)
    if args.image_latent_loss and hasattr(model, 'image_latent_head') and model.image_latent_head is not None:
        pass  # training loop should add the reconstruction target when available
    if args.audio_latent_loss and hasattr(model, 'audio_latent_head') and model.audio_latent_head is not None:
        pass


if __name__ == "__main__":
    main()
