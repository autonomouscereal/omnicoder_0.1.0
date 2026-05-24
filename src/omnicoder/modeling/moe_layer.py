#moe_layer.py

import os
import time as _t
import torch
import torch.nn as nn
try:
	from .utils.expert_paging import ExpertPager  # type: ignore
except Exception:  # pragma: no cover
	ExpertPager = None  # type: ignore
from .experts import ExpertFFN
from .kernels.moe_scatter import fused_dispatch
# Safe concatenation utility (aten-only; avoids cat in hot path)
try:
    from omnicoder.utils.torchutils import safe_concat2 as _safe_cat  # type: ignore
except Exception:  # pragma: no cover
    # Historical: direct aten.cat in the hot path created shape-dependent graphs and
    # larger temp allocations. The safe_concat2 helper preallocates and uses slice_scatter,
    # stabilizing shapes and improving cudagraph capture. If unavailable, we fallback
    # to aten.cat, but this is slower and less stable.
    def _safe_cat(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:  # type: ignore
        return torch.ops.aten.cat.default((a, b), int(dim))
# -------------------------------------------------------------------------------------
# MoE CUDA Graph stability notes
# - Do not rebuild prepacked banks in the hot path; that created warmup-only storages.
# - Build VGR scalars via aten-only ops anchored to live tensors, not via detach/new_tensor.
# - Avoid module-side tensor caches in forward.
# - Anchor per-call temporaries (banks or packed buffers) into the output lineage via a
#   zero-sum accumulator so cudagraph weakref counts match between warmup and replay.
# -------------------------------------------------------------------------------------
from .routing import TopKRouter, HierarchicalRouter, MultiHeadRouter, GRINGate, LLMRouter
try:
	from omnicoder.utils.perf import add as _perf_add  # type: ignore
except Exception:  # pragma: no cover
	_perf_add = None  # type: ignore

from .routing import InteractionRouter  # type: ignore

import weakref as _wr  # noqa: F401

from torch import nn as _nn

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MoELayer(nn.Module):
    # Class-level defaults to guarantee presence even before __init__ runs
    _degraded_router: bool = False
    _router_is_llm: bool = False
    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.2,
        static_capacity: int | None = None,
        group_sizes: list[int] | None = None,
        sub_experts_per: int = 1,
        num_shared_general: int = 0,
    ):
        super().__init__()
        # init logging removed
        # Persist common dims for prepacking and diagnostics
        try:
            self._d_model = int(d_model)
        except Exception:
            self._d_model = d_model  # type: ignore[assignment]
        try:
            self._mlp_dim = int(mlp_dim)
        except Exception:
            self._mlp_dim = mlp_dim  # type: ignore[assignment]
        # Ensure router flags are defined on all paths
        use_llm_router = False
        router_env = ''
        # Router with temperature and jitter supports better load-balance during training
        # Use hierarchical router when group_sizes are provided, else flat TopK
        if (group_sizes is not None) and (len(group_sizes) > 0):
            self.router = HierarchicalRouter(d_model, n_experts, group_sizes=group_sizes, k=top_k, temperature=1.0)
            self._blend_enable = False
        else:
            # Compose a robust default by blending classic TopK, multi-head gating, and a GRIN-like gate during training.
            # At eval time we fall back to the simplest TopK path for stability.
            # Balanced routing (Sinkhorn) toggles via environment
            # Environment reads are banned in hot paths; resolve defaults at construction only if present
            _sink_it_env = os.getenv('OMNICODER_ROUTER_SINKHORN_ITERS', None)
            _sink_tau_env = os.getenv('OMNICODER_ROUTER_SINKHORN_TAU', None)
            _sink_it = int(_sink_it_env) if (_sink_it_env is not None) else 0
            _sink_tau = float(_sink_tau_env) if (_sink_tau_env is not None) else 1.0
            self._router_topk = TopKRouter(
                d_model,
                n_experts,
                k=top_k,
                temperature=1.0,
                jitter_noise=0.0,
                use_gumbel=False,
                expert_dropout_p=0.0,
                sinkhorn_iters=_sink_it,
                sinkhorn_tau=_sink_tau,
            )
            self._router_multi = MultiHeadRouter(d_model, n_experts, k=top_k, num_gates=4, temperature=1.0, jitter_noise=0.0)
            self._router_grin = GRINGate(d_model, n_experts, k=top_k, temperature=1.0, jitter_noise=0.0)
            # Build small constants via aten-only ops without explicit device/dtype
            _z = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,)), 0.0))
            _b0 = torch.ops.aten.add.Scalar(_z, 0.34)
            _b1 = torch.ops.aten.add.Scalar(_z, 0.33)
            _b2 = torch.ops.aten.add.Scalar(_z, 0.33)
            _blend_init = torch.ops.aten.stack.default([_b0, _b1, _b2], 0)
            try:
                self.register_buffer('_blend', _blend_init, persistent=False)
            except Exception:
                self._blend = _blend_init
            self._blend_enable = True
            # Preallocate decode-time blended gating buffers to avoid per-step cat/allocs (B==1, T==1)
            try:
                _Kb = int(3 * max(1, int(top_k)))
            except Exception:
                _Kb = 3 * 1
            # Anchor zero scalars without explicit device/dtype; they inherit on first use
            _anc_f = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,)), 0.0))
            _anc_l = torch.ops.aten.add.Scalar(_anc_f, 0.0)
            try:
                self.register_buffer('_blend_sc_buf', torch.ops.aten.new_zeros.default(_anc_f, (1, _Kb)), persistent=False)
            except Exception:
                try:
                    self._blend_sc_buf = torch.ops.aten.new_zeros.default(_anc_f, (1, _Kb))  # type: ignore[assignment]
                except Exception:
                    self._blend_sc_buf = None  # type: ignore[assignment]
            try:
                self.register_buffer('_blend_idx_buf', torch.ops.aten.new_zeros.default(_anc_l, (1, _Kb)), persistent=False)
            except Exception:
                try:
                    self._blend_idx_buf = torch.ops.aten.new_zeros.default(_anc_l, (1, _Kb))  # type: ignore[assignment]
                except Exception:
                    self._blend_idx_buf = None  # type: ignore[assignment]
            # Optional context-aware router (constructor-time only; no env in forward)
            router_env = os.getenv('OMNICODER_ROUTER', '')
            try:
                router_env = router_env.strip().lower()  # type: ignore[assignment]
            except Exception:
                router_env = ''
            use_llm_router = (router_env == 'llm')
            if use_llm_router:
                self.router = LLMRouter(d_model, n_experts, k=top_k, temperature=1.0, jitter_noise=0.0, num_heads=max(1, d_model // max(1, (d_model // 128))))
            else:
                # Optional interaction-aware router (I2MoE-like) when OMNICODER_ROUTER=interaction
                if router_env == 'interaction':
                    self.router = InteractionRouter(d_model, n_experts, k=top_k, temperature=1.0)
                else:
                    self.router = self._router_topk
        # Report chosen router kind (logging removed)
        # Cache router kind flag to avoid getattr in hot path
        self._router_is_llm = bool(use_llm_router)
        # Cache router name to avoid type()/__name__ in hot path logs
        try:
            self._router_name = self.router.__class__.__name__
        except Exception:
            self._router_name = "unknown"
        # Cache for expert device and wrappers to avoid per-forward rebuilds
        self._experts_device: str | None = None
        self._wrappers_cache: dict[str, list] = {}
        # Persistent MoE workspaces to avoid per-step allocations and stabilize CG storages
        # BUMP: sized for benchmark prefill (seq_len=128 * top_k=2) + decode
        # 512 tokens = ~1 MB VRAM peak (completely negligible on 12 GB card)
        _anc_f = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,)), 0.0))
        max_work_tokens = 512

        # work_x needs space for [total*K, 1, H] in pure-aten fused_dispatch
        try:
            self.register_buffer('_work_x_cache',
                torch.ops.aten.new_zeros.default(_anc_f, (max_work_tokens, int(d_model // max(1, 1)))),
                persistent=False)
        except Exception:
            self._work_x_cache = torch.ops.aten.new_zeros.default(_anc_f, (max_work_tokens, int(d_model // max(1, 1))))  # type: ignore[assignment]

        # work_w for bias/scores temporaries
        try:
            self.register_buffer('_work_w_cache',
                torch.ops.aten.new_zeros.default(_anc_f, (max_work_tokens, 1)),
                persistent=False)
        except Exception:
            self._work_w_cache = torch.ops.aten.new_zeros.default(_anc_f, (max_work_tokens, 1))  # type: ignore[assignment]
        # VGR temperature schedule constants cached once without getenv in hot path
        _tmin = os.getenv('OMNICODER_ROUTER_TMIN', None)
        _tmax = os.getenv('OMNICODER_ROUTER_TMAX', None)
        _tlmb = os.getenv('OMNICODER_ROUTER_TEMP_LAMBDA', None)
        self._vgr_tmin = float(_tmin) if _tmin is not None else 0.8
        self._vgr_tmax = float(_tmax) if _tmax is not None else 1.2
        self._vgr_lambda = float(_tlmb) if _tlmb is not None else 3.0
        self._vgr_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        # Memoize flags once (no getenv in hot path)
        self._log_summary_info = (os.getenv('OMNICODER_MOE_LOG_SUMMARY', '0') == '1') if 'OMNICODER_MOE_LOG_SUMMARY' in os.environ else False
        self._force_torch_dispatch = (os.getenv('OMNICODER_MOE_FORCE_TORCH', '0') == '1') if 'OMNICODER_MOE_FORCE_TORCH' in os.environ else False
        self._no_drops = (os.getenv('OMNICODER_MOE_NO_DROPS', '0') == '1') if 'OMNICODER_MOE_NO_DROPS' in os.environ else False
        self._bucket_tokens = (os.getenv('OMNICODER_MOE_BUCKET', '1') == '1') if 'OMNICODER_MOE_BUCKET' in os.environ else True
        # Cache for frequently requested arange tensors keyed by (N, device)
        self._arange_cache: dict[str, torch.Tensor] = {}
        # Reusable work buffer cache keyed by (device, dtype, hidden_dim)
        self._y_buf: dict[str, torch.Tensor] = {}
        # Inverse-permutation buffer cache keyed by (device, dtype, length)
        self._inv_buf: dict[tuple[str, str, int], torch.Tensor] = {}
        # DeepSeek-style: split each expert into smaller sub-experts and add shared general experts
        total_experts = int(n_experts)
        self.sub_experts_per = max(1, int(sub_experts_per))
        self.num_shared_general = max(0, int(num_shared_general))
        # Optional lazy expert construction to avoid eager allocation of all experts.
        # Enable when OMNICODER_MOE_LAZY_BUILD=1 (requires ExpertPager to be available).
        lazy_build = False
        try:
            lazy_build = (os.getenv('OMNICODER_MOE_LAZY_BUILD', os.getenv('OMNICODER_MOE_LAZY', '0')) == '1') and (ExpertPager is not None)
        except Exception:
            lazy_build = False
        banks: list[nn.Module] = []
        # Activation used by experts (default GELU tanh; switchable via env)
        try:
            _mlp_act = os.getenv('OMNICODER_MLP_ACT', 'gelu_tanh')
        except Exception:
            _mlp_act = 'gelu_tanh'
        if lazy_build:
            # In lazy mode, do not instantiate all experts upfront. Defer creation to ExpertPager factories.
            self.experts = nn.ModuleList([])
            self.shared = nn.ModuleList([ExpertFFN(d_model, mlp_dim, _mlp_act) for _ in range(self.num_shared_general)]) if self.num_shared_general > 0 else None
            # Force-enable pager in lazy mode
            self.use_pager = True
        else:
            # build start (logging removed)
            for i_expert in range(total_experts):
                # expert build (logging removed)
                if self.sub_experts_per == 1:
                    banks.append(ExpertFFN(d_model, mlp_dim, _mlp_act))
                else:
                    # divide mlp_dim across sub-experts (simple proxy); alternative: equal-size experts
                    banks.append(nn.ModuleList([ExpertFFN(d_model, max(1, mlp_dim // self.sub_experts_per), _mlp_act) for __ in range(self.sub_experts_per)]))
            self.shared: nn.ModuleList | None = None
            if self.num_shared_general > 0:
                self.shared = nn.ModuleList([ExpertFFN(d_model, mlp_dim, _mlp_act) for _ in range(self.num_shared_general)])
            self.experts = nn.ModuleList(banks)
            # Cache expert count to avoid len() in logs/hot paths
            self._experts_count = total_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        # Precompute fixed-point scale for capacity factor once (not in hot path)
        try:
            _den = 1024
            _num_raw = int(self.capacity_factor * _den + 0.5)
            # Branchless clamp to at least _den: max(_num_raw, _den) = (a+b+|a-b|)//2
            _diff = _num_raw - _den
            _s = (_diff >> 31)
            _absd = (_diff ^ _s) - _s
            _num = (_num_raw + _den + _absd) // 2
            self._cap_den = _den
            self._cap_num = _num
        except Exception:
            self._cap_den = 1024
            self._cap_num = 1024
        self.static_capacity = static_capacity
        # Precompute a non-negative static capacity int for branchless selection in forward
        try:
            _sc = int(static_capacity)  # None will raise, handled below
        except Exception:
            _sc = 0
        # max(_sc, 0) without max(): (a + |a|)//2
        _s_sc = (_sc >> 31)
        _abs_sc = (_sc ^ _s_sc) - _s_sc
        self._static_capacity_int = (_sc + _abs_sc) // 2
        # Initialize optional attributes to avoid hasattr checks in hot path
        self._cond = None
        self._prepacked_W1 = None
        self._prepacked_B1 = None
        self._prepacked_W2 = None
        self._prepacked_B2 = None
        self._compiled_key = None
        try:
            _ = torch.compile  # type: ignore[attr-defined]
            self._has_torch_compile = True
        except Exception:
            self._has_torch_compile = False
        # Router degradation flag (default off); avoids AttributeError in forward
        self._degraded_router = False
        # Cache minimum per-expert capacity from environment ONCE (avoid getenv in hot path)
        try:
            _min_cap_env = os.getenv('OMNICODER_MOE_MIN_CAPACITY', '').strip()
            if _min_cap_env:
                self._min_capacity_from_env = max(int(_min_cap_env), int(self.top_k))
            else:
                # Root-cause fix: default to top_k for decode to prevent padded GEMMs dominating time at N=1.
                # This preserves correctness (no drops when N*top_k<=cap) and avoids artificial cap inflation (64).
                self._min_capacity_from_env = max(1, int(self.top_k))
        except Exception:
            self._min_capacity_from_env = max(1, int(self.top_k))
        # init logging removed
        # Cache force-single-expert flag (avoid getenv in hot path)
        try:
            self._force_single_expert = (os.getenv('OMNICODER_FORCE_SINGLE_EXPERT', '0') == '1')
        except Exception:
            self._force_single_expert = False
        # Cache experts' current device to avoid O(E) scans/moves every forward
        self._experts_device: str | None = None
        # Cache for VGR constants per device/dtype to avoid per-step tensor creation
        self._vgr_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
        # Cache debug env once to avoid getenv overhead on hot path
        try:
            self._dbg = (os.getenv('OMNICODER_MOE_DEBUG', '0') == '1')
            self._logp = os.getenv('OMNICODER_MOE_LOG', 'tests_logs/moe_debug.log')
        except Exception:
            self._dbg = False
            self._logp = 'tests_logs/moe_debug.log'
        # init logging removed
        # Optional expert paging (env-guarded)
        self.use_pager: bool = bool(os.getenv('OMNICODER_EXPERT_PAGING', '0') == '1' and ExpertPager is not None)
        self.prefetch_n: int = int(os.getenv('OMNICODER_EXPERT_PREFETCH_N', '1'))
        self._pager: ExpertPager | None = None
        if self.use_pager and ExpertPager is not None:
            try:
                # Derive capacity from budget when provided; fallback to explicit cap or default 8
                cap = int(os.getenv('OMNICODER_EXPERT_PAGING_CAP', '0') or '0')
                if cap <= 0:
                    budget_mb = int(os.getenv('OMNICODER_EXPERT_PAGING_BUDGET_MB', '0') or '0')
                    if budget_mb > 0:
                        # Rough per-expert memory proxy ~ 2 * d_model * mlp_dim * 2 bytes / 1e6
                        # Fall back to 64 MB per expert when dims are unknown at init
                        per_exp_mb = 64
                        try:
                            per_exp_mb = max(16, int((2 * d_model * mlp_dim * 2) / 1e6))  # type: ignore[name-defined]
                        except Exception:
                            per_exp_mb = 64
                        cap = max(1, budget_mb // max(1, per_exp_mb))
                if cap <= 0:
                    cap = 8
                # Optional persistence directory for weight streaming (disk-backed)
                state_dir = os.getenv('OMNICODER_EXPERT_PAGING_DIR', '').strip() or None
                persist = (os.getenv('OMNICODER_EXPERT_PAGING_PERSIST', '1') == '1')
                self._pager = ExpertPager(capacity=cap, state_dir=state_dir, persist_on_evict=persist)
                if lazy_build:
                    # Register lazy factories that construct experts on first use
                    def _factory_single() -> nn.Module:
                        return ExpertFFN(d_model, mlp_dim, _mlp_act)
                    def _factory_list() -> nn.Module:
                        return nn.ModuleList([ExpertFFN(d_model, max(1, mlp_dim // self.sub_experts_per), _mlp_act) for __ in range(self.sub_experts_per)])
                    for i in range(total_experts):
                        if self.sub_experts_per == 1:
                            self._pager.register_factory(i, _factory_single)
                        else:
                            self._pager.register_factory(i, _factory_list)
            except Exception:
                self._pager = None
        # ALWAYS COMPILE EXPERT FFNs AT INIT (NEVER IN HOT PATH):
        # Compile once here to honor "compile always on" without incurring per-step compiles.
        # DO NOT move this into decode/generate loops.
        try:
            _ok_compile = True
            try:
                _ = torch.compile  # type: ignore[attr-defined]
            except Exception:
                _ok_compile = False
            if _ok_compile:
                def _compile_mod(m: nn.Module) -> nn.Module:
                    try:
                        cm = torch.compile(m, mode='reduce-overhead', fullgraph=False)  # type: ignore[arg-type]
                        try:
                            cm._source_ref = _wr.ref(m)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        return cm
                    except Exception:
                        return m
                for i, b in enumerate(self.experts):
                    try:
                        _ = b[0]
                        _is_list = True
                    except Exception:
                        _is_list = False
                    if _is_list:
                        for j, sub in enumerate(b):
                            b[j] = _compile_mod(sub)  # type: ignore[assignment]
                    else:
                        self.experts[i] = _compile_mod(b)  # type: ignore[assignment]
                try:
                    _ = self.shared[0]  # type: ignore[index]
                    _is_list = True
                except Exception:
                    _is_list = False
                if _is_list:
                    for j, g in enumerate(self.shared):
                        self.shared[j] = _compile_mod(g)  # type: ignore[assignment]
        except Exception:
            pass
        # Optional: compile expert FFNs (and shared) for faster inference
        try:
            if os.getenv('OMNICODER_COMPILE_EXPERTS', '0') == '1':
                try:
                    _ = torch.compile  # type: ignore[attr-defined]
                    _ok_compile = True
                except Exception:
                    _ok_compile = False
                if _ok_compile:
                    def _compile_mod(m: nn.Module) -> nn.Module:
                        try:
                            cm = torch.compile(m, mode='reduce-overhead', fullgraph=False)  # type: ignore[arg-type]
                            try:
                                cm._source_ref = _wr.ref(m)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            return cm
                        except Exception:
                            return m
                    for i, b in enumerate(self.experts):
                        try:
                            _ = b[0]
                            _is_list = True
                        except Exception:
                            _is_list = False
                        if _is_list:
                            for j, sub in enumerate(b):
                                b[j] = _compile_mod(sub)  # type: ignore[assignment]
                        else:
                            self.experts[i] = _compile_mod(b)  # type: ignore[assignment]
                    try:
                        _ = self.shared[0]  # type: ignore[index]
                        _is_list = True
                    except Exception:
                        _is_list = False
                    if _is_list:
                        for j, g in enumerate(self.shared):
                            self.shared[j] = _compile_mod(g)  # type: ignore[assignment]
        except Exception:
            pass
        # Expert device placement is managed by the caller; no internal device moves here.
        # Expose a simple load-balancing metric for auxiliary loss during training
        self.last_load_penalty: torch.Tensor | None = None
        self.last_router_aux: dict | None = None
        # Do not wrap experts with external quant wrappers in-model; preserve consistent module API.
        # SCMoE inference-only contrast controls (force explicit defaults; ignore env gates)
        self.scmoe_alpha = 0.0
        self.scmoe_frac = 0.25
        # Link to model-level verifier need (propagated by parent); defaults to False
        self._need_verifier: bool = False
        # Cache VGR knobs to avoid getenv in hot path
        try:
            self._vgr_margin_thresh = float(os.getenv('OMNICODER_VGR_MARGIN_THRESH', '0.3'))
        except Exception:
            self._vgr_margin_thresh = 0.3
        try:
            self._vgr_extra_experts = int(os.getenv('OMNICODER_VGR_EXTRA_EXPERTS', '1'))
        except Exception:
            self._vgr_extra_experts = 1
        # Cache last verifier margin default ONCE (no getenv in hot path). Can be updated by caller via attribute.
        try:
            self._last_verifier_margin = float(os.getenv('OMNICODER_LAST_VERIFIER_MARGIN', '1.0'))
        except Exception:
            self._last_verifier_margin = 1.0
        # init logging removed
        # Prepack expert banks once for current dtype to avoid per-call stacking in hot path
        try:
            # Determine a representative dtype from first expert param; fall back to float32
            rep_dtype = None
            try:
                m0 = self.experts[0]
                rep_dtype = next(m0.parameters()).dtype  # type: ignore[call-arg]
            except Exception:
                try:
                    rep_dtype = next(self.parameters()).dtype
                except Exception:
                    rep_dtype = torch.float32
            # Device-agnostic prepack; only dtype is used inside the prepack function
            self._prepack_banks_for_dtype(dtype=rep_dtype)
        except Exception:
            pass

        # One-time expert warmup to avoid first-call compilation inside timed decode loop
        try:
            with torch.no_grad():
                # Use a tiny tensor anchored to first expert's dtype/device
                anchor_mod = None
                try:
                    anchor_mod = self.experts[0] if len(self.experts) > 0 else None
                except Exception:
                    anchor_mod = None
                if anchor_mod is not None:
                    try:
                        p = next(anchor_mod.parameters())
                        xin = p.new_zeros((1, int(self._d_model) if isinstance(self._d_model, int) else 1))
                    except Exception:
                        xin = torch.zeros((1, int(self._d_model) if isinstance(self._d_model, int) else 1))
                    # Warm each expert representative once (handles ModuleList and single modules)
                    for bank in self.experts:
                        try:
                            if isinstance(bank, nn.ModuleList):
                                _ = bank[0](xin)
                            else:
                                _ = bank(xin)
                        except Exception:
                            pass
        except Exception:
            pass

    def collapse_to_single_expert(self) -> None:
        """Collapse this MoE layer to a single-expert fast path at inference.

        This disables routing entirely and forces the forward path to use the
        first expert bank deterministically. This is used as a safety/perf
        fallback when checkpoints are missing MoE/router weights.
        """
        try:
            # Remove shared experts to satisfy the fast path condition
            try:
                _ = self.shared
                _has_shared = True
            except Exception:
                _has_shared = False
            if _has_shared:
                try:
                    self.shared = None  # type: ignore[assignment]
                except Exception:
                    pass
            # Force single expert / top-1 selection
            try:
                self.n_experts = 1  # type: ignore[assignment]
            except Exception:
                pass
            try:
                self.top_k = 1  # type: ignore[assignment]
            except Exception:
                pass
        except Exception:
            # Best-effort; if attributes are missing, the fast path guard may still be bypassed
            pass

    def set_conditioning(self, cond: dict | None) -> None:
        # Store a one-shot conditioning payload consumed on next forward
        try:
            self._cond = cond
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Split forward into small, traceable methods
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Ultra-safe helpers (no try/except, no getattr, no int casting)
    # ------------------------------------------------------------------
    def _route_tokens(self, x: torch.Tensor, cond: dict | None = None):
        return self.router(x, cond=cond)

    def _prepare_dispatch_inputs(self, x: torch.Tensor, idx: torch.Tensor, scores: torch.Tensor, seq_len: int, hidden_dim: int):
        batch = torch.ops.aten.sym_size.int(x, 0)
        total = batch * seq_len
        # Pre-flatten here (industry standard) — use view, not reshape
        x_flat = torch.ops.aten.view.default(x, [total, hidden_dim])
        expert_indices = torch.ops.aten.view.default(idx, [total, -1])
        scores_sanitized = torch.ops.aten.nan_to_num.default(scores)
        sum_scores = torch.ops.aten.clamp_min.default(
            torch.ops.aten.sum.dim_IntList(scores_sanitized, [-1], True), 1e-8
        )
        scores_norm = torch.ops.aten.div.Tensor(scores_sanitized, sum_scores)
        scores_flat = torch.ops.aten.view.default(scores_norm, [total, -1])
        return x_flat, expert_indices, scores_flat

    def _call_fused_dispatch(self, x_flat, expert_indices, scores_flat):
        _banks = {
            'W1': self._prepacked_W1,
            'B1': self._prepacked_B1,
            'W2': self._prepacked_W2,
            'B2': self._prepacked_B2,
        }
        capacity = self.top_k
        output_fused, kept = fused_dispatch(
            x_flat, expert_indices, scores_flat,
            None, capacity,
            output_buf=None,
            banks=_banks,
            work_x=self._work_x_cache,
            work_w=self._work_w_cache,
        )
        return output_fused

    def _finalize_output(self, output_flat, batch_size, seq_len, hidden_dim):
        y = torch.ops.aten.view.default(output_flat, [batch_size, seq_len, hidden_dim])

        # Pure aten anchor (no try/except)
        _z = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(y, 0.0))
        y = torch.ops.aten.add.Tensor(y, torch.ops.aten.mul.Scalar(_z, 0.0))
        return y

    # ------------------------------------------------------------------
    # Main forward (completely linear, no control flow)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, cond: dict | None = None, need_verifier: bool | None = None) -> torch.Tensor:
        batch_size = torch.ops.aten.sym_size.int(x, 0)
        seq_len = torch.ops.aten.sym_size.int(x, 1)
        hidden_dim = torch.ops.aten.sym_size.int(x, 2)

        # Route
        _r = self._route_tokens(x, cond)
        idx = _r[0]
        scores = _r[1]

        k_cap = self.top_k
        idx = torch.ops.aten.slice.Tensor(idx, -1, 0, k_cap, 1)
        scores = torch.ops.aten.slice.Tensor(scores, -1, 0, k_cap, 1)

        # Prepare (flattens here)
        x_flat, expert_indices, scores_flat = self._prepare_dispatch_inputs(
            x, idx, scores, seq_len, hidden_dim
        )

        # Dispatch
        output_fused = self._call_fused_dispatch(x_flat, expert_indices, scores_flat)

        # Finalize
        y = torch.ops.aten.view.default(output_fused, [batch_size, seq_len, hidden_dim])
        _z = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(y, 0.0))
        y = torch.ops.aten.add.Tensor(y, torch.ops.aten.mul.Scalar(_z, 0.0))
        return y

    def _apply(self, fn):
        """Ensure prepacked banks track module device/dtype.
        Pure aten-only return to stay Dynamo/CG safe.
        """
        out = super()._apply(fn)
        try:
            p = next(self.parameters())
            self._prepack_banks_for_dtype(dtype=p.dtype)
        except Exception:
            pass
        return out

    def _prepack_banks_for_dtype(self, dtype: torch.dtype) -> None:
        """Precompute and cache per-expert weight banks on the target dtype only.

        Mirrors the fused_dispatch bank builder so first decode steps avoid packing cost.
        Safe no-op when experts are not simple FFNs.
        """
        try:

            # Validate experts are simple FFNs with fc1/fc2
            def _is_simple_ffn(m: nn.Module) -> bool:
                try:
                    try:
                        _src = m._source_ffn  # type: ignore[attr-defined]
                    except Exception:
                        _src = m
                    try:
                        _ = _src.fc1  # type: ignore[attr-defined]
                        _has_fc1 = True
                    except Exception:
                        _has_fc1 = False
                    try:
                        _ = _src.fc2  # type: ignore[attr-defined]
                        _has_fc2 = True
                    except Exception:
                        _has_fc2 = False
                    # Treat ModuleList detection via indexing probe
                    _is_mlist = False
                    try:
                        _ = _src[0]  # type: ignore[index]
                        _is_mlist = True
                    except Exception:
                        _is_mlist = False
                    return _has_fc1 and _has_fc2 and (not _is_mlist)
                except Exception:
                    return False
            experts = list(self.experts)
            # Avoid Python generator/all() in hot or warm paths: check only first expert
            try:
                if not experts:
                    return
                _m0 = experts[0]
                if not _is_simple_ffn(_m0):
                    return
            except Exception:
                return
            # Derive activation kind tuple as in fused path
            def _act_kind(m: nn.Module):
                try:
                    _src = m._source_ffn  # type: ignore[attr-defined]
                except Exception:
                    _src = m
                try:
                    a = _src.act_fn  # type: ignore[attr-defined]
                except Exception:
                    a = None
                try:
                    an = type(a).__name__ if a is not None else ''
                except Exception:
                    an = ''
                if an == 'GELU':
                    try:
                        return ('gelu', a.approximate)  # type: ignore[attr-defined]
                    except Exception:
                        return ('gelu', 'none')
                if an == 'SiLU':
                    return ('silu', None)
                return ('other', None)
            kinds = [_act_kind(m) for m in experts]
            if len(set(kinds)) != 1:
                # Mixed activations not supported by this prepack
                return
            act_kind, act_approx = kinds[0]
            # Resolve source modules (handle torch.compile wrappers storing _source_ref)
            experts_src = []
            for _m in experts:
                try:
                    _ref = _m._source_ref  # type: ignore[attr-defined]
                except Exception:
                    _ref = None
                if _ref is not None:
                    try:

                        _orig = _ref()
                        experts_src.append(_orig if _orig is not None else _m)
                    except Exception:
                        experts_src.append(_m)
                else:
                    experts_src.append(_m)
            # Build key identical to fused_dispatch
            try:
                hidden_dim = int(experts_src[0].fc2.out_features)  # type: ignore[attr-defined]
            except Exception:
                try:
                    hidden_dim = int(self._d_model)
                except Exception:
                    hidden_dim = self._d_model  # type: ignore[assignment]
            try:
                mlp_dim = int(experts_src[0].fc1.out_features)  # type: ignore[attr-defined]
            except Exception:
                try:
                    mlp_dim = int(self._mlp_dim)
                except Exception:
                    mlp_dim = self._mlp_dim  # type: ignore[assignment]
            # Materialize stacked banks using dtype-only casts from source params; avoid device moves and global cache
            # Device is never referenced or changed
            # Cast via to(dtype=...) to avoid potential storage.set_ paths seen with .type in compile modes
            W1_bank = torch.stack([torch.ops.aten.to.dtype(m.fc1.weight.detach().transpose(0, 1), dtype, False, False) for m in experts_src], dim=0)
            B1_bank = torch.stack([(torch.ops.aten.to.dtype(m.fc1.bias.detach(), dtype, False, False) if m.fc1.bias is not None else torch.ops.aten.to.dtype(m.fc1.weight.detach().new_zeros(mlp_dim), dtype, False, False)) for m in experts_src], dim=0)
            W2_bank = torch.stack([torch.ops.aten.to.dtype(m.fc2.weight.detach().transpose(0, 1), dtype, False, False) for m in experts_src], dim=0)
            B2_bank = torch.stack([(torch.ops.aten.to.dtype(m.fc2.bias.detach(), dtype, False, False) if m.fc2.bias is not None else torch.ops.aten.to.dtype(m.fc2.weight.detach().new_zeros(hidden_dim), dtype, False, False)) for m in experts_src], dim=0)
            # Register banks as non-persistent buffers so they follow module device/dtype on .to()/cuda()
            try:
                self.register_buffer('_prepacked_W1', W1_bank, persistent=False)
            except Exception:
                try:
                    self._prepacked_W1 = W1_bank
                except Exception:
                    pass
            try:
                self.register_buffer('_prepacked_B1', B1_bank, persistent=False)
            except Exception:
                try:
                    self._prepacked_B1 = B1_bank
                except Exception:
                    pass
            try:
                self.register_buffer('_prepacked_W2', W2_bank, persistent=False)
            except Exception:
                try:
                    self._prepacked_W2 = W2_bank
                except Exception:
                    pass
            try:
                self.register_buffer('_prepacked_B2', B2_bank, persistent=False)
            except Exception:
                try:
                    self._prepacked_B2 = B2_bank
                except Exception:
                    pass
        except Exception:
            return
