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
                    from .routing import InteractionRouter  # type: ignore
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
        _anc_f = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(torch.ops.aten.new_zeros.default(torch.tensor(0.0), (1,)), 0.0))
        # Decode path (B=1, T=1) workspaces
        try:
            self.register_buffer('_work_x_cache', torch.ops.aten.new_zeros.default(_anc_f, (max(1, int(n_experts * top_k)), int(d_model // max(1, 1)))), persistent=False)
        except Exception:
            self._work_x_cache = torch.ops.aten.new_zeros.default(_anc_f, (max(1, int(n_experts * top_k)), int(d_model // max(1, 1))))  # type: ignore[assignment]
        try:
            self.register_buffer('_work_w_cache', torch.ops.aten.new_zeros.default(_anc_f, (max(1, int(n_experts * top_k)), 1))),
        except Exception:
            self._work_w_cache = torch.ops.aten.new_zeros.default(_anc_f, (max(1, int(n_experts * top_k)), 1))  # type: ignore[assignment]
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
                            import weakref as _wr  # local import to avoid global dependency
                            try:
                                cm._source_ref = _wr.ref(m)  # type: ignore[attr-defined]
                            except Exception:
                                pass
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
                                import weakref as _wr  # local import to avoid global dependency
                                try:
                                    cm._source_ref = _wr.ref(m)  # type: ignore[attr-defined]
                                except Exception:
                                    pass
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

    def forward(self, x: torch.Tensor, cond: dict | None = None, need_verifier: bool | None = None) -> torch.Tensor:
        # x: (B, T, C)
        # NOTE [Export friendliness]: avoid forward-time structural changes; keep routing unified.
        # Ultra-fast path: single expert with top_k=1 and no shared experts. Only activate when explicitly forced.
        if self._force_single_expert:
            try:
                try:
                    _ne = self.n_experts
                    _tk = self.top_k
                    _sh = self.shared if (self.shared is not None) else []
                except Exception:
                    _ne, _tk, _sh = (0, 0, [])
                if (_ne == 1) and (_tk == 1):
                    if (_sh is None) or (len(_sh) == 0):  # type: ignore[arg-type]
                        bank = self.experts[0]
                        from torch import nn as _nn  # local import to avoid global
                        try:
                            _ = bank[0]
                            _is_list = True
                        except Exception:
                            _is_list = False
                        if _is_list:
                            outs = [sub(x) for sub in bank]
                            y = sum(outs) / float(len(outs))
                        else:
                            y = bank(x)
                        # removed hot-path logging
                        return y
            except Exception:
                pass
        # Derive shapes via aten-only ops to avoid Python shape guards
        batch_size = torch.ops.aten.sym_size.int(x, 0)
        seq_len = torch.ops.aten.sym_size.int(x, 1)
        hidden_dim = torch.ops.aten.sym_size.int(x, 2)
        # removed hot-path logging
        # Route tokens (timing removed)
        # Degraded/partial router fallback: uniform routing when flagged
        # PERMANENT: disable degraded router fallback in hot path; robust unified routing only
        if False:
            E = self.n_experts
            K = self.top_k
            # Build uniform routing probs without .device and without Python max():
            # 1) ones tensor anchored to x lineage
            probs_full = torch.ops.aten.new_ones.default(x, (batch_size, seq_len, E), dtype=x.dtype)
            # 2) denom = clamp_min(E, 1) as a 0-d tensor anchored to x (avoid Python min/max)
            _zero = torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.slice.Tensor(x, -1, 0, 1, 1)), 0.0)
            _E_t = torch.ops.aten.add.Scalar(_zero, float(E))
            _den = torch.ops.aten.clamp_min.default(_E_t, 1)
            probs_full = torch.ops.aten.div.Tensor(probs_full, _den)
            # Use configured K directly (assumed valid); avoid Python min()
            _tk = torch.ops.aten.topk.default(probs_full, K, -1, True, True)
            topk_vals, idx = _tk[0], _tk[1]
            # Stabilized softmax via aten: subtract amax over last dim
            mx = torch.ops.aten.amax.default(topk_vals, [-1], True)
            scores = torch.ops.aten.softmax.int(torch.ops.aten.sub.Tensor(topk_vals, mx), -1)
        elif (not self._router_is_llm) and self._blend_enable and self.training:
            # removed hot-path logging
            _ra = self._router_topk(x)
            idx_a, scores_a, p_a = _ra[0], _ra[1], _ra[2]
            _rb = self._router_multi(x)
            idx_b, scores_b, p_b = _rb[0], _rb[1], _rb[2]
            _rc = self._router_grin(x)
            idx_c, scores_c, p_c = _rc[0], _rc[1], _rc[2]
            w = torch.ops.aten.softmax.int(self._blend, 0)
            probs_full = w[0] * p_a + w[1] * p_b + w[2] * p_c
            # Select top-k directly for efficiency
            _tk2 = torch.ops.aten.topk.default(probs_full, self.top_k, -1, True, True)
            topk_vals, idx = _tk2[0], _tk2[1]
            mx2 = torch.ops.aten.amax.default(topk_vals, [-1], True)
            scores = torch.ops.aten.softmax.int(torch.ops.aten.sub.Tensor(topk_vals, mx2), -1)
        else:
            # NOTE [No module-state mutation]: pass per-call conditioning directly to the router.
            try:
                _r = self.router(x, cond=cond)  # type: ignore[call-arg]
            except Exception:
                _r = self.router(x)  # type: ignore[misc]
            idx, scores, probs_full = _r[0], _r[1], _r[2]
            # removed hot-path logging/timing
        # IMPORTANT: Do NOT mutate module state during forward under gradient checkpointing.
        # Forward-time state mutations (like clearing conditioning) can cause the recompute
        # pass to take a different path and trip tensor-save count mismatches. We therefore
        # intentionally avoid clearing one-shot conditioning here. Callers may overwrite
        # conditioning on the next step as needed.

        # No K=1 specialization; keep unified TopK/softmax path for graph stability

        # Keep general dispatch path; single-token specialization is handled inside fused_dispatch.

        # Verifier-Guided Routing (VGR) hooks (inference-time; no grad):
        # Apply temperature scaling only; keep K fixed to preserve static graph shapes.
        try:
            # Always enable VGR at inference; ignore env gates
            if (not self.training):
                # T = Tmin + (Tmax - Tmin) * sigmoid(lambda * entropy)
                tmin = self._vgr_tmin
                tmax = self._vgr_tmax
                lam = self._vgr_lambda
                # Derive a crude entropy proxy from router probs for the last time step
                last = probs_full[:, -1, :] if probs_full.dim() == 3 else probs_full
                p = torch.clamp(last, min=1e-9)
                ent_t = -torch.sum(p * p.log(), dim=-1).mean()
                # Build dtype-local scalars fresh each call to avoid stale graph-pool storages
                # under CUDA Graphs; do NOT cache tensors on the module during capture.
                # IMPORTANT: Use aten-only ops (no .detach/new_tensor) to construct scalars
                # anchored to p's lineage and dtype/device.
                _zero_anchor = torch.ops.aten.mul.Scalar(
                    torch.ops.aten.sum.default(torch.ops.aten.slice.Tensor(p, -1, 0, 1, 1)), 0.0
                )
                tmin_t = torch.ops.aten.add.Scalar(_zero_anchor, float(tmin))
                tmax_t = torch.ops.aten.add.Scalar(_zero_anchor, float(tmax))
                lam_t  = torch.ops.aten.add.Scalar(_zero_anchor, float(lam))
                # Compute T_t = tmin + (tmax - tmin) * sigmoid(lam * entropy) (aten-only)
                _diffT = torch.ops.aten.sub.Tensor(tmax_t, tmin_t)
                _z = torch.ops.aten.mul.Tensor(lam_t, ent_t)
                _sig = torch.ops.aten.sigmoid.default(_z)
                T_t = torch.ops.aten.add.Tensor(tmin_t, torch.ops.aten.mul.Tensor(_diffT, _sig))
                # Apply stabilized softmax to adjusted scores (aten-only)
                adj = torch.ops.aten.div.Tensor(scores, torch.ops.aten.clamp_min.default(T_t, 1e-6))
                mx3 = torch.ops.aten.amax.default(adj, [-1], True)
                scores = torch.ops.aten.softmax.int(torch.ops.aten.sub.Tensor(adj, mx3), -1)
                # IMPORTANT: Do not widen top_k dynamically; keep K static for CUDA Graph stability.
        except Exception:
            pass

        # Enforce top_k cap defensively to avoid accidental widening (branch-free on dynamic shapes)
        try:
            k_cap = self.top_k
        except Exception:
            k_cap = 1
        # Inference-time: use static K_top in graph while still evaluating multiple experts via fixed-width blend
        # Keep k_cap equal to configured top_k to preserve expert diversity downstream in fused path.
        # Downstream packing stays static because capacity is fixed and K is constant per model.
        if (not self.training):
            k_cap = int(self.top_k)
        # Avoid Python slicing in hot path; use aten.slice on last dim (K)
        idx = torch.ops.aten.slice.Tensor(idx, -1, 0, k_cap, 1)
        scores = torch.ops.aten.slice.Tensor(scores, -1, 0, k_cap, 1)
        # Flatten batch/time for simpler indexing
        # Verbose debug logging for crash triage (enable via OMNICODER_MOE_DEBUG=1)
        _dbg = self._dbg
        _logp = self._logp
        # removed hot-path logging/timing
        # Flatten across batch and time for dispatch; preserve batch for the final reshape
        x_flat = torch.ops.aten.reshape.default(x, (torch.ops.aten.sym_size.int(x, 0) * seq_len, hidden_dim))
        # Note: downstream dispatch paths are designed to handle zero-length tensors as no-ops safely.
        # We avoid any Python or aten-based boolean guards here to keep graphs branch-free and export-safe.
        expert_indices_per_token = idx.reshape(torch.ops.aten.sym_size.int(x, 0) * seq_len, -1)
        # Sanitize gate scores and normalize per token
        expert_scores_sanitized = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        expert_scores_per_token = expert_scores_sanitized / expert_scores_sanitized.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scores_flat = expert_scores_per_token.reshape(torch.ops.aten.sym_size.int(x, 0) * seq_len, -1)
        # hot path log removed
        # removed hot-path logging
        # Defer output buffer allocation; reuse dispatch output directly

        # Capacity-aware per-expert dispatch (token-capacity per expert)
        load_penalty = x.new_zeros(())
        total_tokens = batch_size * seq_len
        # NOTE [Constant capacity for CUDA Graph stability]:
        # - Dynamic capacity introduces per-step variation in packed buffer sizes and temporary tensors.
        # - We select a constant capacity per expert equal to max(top_k, min_capacity_from_env),
        #   ignoring dynamic tokens/expected_ceiled to keep capture/replay identical.
        top_k_py = self.top_k
        min_from_env = int(self._min_capacity_from_env)
        # Inference-time: fix capacity to a constant derived from top_k to keep packed shapes static (preserve diversity)
        if (not self.training):
            try:
                _tk = int(self.top_k)
            except Exception:
                _tk = 1
            # Minimal capacity that admits all top_k per token without drops
            capacity = max(1, _tk)
        else:
            _d2 = top_k_py - min_from_env
            _s2 = (_d2 >> 31)
            _absd2 = (_d2 ^ _s2) - _s2
            capacity = (top_k_py + min_from_env + _absd2) // 2
        # removed logging
        # Grouped token-wise batched dispatch per expert
        # Try fused dispatch first (will fallback to torch ops)
        # Dispatch wrapper supporting sub-experts and shared general experts
        def _call_bank(b: nn.Module, xin: torch.Tensor) -> torch.Tensor:
            _is_list = False
            try:
                _ = b[0]  # type: ignore[index]
                _is_list = True
            except Exception:
                _is_list = False
            if _is_list:
                # Lightweight per-token sub-expert selector (no extra parameters):
                # use a simple hash over token index to pick one sub-expert deterministically.
                # This avoids overhead of extra gating while enabling diversity.
                try:
                    # Xin shape (M, C); generate indices 0..sub_experts_per-1
                    m = int(len(b))
                    # Use a single per-dtype arange buffer to avoid unbounded key growth and memory leak
                    dev_key = str(xin.dtype)
                    need = int(torch.ops.aten.sym_size.int(xin, 0))
                    # Avoid persistent caches under CUDA graphs: build per-call
                    idx_base = torch.ops.aten.new_ones.default(xin, (need,), dtype=torch.long)
                    idx_base = torch.ops.aten.cumsum.default(idx_base, 0)
                    idx_base = torch.ops.aten.sub.Tensor(idx_base, torch.ops.aten.new_ones.default(idx_base, idx_base.shape, dtype=idx_base.dtype))
                    idx_local = idx_base.remainder(max(1, m))
                    # Group indices per sub-expert to minimize kernel launches
                    y = torch.empty_like(xin)
                    for sid in range(m):
                        sel = (idx_local == sid)
                        # Branchless: compute sub-output and scatter via masked add
                        _sub_out = b[sid](xin[sel])
                        # Materialize destination slice and scatter; aten indexing handles empty masks efficiently
                        y[sel] = _sub_out
                    return y
                except Exception:
                    outs = [sub(xin) for sub in b]
                    return torch.stack(outs, dim=0).mean(dim=0)
            # Run directly; do not move devices in hot path
            return b(xin)
        # Unified path: remove pager conditionals from hot path; use direct expert modules
        expert_wrappers = None
        # Do not move devices in the hot path. Assume callers moved the module via .to(device) at setup.
        # Record the current input device to prevent redundant checks on subsequent calls.
        # Drop tracking input device; all compilation is handled at init time (no hot-path recompiles)
        # Removed prepack from hot path: banks are prepared during module initialization/move.
        # Avoid Python scalar extraction in normal runs; include detailed stats only when debug enabled
        # Vectorized decode-time expert fast path (inference-only):
        # - Preserves diversity by evaluating exactly top_k per token in a constant-width path
        # - Uses prepacked banks and two-stage BMMs, then weighted sum with gate scores (aten-only)
        # - Shapes are static: K = top_k, capacity fixed above
        _force_torch = self._force_torch_dispatch
        # Use a local output buffer; avoid forward-time module state mutation under CUDA Graphs
        ybuf = torch.ops.aten.new_zeros.default(x_flat, (torch.ops.aten.sym_size.int(x_flat, 0), torch.ops.aten.sym_size.int(x_flat, 1)))
        # HOT-LOG tensor (1xK slots) carried through to kernels; K=8 for this layer
        # Allocate via like-factory anchored to x_flat to avoid device/dtype moves in hot path
        hotlog = None
        # Prefer prepacked banks (built once in __init__) for forward/recompute parity and zero per-call cost.
        _W1p = self._prepacked_W1
        _B1p = self._prepacked_B1
        _W2p = self._prepacked_W2
        _B2p = self._prepacked_B2
        # Single, deterministic path: require prepacked banks (built at init/_apply)
        _ = _W1p.shape  # type: ignore[attr-defined]
        _ = _B1p.shape  # type: ignore[attr-defined]
        _ = _W2p.shape  # type: ignore[attr-defined]
        _ = _B2p.shape  # type: ignore[attr-defined]
        _banks = {'W1': _W1p, 'B1': _B1p, 'W2': _W2p, 'B2': _B2p}
        # Provide constant expert count to dispatcher to avoid dynamic new_dynamic_size
        try:
            _banks['E_const'] = int(self.n_experts)  # type: ignore[index]
        except Exception:
            pass
        # PERF/CG: Pre-expand expert biases once for cap=top_k to eliminate repeated atan.expand in the hot path.
        # These expanded buffers are non-persistent (not saved) and optional — fused_dispatch will expand on the fly
        # if they are not available. This keeps numerics identical while improving stability and TPS.
        try:
            _E = torch.ops.aten.sym_size.int(_W1p, 0)
            _cap = int(self.top_k)
            _M = torch.ops.aten.sym_size.int(_B1p, 1)
            _H = torch.ops.aten.sym_size.int(_B2p, 1)
            # Build expanded biases anchored to existing weights for device/dtype parity
            _b1 = torch.ops.aten.reshape.default(_B1p, (_E, 1, _M))
            _b2 = torch.ops.aten.reshape.default(_B2p, (_E, 1, _H))
            _B1e = torch.ops.aten.expand.default(_b1, (_E, _cap, _M))
            _B2e = torch.ops.aten.expand.default(_b2, (_E, _cap, _H))
            _banks['B1e'] = _B1e
            _banks['B2e'] = _B2e
        except Exception:
            pass
        output_fused = x_flat
        kept = None
        # Removed Python int() cast on symbolic seq_len to avoid unbacked symints during compile
        # Blended-gating constant-shape vectorized path (inference, decode only):
        # - Fix K_blend = num_blend_routers * top_k
        # - Take top_k from each router (no dedup), weight by fixed blend w, normalize, then compute
        #   per-token expert MLPs for all K_blend in parallel via two bmm ops.
        # - Shapes are constant across steps; suitable for CUDA-graph capture.
        # Deterministic dispatch path (always fused dispatch with fixed capacity=top_k).
        capacity = self.top_k
        output_fused, kept = fused_dispatch(
            x_flat,
            expert_indices_per_token,
            scores_flat,
            expert_wrappers,
            capacity,
            output_buf=ybuf,
            banks=_banks,
            hotlog=None,
            work_x=getattr(self, '_work_x_cache', None),
            work_w=getattr(self, '_work_w_cache', None),
        )
        # Guard against any unexpected earlier exceptions; ensure output_fused is defined
        # removed hot-path logging
        try:
            output_fused = output_fused
        except Exception:
            try:
                output_fused = x_flat
            except Exception:
                pass
        if (self.shared is not None) and (self.num_shared_general > 0) and self.training:
            try:
                # Average shared experts outputs without Python len() casts
                share_out = sum([g(x_flat) for g in self.shared]) / self.num_shared_general
                output_fused = 0.95 * output_fused + 0.05 * share_out
            except Exception:
                pass
        output_flat = output_fused
        # removed hot-path logging

        # SCMoE contrast path removed in forward to keep checkpoint recomputation identical (no stochastic masking).

        # z-loss style load balancing proxy (encourage uniform routing) — training-only
        if self.training:
            importance = torch.ops.aten.mean.dim(probs_full, [0, 1], False)  # (E,)
            # Build safe 1/E without Python max/min and anchor to tensor lineage
            try:
                _E_py = self.n_experts
            except Exception:
                _E_py = 1
            _dE = _E_py - 1
            _sE = (_dE >> 31)
            _absdE = (_dE ^ _sE) - _sE
            _E_safe = (_E_py + 1 + _absdE) // 2
            _zero_imp = torch.ops.aten.mul.Scalar(torch.ops.aten.sum.default(torch.ops.aten.slice.Tensor(importance, -1, 0, 1, 1)), 0.0)
            _E_t = torch.ops.aten.add.Scalar(_zero_imp, float(_E_safe))
            _E_t = torch.ops.aten.clamp_min.default(_E_t, 1.0)
            invE = torch.ops.aten.reciprocal.default(_E_t)
            ones_imp = torch.ops.aten.new_ones.default(importance, importance.shape, dtype=importance.dtype)
            uniform = torch.ops.aten.mul.Tensor(ones_imp, invE)
            diff = torch.ops.aten.sub.Tensor(importance, uniform)
            load_penalty = torch.ops.aten.sum.default(torch.ops.aten.pow.Tensor_Scalar(diff, 2.0))
            p = torch.ops.aten.clamp.default(importance, 1e-9)
            ent = torch.ops.aten.neg.default(torch.ops.aten.sum.default(torch.ops.aten.mul.Tensor(p, torch.ops.aten.log.default(p))))
            load_penalty = torch.ops.aten.sub.Tensor(load_penalty, torch.ops.aten.mul.Scalar(ent, 0.01))

        # NOTE [No forward-time tensor persistence]: avoid storing tensors on module in hot path (CG safe)
        # FIX: prefer torch.reshape to minimize FX call_method targets under Inductor
        # Return with correct batch dimension to avoid broadcast/dynamic re-trace
        y = torch.reshape(output_flat, (batch_size, seq_len, hidden_dim))
        # Minimal aten-only anchors: bind symbolic sizes (E, K, capacity, H) to output
        try:
            _z_moe = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(y, 0.0))
            # Expert count E from prepacked bank if available
            try:
                _W1p0 = self._prepacked_W1  # type: ignore[attr-defined]
                _E_sym = torch.ops.aten.sym_size.int(_W1p0, 0)
            except Exception:
                _E_sym = torch.ops.aten.sym_size.int(x_flat, 0)  # safe fallback
            _e_buf = torch.ops.aten.new_zeros.default(y, (_E_sym,))
            _e_anc = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(_e_buf, 0.0))
            # Router K from idx tensor second dim
            _K_sym = torch.ops.aten.sym_size.int(expert_indices_per_token, 1)
            _k_buf = torch.ops.aten.new_zeros.default(expert_indices_per_token, (_K_sym,))
            _k_anc = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(_k_buf, 0.0))
            # Capacity anchor (scalar)
            _cap0 = torch.ops.aten.mul.Scalar(_z_moe, 0.0)
            _cap_anc = torch.ops.aten.add.Scalar(_cap0, float(capacity))
            # Hidden dim H
            _H_sym = torch.ops.aten.sym_size.int(y, 2)
            _h_buf = torch.ops.aten.new_zeros.default(y, (_H_sym,))
            _h_anc = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(_h_buf, 0.0))
            # Flatten length N = B*T used inside fused path
            _N_sym = torch.ops.aten.sym_size.int(x_flat, 0)
            _n_buf = torch.ops.aten.new_zeros.default(x_flat, (_N_sym,))
            _n_anc = torch.ops.aten.sum.default(torch.ops.aten.mul.Scalar(_n_buf, 0.0))
            _anc = torch.ops.aten.add.Tensor(_z_moe, _e_anc)
            _anc = torch.ops.aten.add.Tensor(_anc, _k_anc)
            _anc = torch.ops.aten.add.Tensor(_anc, _cap_anc)
            _anc = torch.ops.aten.add.Tensor(_anc, _h_anc)
            _anc = torch.ops.aten.add.Tensor(_anc, _n_anc)
            y = torch.ops.aten.add.Tensor(y, torch.ops.aten.mul.Scalar(_anc, 0.0))
        except Exception:
            pass
        # removed hot-path logging
        return y

    def _apply(self, fn):
        """Ensure prepacked banks track module device/dtype without hot-path checks.

        Called by nn.Module.to()/cuda()/cpu(); we rebuild banks once here so that
        forward never inspects devices or moves tensors.
        """
        out = super()._apply(fn)
        try:
            p = next(self.parameters())
            # Device-agnostic prepack; only dtype is honored
            self._prepack_banks_for_dtype(dtype=p.dtype)  # type: ignore[arg-type]
            # Preallocate fixed-size work buffers for fused_dispatch using constant capacity
            try:
                cap_est = int(self.top_k)
            except Exception:
                cap_est = 1
            try:
                min_est = int(self._min_capacity_from_env)
            except Exception:
                min_est = cap_est
            _d = cap_est - min_est
            _s = (_d >> 31)
            _absd = (_d ^ _s) - _s
            cap_fixed = (cap_est + min_est + _absd) // 2
            EC0 = int(self.n_experts) * int(cap_fixed)
            H0 = int(self._d_model) if isinstance(self._d_model, int) else self._d_model  # type: ignore[assignment]
            try:
                self.register_buffer('_work_x_cache', p.new_zeros((EC0, H0)), persistent=False)
            except Exception:
                try:
                    self._work_x_cache = p.new_zeros((EC0, H0))  # type: ignore[assignment]
                except Exception:
                    self._work_x_cache = None  # type: ignore[assignment]
            try:
                self.register_buffer('_work_w_cache', p.new_zeros((EC0, 1)), persistent=False)
            except Exception:
                try:
                    self._work_w_cache = p.new_zeros((EC0, 1))  # type: ignore[assignment]
                except Exception:
                    self._work_w_cache = None  # type: ignore[assignment]
            # Reset y buffer cache to force size check on next forward
            try:
                self._y_buf_cache = None  # type: ignore[assignment]
            except Exception:
                pass
        except Exception:
            pass
        return out

    def _prepack_banks_for_dtype(self, dtype: torch.dtype) -> None:
        """Precompute and cache per-expert weight banks on the target dtype only.

        Mirrors the fused_dispatch bank builder so first decode steps avoid packing cost.
        Safe no-op when experts are not simple FFNs.
        """
        try:
            from torch import nn as _nn
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
                        import weakref as _wr  # noqa: F401
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