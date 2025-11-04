import torch, torch.nn as nn
import torch.nn.functional as F
from omnicoder.utils.logger import get_logger

class Planner(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.policy = nn.Linear(d_model, d_model)  # produce 'intent' vector

    def forward(self, state, hidden: torch.Tensor | None = None):
        # Provide explicit h0 for ONNX export when hidden is missing (avoid None attribute access)
        if hidden is None:
            b = torch.ops.aten.sym_size.int(state, 0)
            d = torch.ops.aten.sym_size.int(state, state.dim() - 1)
            hidden = torch.ops.aten.new_zeros.default(state, (1, b, d))
        # Ensure fixed batch size behavior by always supplying h0 (even if caller passes None)
        out, h = self.rnn(state, hidden)
        intent = self.policy(out[:, -1:, :])  # last step
        return intent, h

class Worker(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.act = nn.Linear(d_model, d_model)

    def forward(self, state, intent, hidden: torch.Tensor | None = None):
        # concatenate intent to each step
        T = torch.ops.aten.sym_size.int(state, 1)
        intent_rep = intent.expand(-1, T, -1)
        # Provide explicit h0 for ONNX export when hidden is missing (avoid None attribute access)
        if hidden is None:
            b = torch.ops.aten.sym_size.int(state, 0)
            d = torch.ops.aten.sym_size.int(state, state.dim() - 1)
            hidden = torch.ops.aten.new_zeros.default(state, (1, b, d))
        # Ensure fixed batch size behavior by always supplying h0 (even if caller passes None)
        out, h = self.rnn(state + intent_rep, hidden)
        return self.act(out), h

class HRM(nn.Module):
    """Hierarchical Reasoning Module with optional adaptive halting.

    When adaptive_halting is enabled, uses a small halting head to predict per-step
    continuation probability and stops when accumulated halting probability crosses
    the threshold (similar to Adaptive Computation Time).
    """

    def __init__(
        self,
        d_model: int = 256,
        steps: int = 3,
        adaptive_halting: bool = False,
        halting_threshold: float = 0.99,
        max_steps_budget: int | None = None,
    ):
        super().__init__()
        # Logging disabled in hot path constructors to avoid I/O in model builds.
        self.planner = Planner(d_model)
        self.worker = Worker(d_model)
        self.steps = int(max(1, steps))
        self.adaptive_halting = bool(adaptive_halting)
        self.halting_threshold = float(halting_threshold)
        self.max_steps_budget = int(max_steps_budget) if max_steps_budget is not None else None
        if self.adaptive_halting:
            self.halt_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.halt_head = None
        # Logging disabled in hot path constructors
        # Persistent hidden state caches for decode-step reuse
        self._planner_h: torch.Tensor | None = None
        self._worker_h: torch.Tensor | None = None
        self._last_bsz: int | None = None
        # Export-only injection points: allow wrappers to feed initial hidden states as model inputs
        # to satisfy ONNX RNN symbolic expectations without disabling features or adding env gating.
        self._export_h0_planner: torch.Tensor | None = None
        self._export_h0_worker: torch.Tensor | None = None

    def reset_state(self) -> None:
        self._planner_h = None
        self._worker_h = None
        self._last_bsz = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Logging disabled in hot path forward (no I/O in captured regions)
        # Stateless forward for trace/export safety (no Python bools from tensor-derived values)

        # Unified fixed-steps execution (vectorized); adaptive halting via masking without Python conditionals on tensors
        steps_limit = self.steps
        if (self.max_steps_budget is not None) and (self.max_steps_budget > 0):
            steps_limit = min(steps_limit, int(self.max_steps_budget))

        if not self.adaptive_halting:
            # Fixed-step execution with NO module-state mutation (CUDA Graph friendly)
            # If decode/export wrapper provided initial hidden states, use them for the first step
            h_p = self._export_h0_planner
            h_w = self._export_h0_worker
            for _ in range(steps_limit):
                intent, h_p = self.planner(state, h_p)
                state, h_w = self.worker(state, intent, h_w)
            # Clear export injection points to avoid unintended carry-over
            self._export_h0_planner = None
            self._export_h0_worker = None
            # Logging disabled in hot path forward
            return state

        # Adaptive halting via arithmetic mask: continue updates where accumulated<threshold
        # Build accumulated with aten.slice instead of Python slicing: state[..., :1]
        _slice_lastdim = torch.ops.aten.slice.Tensor(state, -1, 0, 1, 1)
        accumulated = torch.ops.aten.mul.Scalar(_slice_lastdim, 0.0)
        # Collapse time to last-step for halting signal
        for _ in range(steps_limit):
            # Use export-provided hidden only on the first step if present
            _hp_in = self._export_h0_planner
            _hw_in = self._export_h0_worker
            intent, _hp = self.planner(state, _hp_in)
            state_new, _hw = self.worker(state, intent, _hw_in)
            self._export_h0_planner = None
            self._export_h0_worker = None
            # Select last time step without Python slicing: select + unsqueeze
            _last_tok = torch.ops.aten.select.int(state_new, 1, -1)  # (B,C)
            last = torch.ops.aten.unsqueeze.default(_last_tok, 1)    # (B,1,C)
            p = torch.sigmoid(self.halt_head(last))  # type: ignore[arg-type]
            one = torch.ops.aten.add.Scalar(p, 1.0)
            one = torch.ops.aten.sub.Tensor(one, p)  # one constructed from p lineage
            accumulated = torch.ops.aten.add.Tensor(accumulated, torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(one, accumulated), p))
            # active mask = (accumulated < threshold)
            thr_t = torch.ops.aten.add.Scalar(accumulated, float(self.halting_threshold))
            thr_t = torch.ops.aten.sub.Tensor(thr_t, accumulated)  # thr as scalar tensor via lineage
            active = torch.ops.aten.lt.Tensor(accumulated, thr_t)
            active_f = torch.ops.aten.to.dtype(active, state.dtype, False, False)
            state = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(active_f, state_new), torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(torch.ops.aten.add.Scalar(active_f, 1.0), active_f), state))
        # Logging disabled in hot path forward
        return state
