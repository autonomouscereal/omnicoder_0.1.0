import os
from typing import Optional


class TextTokenizer:
    """
    Minimal placeholder tokenizer to keep demos runnable without external deps.
    This does NOT implement a real BPE. It maps characters to ids in a tiny range
    and back, only for smoke tests of the generation loop.
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Reserve multimodal slices if config defines them (no strict enforcement here; docs expose ranges)
        try:
            from omnicoder.config import MultiModalConfig  # type: ignore
            mmc = MultiModalConfig()
            self.v_img_start = mmc.image_vocab_start
            self.v_vid_start = mmc.video_vocab_start
            self.v_aud_start = mmc.audio_vocab_start
        except Exception:
            self.v_img_start = None
            self.v_vid_start = None
            self.v_aud_start = None
        # Reserve 0 for padding, 1 for unknown, rest map from basic ASCII
        self.offset = 2

    def encode(self, text: str):
        ids = []
        for ch in text:
            code = ord(ch)
            if 32 <= code < 127:  # printable ASCII
                token = (code - 32) + self.offset
            else:
                token = 1  # unknown
            ids.append(token)
        if not ids:
            ids = [1]
        return ids

    def decode(self, ids):
        chars = []
        for token in ids:
            if token >= self.offset:
                code = (token - self.offset) + 32
                if 32 <= code < 127:
                    chars.append(chr(code))
                else:
                    chars.append('?')
            elif token == 0:
                chars.append(' ')
            else:
                chars.append('?')
        return ''.join(chars)


class AutoTokenizerWrapper:
    """Wrap HuggingFace AutoTokenizer to our minimal interface if available."""

    def __init__(self, model_name: str = "gpt2"):
        from transformers import AutoTokenizer  # type: ignore
        import os as _os
        # Prefer authenticated access when token provided; allow local path
        is_local = os.path.isdir(model_name)
        token = _os.getenv("HF_TOKEN", None)
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                token=token,
                use_auth_token=None,
                local_files_only=bool(is_local),
                trust_remote_code=True,
            )
        except TypeError:
            # transformers without 'token' kwarg
            tok = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=token,
                local_files_only=bool(is_local),
                trust_remote_code=True,
            )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tok = tok
        # record special ids if present
        self.bos_token_id: Optional[int] = getattr(tok, "bos_token_id", None)
        self.eos_token_id: Optional[int] = getattr(tok, "eos_token_id", None)

    def encode(self, text: str):
        try:
            ids = self._tok.encode(text, add_special_tokens=False)
            return list(map(int, ids))
        except Exception:
            # Fallback to call-style encoding API
            return self._tok(text, add_special_tokens=False).input_ids  # type: ignore[attr-defined]

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=True)


class HFJsonTokenizerWrapper:
    """
    Load a fast tokenizer directly from tokenizer.json via huggingface_hub+tokenizers,
    avoiding importing transformers (works around torch._dynamo import issues).

    - Accepts either a local directory containing tokenizer.json or an HF repo id
    - Requires 'tokenizers' and 'huggingface_hub' only
    """

    def __init__(self, model_name: str):
        import os as _os
        from pathlib import Path as _P
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            from tokenizers import Tokenizer as _Tk  # type: ignore
        except Exception as _e:
            raise RuntimeError(f"HFJsonTokenizerWrapper deps missing: {_e}")
        token_file: str | None = None
        # Prefer local tokenizer.json if model_name is a directory
        if _os.path.isdir(model_name):
            cand = _P(model_name) / "tokenizer.json"
            if cand.exists():
                token_file = str(cand)
        # Otherwise, fetch from hub
        if token_file is None:
            try:
                token_file = hf_hub_download(repo_id=model_name, filename="tokenizer.json", token=_os.getenv("HF_TOKEN"))
            except Exception as _e:
                raise RuntimeError(f"failed to fetch tokenizer.json for {model_name}: {_e}")
        try:
            self._tok = _Tk.from_file(token_file)
        except Exception as _e:
            raise RuntimeError(f"failed to load tokenizer.json: {_e}")
        # Try to expose common attributes
        try:
            self.vocab_size = int(self._tok.get_vocab_size())  # type: ignore[attr-defined]
        except Exception:
            self.vocab_size = None  # type: ignore[assignment]
        # Resolve special token ids when present
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        try:
            for name in ("<s>", "<bos>", "<BOS>"):
                tid = self._tok.token_to_id(name)
                if isinstance(tid, int) and tid >= 0:
                    self.bos_token_id = tid
                    break
        except Exception:
            pass
        try:
            for name in ("</s>", "<eos>", "<EOS>"):
                tid = self._tok.token_to_id(name)
                if isinstance(tid, int) and tid >= 0:
                    self.eos_token_id = tid
                    break
        except Exception:
            pass

    def encode(self, text: str):
        try:
            return list(map(int, self._tok.encode(text).ids))
        except Exception:
            return []

    def decode(self, ids):
        try:
            return self._tok.decode(ids)
        except Exception:
            return ""

class SubsetTokenizerWrapper:
    """
    Wrap a base HF tokenizer and restrict ids to [0, allowed_vocab).

    Intended for cases where a student model was trained on a 32k slice from a larger
    tokenizer (e.g., Meta-Llama-3 128k). Any out-of-range ids on encode are mapped to a
    safe replacement within range (unk/eos/last_id). Decode delegates to the base tokenizer
    so textualization stays faithful to the original vocabulary semantics for in-range ids.
    """

    def __init__(self, base_tok: AutoTokenizerWrapper, allowed_vocab: int = 32000):
        self.base = base_tok
        self.allowed = int(max(2, allowed_vocab))
        # Report a 32k-like surface to downstream code
        self.vocab_size = self.allowed
        # Forward special ids when they lie within range; else clamp
        self.bos_token_id: Optional[int] = getattr(base_tok, "bos_token_id", None)
        self.eos_token_id: Optional[int] = getattr(base_tok, "eos_token_id", None)
        try:
            if isinstance(self.bos_token_id, int) and not (0 <= int(self.bos_token_id) < self.allowed):
                self.bos_token_id = self.allowed - 1
        except Exception:
            self.bos_token_id = None
        try:
            if isinstance(self.eos_token_id, int) and not (0 <= int(self.eos_token_id) < self.allowed):
                self.eos_token_id = self.allowed - 1
        except Exception:
            self.eos_token_id = None
        # Name hint for diagnostics
        self.name = "llama3_subset"

    def encode(self, text: str):
        ids = self.base.encode(text)
        # Choose a safe replacement id within range
        try:
            unk_id = getattr(self.base._tok, "unk_token_id", None)  # type: ignore[attr-defined]
        except Exception:
            unk_id = None
        eos_id = self.eos_token_id
        repl = None
        if isinstance(unk_id, int) and 0 <= int(unk_id) < self.allowed:
            repl = int(unk_id)
        elif isinstance(eos_id, int) and 0 <= int(eos_id) < self.allowed:
            repl = int(eos_id)
        else:
            repl = self.allowed - 1
        return [int(i) if (0 <= int(i) < self.allowed) else int(repl) for i in ids]

    def decode(self, ids):
        # Delegate to base tokenizer for textualization
        return self.base.decode(ids)

def get_text_tokenizer(prefer_hf: bool = True, hf_id: Optional[str] = None):
    """Return a tokenizer instance.

    - If prefer_hf is True, try to load HF tokenizer from `hf_id` argument or
      from environment variable `OMNICODER_HF_TOKENIZER`. Falls back to 'gpt2'.
    - If HF loading fails or prefer_hf is False, return the simple ASCII tokenizer.
    """
    from omnicoder.utils.logger import get_logger
    _log = get_logger("omnicoder.tokenizer")
    # Absolute override: when set, always use the simple 32k tokenizer regardless of arguments/env
    try:
        if os.getenv("OMNICODER_FORCE_SIMPLE_TOKENIZER", "0") == "1":
            _log.info("get_text_tokenizer: OMNICODER_FORCE_SIMPLE_TOKENIZER=1 → using simple TextTokenizer (vocab=32000)")
            return TextTokenizer(vocab_size=32000)
    except Exception:
        pass
    if prefer_hf:
        # Build a robust candidate list
        candidates: list[str] = []
        if hf_id and str(hf_id).strip():
            candidates.append(str(hf_id).strip())
        env_id = os.environ.get("OMNICODER_HF_TOKENIZER", "").strip()
        if env_id:
            candidates.append(env_id)
        # If explicitly requested, try a Llama-3 subset adapter first (local or HF id)
        try:
            want_l3_subset = os.getenv("OMNICODER_LLAMA3_SUBSET", "0") == "1"
        except Exception:
            want_l3_subset = False
        # Probe unified vocab to steer mapping
        text_size = 0
        try:
            import json as _json
            from pathlib import Path as _P
            um = _P("/workspace/weights/release/unified_vocab_map.json")
            if um.exists():
                meta = _json.loads(um.read_text(encoding='utf-8'))
                text_size = int(meta.get("text_size", 0)) if isinstance(meta, dict) else 0
        except Exception:
            text_size = 0
        # Prefer subset wrapper when student text vocab is 32k and we have any hint of Llama-3
        if text_size == 32000 and (want_l3_subset or any("meta-llama" in s.lower() for s in [env_id or "", str(hf_id or ""), os.environ.get("OMNICODER_API_TEACHER_PATH", "")] )):
            # Try local teacher path first
            local_tok = os.environ.get("OMNICODER_HF_TOKENIZER_LOCAL", "").strip() or os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
            try:
                if local_tok and os.path.isdir(local_tok):
                    base = AutoTokenizerWrapper(local_tok)
                    return SubsetTokenizerWrapper(base, allowed_vocab=32000)
            except Exception:
                pass
            # Then try explicit hf_id or known Llama-3 id
            try:
                target = env_id or hf_id or "meta-llama/Meta-Llama-3-8B-Instruct"
                if target:
                    base = AutoTokenizerWrapper(str(target))
                    return SubsetTokenizerWrapper(base, allowed_vocab=32000)
            except Exception:
                # fall back to generic flow below
                pass
        # If a local teacher/tokenizer path exists, try it first
        local_tok = os.environ.get("OMNICODER_HF_TOKENIZER_LOCAL", "").strip() or os.environ.get("OMNICODER_API_TEACHER_PATH", "").strip()
        if local_tok and os.path.isdir(local_tok):
            # Only add if tokenizer files likely present
            if any(os.path.exists(os.path.join(local_tok, fn)) for fn in ("tokenizer.json", "tokenizer.model", "vocab.json")):
                candidates.insert(0, local_tok)
        # Optional: If text vocab is 32k and env points to Llama 3 (128k), prefer an open LLaMA tokenizer
        # Normally can be disabled with OMNICODER_DISABLE_TOKENIZER_REMAP=1, but when
        # OMNICODER_LLAMA3_SUBSET=1 we still want to try a compatible 32k tokenizer.
        disable_remap = os.getenv("OMNICODER_DISABLE_TOKENIZER_REMAP", "0") == "1"
        if text_size == 32000 and (not disable_remap or want_l3_subset):
            if any("meta-llama" in c.lower() for c in candidates):
                candidates.append("hf-internal-testing/llama-tokenizer")
            # Known accessible LLaMA-family tokenizers
            candidates.append("hf-internal-testing/llama-tokenizer")
            candidates.append("openlm-research/open_llama_7b")
        # Control inclusion of GPT-2 fallback explicitly. Using GPT-2 with a 32k student model
        # leads to a hard tokenizer<->model vocab mismatch and degenerate output.
        forbid_gpt2 = os.getenv("OMNICODER_FORBID_GPT2", "0") == "1" or text_size == 32000
        if not forbid_gpt2:
            # Always include a widely available fallback when not forbidden
            candidates.append("gpt2")
        # Deduplicate preserving order
        seen = set()
        tried_errors: list[str] = []
        ordered: list[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        _log.info("get_text_tokenizer: candidates_ordered=%s", ordered)
        for cand in ordered:
            # First try direct tokenizer.json path (no transformers import)
            try:
                _log.info("get_text_tokenizer: trying HF-json tokenizer id=%s", cand)
                tok_j = HFJsonTokenizerWrapper(cand)
                try:
                    vs = int(getattr(tok_j, "vocab_size", 0) or 0)
                    if vs:
                        setattr(tok_j, "vocab_size", vs)
                except Exception:
                    pass
                _log.info("get_text_tokenizer: HF-json ok id=%s", cand)
                return tok_j
            except Exception as _je:
                tried_errors.append(f"json:{cand}: {str(_je)}")
                try:
                    _log.error("get_text_tokenizer: HF-json failed id=%s err=%s", cand, str(_je))
                except Exception:
                    pass
            # Then try transformers AutoTokenizer as a fallback
            try:
                _log.info("get_text_tokenizer: trying HF tokenizer id=%s", cand)
                tok = AutoTokenizerWrapper(cand)
                try:
                    vs = int(getattr(tok._tok, "vocab_size", 0))  # type: ignore[attr-defined]
                    if vs:
                        setattr(tok, "vocab_size", vs)
                except Exception:
                    pass
                _log.info("get_text_tokenizer: HF ok id=%s", cand)
                return tok
            except Exception as _e:
                err = f"{cand}: {str(_e)}"
                tried_errors.append(err)
                try:
                    # Downgrade to WARNING to avoid alarming logs in offline environments; we fall back cleanly
                    _log.warning("get_text_tokenizer: HF failed id=%s err=%s", cand, str(_e))
                except Exception:
                    pass
        # If we get here, all HF candidates failed.
    try:
        if prefer_hf and tried_errors:
            _log.error("get_text_tokenizer: all HF candidates failed; errors=%s", "; ".join(tried_errors[:4]))
        # Respect global forbid flag: prefer byte-level universal tokenizer over simple text
        if os.getenv("OMNICODER_FORBID_SIMPLE", "1") == "1":
            _log.info("get_text_tokenizer: OMNICODER_FORBID_SIMPLE=1 → using ByteTokenizer (universal)")
            return ByteTokenizer()
        _log.info("get_text_tokenizer: using simple TextTokenizer (vocab=32000)")
    except Exception:
        pass
    # Default to 32k only when simple is allowed
    return TextTokenizer(vocab_size=32000)


class ByteTokenizer:
    """
    Simple byte-level tokenizer (ByT5-style surrogate).

    - Maps each input byte [0..255] to an id in [2..257]; 0=pad, 1=unk
    - Encodes UTF-8 bytes directly for robustness to OOD/typos
    """

    def __init__(self):
        self.vocab_size = 258
        self.offset = 2

    def encode(self, text: str):
        b = text.encode('utf-8', errors='replace')
        if not b:
            return [1]
        return [int(x) + self.offset for x in b]

    def decode(self, ids):
        buf = bytearray()
        for t in ids:
            if t >= self.offset:
                buf.append(int(t - self.offset) & 0xFF)
            # ignore pad/unk on decode
        try:
            return buf.decode('utf-8', errors='replace')
        except Exception:
            return ''


class CompositeTokenizer:
    """
    Universal tokenizer that combines an HF text tokenizer with a byte-level fallback.

    Behavior:
    - Prefer HF tokenization when available.
    - If HF tokenization fails (exception) or returns a non-list, fallback to byte-level.
    - Decoding uses HF when possible; otherwise bytes.
    """
    def __init__(self, token_tok, byte_tok):
        self.token_tok = token_tok
        self.byte_tok = byte_tok
        # Expose a vocab_size attribute for downstream checks when present
        self.vocab_size = int(getattr(token_tok, 'vocab_size', 0) or 32000)

    def encode(self, text: str) -> list[int]:
        try:
            ids = self.token_tok.encode(text)
            if isinstance(ids, list) and ids:
                return ids
        except Exception:
            pass
        return self.byte_tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        try:
            return self.token_tok.decode(ids)
        except Exception:
            return self.byte_tok.decode(ids)


def get_dual_tokenizers(prefer_hf: bool = True, hf_id: Optional[str] = None):
    """Return (token_tokenizer, byte_tokenizer)."""
    return get_text_tokenizer(prefer_hf=prefer_hf, hf_id=hf_id), ByteTokenizer()


def get_universal_tokenizer(prefer_hf: bool = True, hf_id: Optional[str] = None):
    """Return a CompositeTokenizer that combines HF and byte-level tokenization."""
    tok, byt = get_dual_tokenizers(prefer_hf=prefer_hf, hf_id=hf_id)
    return CompositeTokenizer(tok, byt)


