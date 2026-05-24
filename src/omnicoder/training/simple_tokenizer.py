import os
from pathlib import Path
from typing import Optional, Any, List, Union
import torch

class TextTokenizer:
    """
    Minimal placeholder tokenizer to keep demos runnable without external deps.
    This does NOT implement a real BPE. It maps characters to ids in a tiny range
    and back, only for smoke tests of the generation loop.
    """
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Reserve multimodal slices if config defines them
        try:
            from omnicoder.config import MultiModalConfig  # type: ignore
            mmc = MultiModalConfig()
            self.v_img_start = mmc.image_vocab_start
            self.v_vid_start = mmc.video_vocab_start
            self.v_aud_start = mmc.audio_vocab_start
        except Exception:
            self.v_img_start = self.v_vid_start = self.v_aud_start = None
        self.offset = 2  # 0=pad, 1=unk

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            code = ord(ch)
            if 32 <= code < 127:
                token = (code - 32) + self.offset
            else:
                token = 1
            ids.append(token)
        if not ids:
            ids = [1]
        return ids

    def decode(self, ids: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        chars = []
        for token in ids:
            if isinstance(token, (list, tuple)):
                token = token[0] if token else 1
            if token >= self.offset:
                code = (token - self.offset) + 32
                chars.append(chr(code) if 32 <= code < 127 else '?')
            elif token == 0:
                chars.append(' ')
            else:
                chars.append('?')
        return ''.join(chars)


class AutoTokenizerWrapper:
    """Wrap HuggingFace AutoTokenizer."""
    def __init__(self, model_name: str = "gpt2"):
        from transformers import AutoTokenizer
        is_local = os.path.isdir(model_name)
        token = os.getenv("HF_TOKEN")
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                token=token,
                local_files_only=bool(is_local),
                trust_remote_code=True,
            )
        except TypeError:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=token,
                local_files_only=bool(is_local),
                trust_remote_code=True,
            )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tok = tok
        self.bos_token_id = getattr(tok, "bos_token_id", None)
        self.eos_token_id = getattr(tok, "eos_token_id", None)

    def encode(self, text: str) -> List[int]:
        try:
            ids = self._tok.encode(text, add_special_tokens=False)
            return list(map(int, ids))
        except Exception:
            return self._tok(text, add_special_tokens=False).input_ids

    def decode(self, ids: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return self._tok.decode(ids, skip_special_tokens=True)


class HFJsonTokenizerWrapper:
    """Load directly from tokenizer.json (avoids some transformers/dynamo issues)."""
    def __init__(self, model_name: str):
        from pathlib import Path
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except Exception as e:
            raise RuntimeError(f"HFJsonTokenizerWrapper deps missing: {e}")

        token_file = None
        if os.path.isdir(model_name):
            cand = Path(model_name) / "tokenizer.json"
            if cand.exists():
                token_file = str(cand)

        if token_file is None:
            try:
                token_file = hf_hub_download(repo_id=model_name, filename="tokenizer.json", token=os.getenv("HF_TOKEN"))
            except Exception as e:
                raise RuntimeError(f"failed to fetch tokenizer.json for {model_name}: {e}")

        self._tok = Tokenizer.from_file(token_file)
        try:
            self.vocab_size = int(self._tok.get_vocab_size())
        except Exception:
            self.vocab_size = 32000

        self.bos_token_id = self.eos_token_id = None
        for name in ("<s>", "<bos>", "<BOS>"):
            tid = self._tok.token_to_id(name)
            if isinstance(tid, int) and tid >= 0:
                self.bos_token_id = tid
                break
        for name in ("</s>", "<eos>", "<EOS>"):
            tid = self._tok.token_to_id(name)
            if isinstance(tid, int) and tid >= 0:
                self.eos_token_id = tid
                break

    def encode(self, text: str) -> List[int]:
        try:
            return list(map(int, self._tok.encode(text).ids))
        except Exception:
            return []

    def decode(self, ids: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        try:
            return self._tok.decode(ids)
        except Exception:
            return ""


class SubsetTokenizerWrapper:
    """Restrict to 32k vocab slice (e.g. Llama-3 student)."""
    def __init__(self, base_tok: Any, allowed_vocab: int = 32000):
        self.base = base_tok
        self.allowed = int(max(2, allowed_vocab))
        self.vocab_size = self.allowed
        self.bos_token_id = getattr(base_tok, "bos_token_id", None)
        self.eos_token_id = getattr(base_tok, "eos_token_id", None)
        if isinstance(self.bos_token_id, int) and not (0 <= self.bos_token_id < self.allowed):
            self.bos_token_id = self.allowed - 1
        if isinstance(self.eos_token_id, int) and not (0 <= self.eos_token_id < self.allowed):
            self.eos_token_id = self.allowed - 1
        self.name = "llama3_subset"

    def encode(self, text: str) -> List[int]:
        ids = self.base.encode(text)
        try:
            unk_id = getattr(self.base._tok, "unk_token_id", None) if hasattr(self.base, "_tok") else None
        except Exception:
            unk_id = None
        repl = unk_id if isinstance(unk_id, int) and 0 <= unk_id < self.allowed else (self.eos_token_id or self.allowed - 1)
        return [int(i) if 0 <= int(i) < self.allowed else int(repl) for i in ids]

    def decode(self, ids: Any) -> str:
        return self.base.decode(ids)


class ByteTokenizer:
    """Robust byte-level fallback."""
    def __init__(self):
        self.vocab_size = 258
        self.offset = 2

    def encode(self, text: str) -> List[int]:
        b = text.encode('utf-8', errors='replace')
        return [int(x) + self.offset for x in b] if b else [1]

    def decode(self, ids: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        # Full flatten
        flat = []
        for item in ids:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        buf = bytearray()
        for t in flat:
            if isinstance(t, (int, float)):
                t = int(t)
                if t >= self.offset:
                    buf.append((t - self.offset) & 0xFF)
        try:
            return buf.decode('utf-8', errors='replace')
        except Exception:
            return ''


class CompositeTokenizer:
    def __init__(self, token_tok, byte_tok):
        self.token_tok = token_tok
        self.byte_tok = byte_tok
        self.vocab_size = int(getattr(token_tok, 'vocab_size', 0) or 32000)

    def encode(self, text: str) -> List[int]:
        try:
            ids = self.token_tok.encode(text)
            if isinstance(ids, list) and ids:
                return ids
        except Exception:
            pass
        return self.byte_tok.encode(text)

    def decode(self, ids: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        try:
            return self.token_tok.decode(ids)
        except Exception:
            return self.byte_tok.decode(ids)


def get_text_tokenizer(prefer_hf: bool = True, hf_id: Optional[str] = None):
    from omnicoder.utils.logger import get_logger
    _log = get_logger("omnicoder.tokenizer")

    bpe_path = os.getenv("OMNICODER_BPE_PATH")
    if bpe_path and Path(bpe_path).exists():
        try:
            from omnicoder.tokenization.omni_tokenizer import OmniTokenizer
            _log.info(f"Using OmniTokenizer with trained BPE: {bpe_path}")
            return OmniTokenizer(vocab_size=128000, bpe_path=bpe_path)
        except Exception as e:
            _log.warning(f"Failed to load OmniTokenizer with BPE: {e}")

    # === Original fallback logic (unchanged) ===
    if os.getenv("OMNICODER_FORCE_SIMPLE_TOKENIZER", "0") == "1":
        _log.info("OMNICODER_FORCE_SIMPLE_TOKENIZER=1 → using simple TextTokenizer")
        return TextTokenizer(vocab_size=32000)

    if prefer_hf:
        candidates: list[str] = []
        if hf_id:
            candidates.append(str(hf_id))
        env_id = os.getenv("OMNICODER_HF_TOKENIZER", "").strip()
        if env_id:
            candidates.append(env_id)
        env_many = os.getenv("OMNICODER_TOKENIZER_CANDIDATES", "").strip()
        if env_many:
            candidates.extend([x.strip() for x in env_many.split(",") if x.strip()])
        candidates.extend([
            "Qwen/Qwen3-4B",
            "Qwen/Qwen2.5-7B",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "hf-internal-testing/llama-tokenizer",
        ])

        seen: set[str] = set()
        for model_name in candidates:
            if model_name in seen:
                continue
            seen.add(model_name)
            try:
                tok = HFJsonTokenizerWrapper(model_name)
                _log.info("Using HF tokenizer.json: %s", model_name)
                return tok
            except Exception as e:
                _log.debug("HFJsonTokenizerWrapper failed for %s: %s", model_name, e)
            try:
                tok = AutoTokenizerWrapper(model_name)
                _log.info("Using HF AutoTokenizer: %s", model_name)
                return tok
            except Exception as e:
                _log.debug("AutoTokenizerWrapper failed for %s: %s", model_name, e)

    # Final fallbacks
    if os.getenv("OMNICODER_FORBID_SIMPLE", "1") == "1":
        _log.info("OMNICODER_FORBID_SIMPLE=1 → using ByteTokenizer")
        return ByteTokenizer()

    _log.info("using simple TextTokenizer (vocab=32000)")
    return TextTokenizer(vocab_size=32000)


# Keep your helper functions unchanged
def get_dual_tokenizers(prefer_hf: bool = True, hf_id: Optional[str] = None):
    return get_text_tokenizer(prefer_hf=prefer_hf, hf_id=hf_id), ByteTokenizer()


def get_universal_tokenizer(prefer_hf: bool = True, hf_id: Optional[str] = None):
    tok, byt = get_dual_tokenizers(prefer_hf=prefer_hf, hf_id=hf_id)
    return CompositeTokenizer(tok, byt)
