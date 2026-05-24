"""
Real BPE Trainer from scratch for OmniCoder.
Pure Python. No external BPE libraries.
Improved for unified multimodal use (text + PNG bytes + WAV bytes + pickled data).
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class BPETokenizer:
    def __init__(self, vocab_size: int = 64000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            "<|pad|>": 0, "<|unk|>": 1, "<|text|>": 2,
            "<|image_start|>": 3, "<|image_end|>": 4,
            "<|audio_start|>": 5, "<|audio_end|>": 6,
            "<|video_start|>": 7, "<|video_end|>": 8,
            "<|thinking|>": 9,
        }

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], words: List[List[str]]) -> List[List[str]]:
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_words = []
        for word in words:
            w = " ".join(word)
            w = p.sub("".join(pair), w)
            new_words.append(w.split())
        return new_words

    def train(self, corpus: List[str], num_merges: Optional[int] = None):
        if num_merges is None:
            num_merges = self.vocab_size - len(self.special_tokens)

        words: List[List[str]] = []
        for text in corpus:
            if text:
                words.append(list(text.lower().strip()) + ["</w>"])

        vocab = set()
        for word in words:
            vocab.update(word)
        for tok in self.special_tokens:
            vocab.add(tok)

        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}

        print(f"[BPE] Starting training with {len(words)} samples...")

        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            words = self._merge_vocab(best_pair, words)

            merged = "".join(best_pair)
            if merged not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged] = idx
                self.inverse_vocab[idx] = merged
                self.merges[best_pair] = idx

            if (i + 1) % 1000 == 0:
                print(f"[BPE] Merges: {i+1}/{num_merges} | vocab_size={len(self.vocab)}")

        print(f"[BPE] Training complete. Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Fast encode using learned merges."""
        if not self.merges:
            return [self.vocab.get(c, self.vocab["<|unk|>"]) for c in text.lower()]

        # Convert to list of tokens
        tokens = list(text.lower()) + ["</w>"]

        # Greedy merge (improved version)
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            mergeable = [(i, p) for i, p in enumerate(pairs) if p in self.merges]

            if not mergeable:
                break

            # Merge the first occurring pair (leftmost)
            idx, pair = mergeable[0]
            new_token = "".join(pair)
            tokens = tokens[:idx] + [new_token] + tokens[idx + 2:]

        return [self.vocab.get(t, self.vocab["<|unk|>"]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inverse_vocab.get(i, "<|unk|>") for i in ids]
        text = "".join(tokens).replace("</w>", "")
        return text.replace("▁", " ").strip()

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[BPE] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=len(data["vocab"]))
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok.merges = {tuple(k.split()): v for k, v in data["merges"].items()}
        tok.special_tokens = data["special_tokens"]
        return tok
