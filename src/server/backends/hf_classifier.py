"""Local HF sequence classifier (Prompt Guard 2, deberta prompt-injection, ...).

Cheap first stage: ~1-10 ms/sample on a MIG slice. Long inputs are split into
overlapping token windows and the max window score is used (an injection hidden in
the middle of a long paste must not be diluted by truncation).
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import BACKENDS, GuardBackend, GuardInput, Verdict


@BACKENDS.register("hf_classifier")
class HFClassifierBackend(GuardBackend):
    """Local HF sequence classifier (Prompt Guard 2, injection / content-safety heads); max score over token windows."""

    def __init__(self, name: str, cfg: dict[str, Any], root_cfg: dict[str, Any]):
        super().__init__(name, cfg, root_cfg)
        self.model_id = cfg["model"]
        self.revision = cfg.get("revision")
        self.device = cfg.get("device", "cuda")
        self.dtype = cfg.get("dtype", "float16")
        self.max_length = int(cfg.get("max_length", 512))
        self.stride = int(cfg.get("stride", 64))
        self.batch_size = int(cfg.get("batch_size", 16))
        # label names counted as "unsafe"; compare case-insensitively against model.config.id2label.
        # Prefer `unsafe_label_ids` when a config has no id2label (Llama-Prompt-Guard-2 -> LABEL_0/LABEL_1).
        self.unsafe_labels = {s.lower() for s in cfg.get("unsafe_labels", ["MALICIOUS", "INJECTION", "LABEL_1", "unsafe"])}
        self.unsafe_label_ids: list[int] | None = cfg.get("unsafe_label_ids")
        self.activation = cfg.get("activation", "softmax")  # softmax (single-label) | sigmoid (multi-label heads)
        # multi-label heads (Horizon content-safety-guard): label -> unified category, reported when flagged
        self.category_labels: dict[str, str] = {k.lower(): v for k, v in cfg.get("category_labels", {}).items()}
        self.category_threshold = float(cfg.get("category_threshold", 0.5))
        self.pair_response = bool(cfg.get("pair_response", False))  # response check as (prompt, response) pair
        self.category = cfg.get("category", "prompt_injection")
        self._model = self._tok = None
        self._unsafe_ids: list[int] = []
        self._pair_prompt_tokens = 0  # set by _init_pair_budget once the tokenizer is loaded
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._load)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=getattr(torch, self.dtype)
        ).to(self.device).eval()
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        self._id2label = id2label
        self._unsafe_ids = list(self.unsafe_label_ids) if self.unsafe_label_ids is not None else \
            [i for i, lbl in id2label.items() if lbl in self.unsafe_labels]
        if not self._unsafe_ids:
            raise ValueError(f"{self.model_id}: none of {self.unsafe_labels} in id2label={id2label}; set unsafe_label_ids")
        self._cat_ids = {i: self.category_labels[lbl] for i, lbl in id2label.items() if lbl in self.category_labels}
        self._init_pair_budget()

    def _init_pair_budget(self) -> None:
        """Prompt tokens a (prompt, response) window may hold: at most half the window, and always leaving
        more than ``stride`` tokens for the response. Otherwise the fast tokenizer panics (``stride`` must be
        strictly less than the room left — a pyo3 ``PanicException``, not an ``Exception``) or, for longer
        prompts, raises "Sequence to truncate too short"; and a response window of ``stride + 1`` tokens
        would advance one token per window."""
        assert self._tok is not None
        room = self.max_length - self._tok.num_special_tokens_to_add(pair=True)
        self._pair_prompt_tokens = min(room // 2, room - self.stride - 1)
        if self.pair_response and self._pair_prompt_tokens < 1:
            raise ValueError(f"stage {self.name!r}: stride={self.stride} leaves no room for the prompt in a "
                             f"max_length={self.max_length} pair window; lower stride")

    def _fit_pair_prompt(self, prompt: str) -> str:
        """The prompt's last ``_pair_prompt_tokens`` tokens (the end of the prompt is what the response answers)."""
        assert self._tok is not None
        keep = self._pair_prompt_tokens
        for _ in range(3):  # decode -> encode round trips can shift the count by a few tokens
            ids = self._tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) <= self._pair_prompt_tokens:
                return prompt
            prompt = self._tok.decode(ids[-keep:], skip_special_tokens=True)
            keep = max(1, keep - 8)
        return ""

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "model": self.model_id, "category": self.category}

    def _score_sync(self, text: str, pair: str | None = None) -> tuple[float, list[str]]:
        import torch

        assert self._tok is not None and self._model is not None
        if pair is not None:
            text = self._fit_pair_prompt(text)
        enc = self._tok(
            text, pair, truncation="only_second" if pair else True, max_length=self.max_length, stride=self.stride,
            return_overflowing_tokens=True, padding=True, return_tensors="pt",
        )
        enc.pop("overflow_to_sample_mapping", None)
        best, cat_best = 0.0, {}
        with torch.inference_mode():
            for i in range(0, enc["input_ids"].shape[0], self.batch_size):
                batch = {k: v[i : i + self.batch_size].to(self.device) for k, v in enc.items()}
                logits = self._model(**batch).logits.float()
                probs = logits.sigmoid() if self.activation == "sigmoid" else logits.softmax(-1)
                unsafe_p = probs[:, self._unsafe_ids].max(-1).values if self.activation == "sigmoid" \
                    else probs[:, self._unsafe_ids].sum(-1)
                best = max(best, float(unsafe_p.max()))
                for idx, cat in self._cat_ids.items():
                    cat_best[cat] = max(cat_best.get(cat, 0.0), float(probs[:, idx].max()))
        cats = sorted(c for c, p in cat_best.items() if p >= self.category_threshold)
        return best, cats

    async def _classify(self, inp: GuardInput) -> Verdict:
        text = inp.target_text
        if not text.strip():
            return Verdict(stage=self.name, unsafe=False, score=0.0)
        pair = None
        if inp.input_type == "response" and self.pair_response:
            prompt = next((m.get("content") for m in reversed(inp.messages[:-1]) if m.get("role") == "user"), "")
            text, pair = (prompt or ""), text
        async with self._lock:  # one forward at a time on the slice; the LLM stage is the bottleneck anyway
            score, cats = await asyncio.get_running_loop().run_in_executor(None, self._score_sync, text, pair)
        unsafe = score >= self.threshold
        return Verdict(stage=self.name, unsafe=unsafe, score=score,
                       categories=(cats or [self.category]) if unsafe else [])
