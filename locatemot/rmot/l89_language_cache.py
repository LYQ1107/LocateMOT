"""Pure frozen GroundingDINO/BERT language-token cache for L89.

The cache contains no candidate, target, label, score, or image state.  Query
IDs are lookup metadata only and never enter the L89 model tensors.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class L89LanguageItem:
    dataset: str
    video: str
    query_id: int
    sentence_sha256: str
    tokens: torch.Tensor
    mask: torch.Tensor


def sentence_sha256(sentence: str) -> str:
    return hashlib.sha256(str(sentence).encode("utf-8")).hexdigest()


def cache_key(dataset: str, video: str, query_id: int, sentence: str) -> str:
    return hashlib.sha256(
        f"{dataset}|{video}|{int(query_id)}|{sentence_sha256(sentence)}".encode("utf-8")
    ).hexdigest()


class L89LanguageTokenCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.jsonl"
        self._index: dict[str, dict[str, object]] = {}
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item["cache_key"])
            if key in self._index:
                raise AssertionError(f"duplicate L89 language cache key: {key}")
            self._index[key] = item
        if not self._index:
            raise AssertionError("empty L89 language cache")

    @property
    def entry_count(self) -> int:
        return len(self._index)

    def get(
        self,
        dataset: str,
        video: str,
        query_id: int,
        sentence: str,
    ) -> L89LanguageItem:
        sentence = str(sentence)
        key = cache_key(dataset, video, query_id, sentence)
        info = self._index.get(key)
        if info is None:
            raise KeyError(f"L89 language cache miss: {dataset}|{video}|{query_id}")
        if str(info.get("dataset")) != str(dataset) or str(info.get("video")) != str(video):
            raise AssertionError("L89 language cache dataset/video mismatch")
        if int(info.get("query_id", -1)) != int(query_id):
            raise AssertionError("L89 language cache query lookup mismatch")
        expected_sha = sentence_sha256(sentence)
        if str(info.get("sentence_sha256")) != expected_sha:
            raise AssertionError("L89 language cache sentence SHA mismatch")
        path = self.root / str(info["file"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        forbidden = {
            "target_ids", "positive_indices", "positive_count", "category",
            "labels", "candidate_gt", "candidate_scores", "candidate_index",
        }
        if forbidden.intersection(payload):
            raise AssertionError(f"forbidden labels in L89 language cache item: {path}")
        if payload.get("labels_in_cache") or payload.get("candidate_gt_in_cache"):
            raise AssertionError(f"invalid L89 language cache flags: {path}")
        tokens = payload.get("tokens")
        mask = payload.get("text_token_mask")
        if not torch.is_tensor(tokens) or not torch.is_tensor(mask):
            raise AssertionError(f"invalid L89 language cache tensors: {path}")
        if tokens.ndim != 2 or mask.ndim != 1 or tokens.shape[0] != mask.shape[0] or tokens.shape[1] != 256:
            raise AssertionError(f"L89 language token shape drift: {path} {tuple(tokens.shape)} {tuple(mask.shape)}")
        mask = mask.bool().clone()
        if not bool(mask.any()) or not bool(torch.isfinite(tokens.float()).all()):
            raise AssertionError(f"invalid L89 language cache values: {path}")
        return L89LanguageItem(
            dataset=str(dataset), video=str(video), query_id=int(query_id),
            sentence_sha256=expected_sha, tokens=tokens.float().clone(), mask=mask,
        )

    def get_batch(
        self,
        dataset: str,
        video: str,
        query_ids: list[int],
        sentences: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(query_ids) != len(sentences) or not query_ids:
            raise ValueError("L89 language batch query/sentence mismatch")
        items = [self.get(dataset, video, qid, sentence) for qid, sentence in zip(query_ids, sentences)]
        length = max(int(item.tokens.shape[0]) for item in items)
        tokens = torch.zeros((len(items), length, 256), dtype=torch.float32)
        mask = torch.zeros((len(items), length), dtype=torch.bool)
        for index, item in enumerate(items):
            size = int(item.tokens.shape[0])
            tokens[index, :size] = item.tokens
            mask[index, :size] = item.mask
        tokens = tokens.to(device=device)
        mask = mask.to(device=device)
        if not bool(mask.any(dim=1).all()) or not bool(torch.isfinite(tokens).all()):
            raise FloatingPointError("invalid L89 language batch")
        return tokens, mask


__all__ = ["L89LanguageItem", "L89LanguageTokenCache", "cache_key", "sentence_sha256"]
