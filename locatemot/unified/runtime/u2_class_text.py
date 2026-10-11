"""Class features from the actual shared GroundingDINO U1 BERT/projection."""
from __future__ import annotations

import torch


def encode_class_text(runtime, categories: list[dict], chunk_size: int = 32) -> tuple[torch.Tensor, dict]:
    """Reuse exactly native detector entity prompts and token pooling.

    No image, GT, second language model or learned fixed class table is used.
    The returned 256-D features are frozen foundation outputs; the existing
    OVMOT text projection and temperature provide the learned adaptation.
    """
    model = runtime.model
    if model.training or model.language_model.training:
        raise ValueError("Class text caching requires the frozen foundation in eval mode")
    features, token_counts, prompts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(categories), chunk_size):
            names = [category["name"] for category in categories[start:start + chunk_size]]
            tokenized, prompt, tokens_positive, entities = model.get_tokens_and_prompts(names, True)
            count = tokenized.input_ids.shape[1]
            if count > model.language_model.max_tokens:
                raise ValueError("The native class phrase chunk exceeds the registered BERT token budget")
            _, positive_map = model.get_positive_map(tokenized, tokens_positive)
            language = model.language_model([prompt])
            embedded = language["embedded"]
            if model.text_feat_map is not None:
                embedded = model.text_feat_map(embedded)
            phrase_weights = positive_map[:, :embedded.shape[1]].to(embedded)
            if (phrase_weights.sum(dim=1) <= 0).any():
                raise ValueError("A native LVIS class phrase has no BERT token mapping")
            phrase_weights = phrase_weights / phrase_weights.sum(dim=1, keepdim=True)
            pooled = phrase_weights @ embedded[0]
            if pooled.shape != (len(names), 256) or not torch.isfinite(pooled).all():
                raise ValueError("Shared foundation class text features have the wrong dimension or nonfinite values")
            features.append(pooled.detach().float().cpu())
            token_counts.append(count)
            prompts.append(prompt)
    return torch.cat(features), {"chunk_size": chunk_size, "chunk_count": len(features),
                                 "chunk_token_counts": token_counts, "prompts": prompts,
                                 "language_source": "actual checkpoint language_model plus text_feat_map",
                                 "pooling": "native positive-map normalized mean of phrase token features",
                                 "embedding_dimension": 256, "image_features_used": False}
