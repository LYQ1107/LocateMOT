"""Worker-local memoization of fixed native text in frozen U2 inference.

Image features and multimodal transformers always run for every image/chunk.
Native consumers receive independent copies of cached text tensors and maps.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def frozen_text_cache(runtime):
    """Reuse native BERT/token maps within one fixed-weight inference worker."""
    model = runtime.model
    language = model.language_model
    if model.training or language.training:
        raise ValueError("Frozen text reuse is an evaluation-only operation")
    methods = [(language, "forward"), (model, "get_tokens_and_prompts"),
               (model, "get_tokens_positive_and_prompts")]
    originals = {(id(module), name): (getattr(module, name), name in module.__dict__)
                 for module, name in methods}
    cached_language, cached_tokens, cached_positive = {}, {}, {}
    stats = {"bert_calls": 0, "bert_hits": 0, "token_prompt_calls": 0,
             "token_prompt_hits": 0, "positive_prompt_calls": 0, "positive_prompt_hits": 0}

    def language_forward(self, captions, **kwargs):
        # These native inference callers have no language forward kwargs.
        if torch.is_grad_enabled() or kwargs or self.training:
            raise ValueError("Text reuse requires native frozen inference with no forward overrides")
        key = tuple(captions)
        if key not in cached_language:
            stats["bert_calls"] += 1
            result = originals[(id(language), "forward")][0](captions)
            cached_language[key] = {name: value.detach().clone() for name, value in result.items()}
        else:
            stats["bert_hits"] += 1
        # Native prediction replaces embedded with projected/fused features.
        return {name: value.clone() for name, value in cached_language[key].items()}

    def token_prompts(self, original_caption, custom_entities=False, enhanced_text_prompts=None):
        original = originals[(id(model), "get_tokens_and_prompts")][0]
        if enhanced_text_prompts is not None or not custom_entities:
            return original(original_caption, custom_entities, enhanced_text_prompts)
        key = (original_caption if isinstance(original_caption, str) else tuple(original_caption), custom_entities)
        if key not in cached_tokens:
            stats["token_prompt_calls"] += 1
            cached_tokens[key] = copy.deepcopy(original(original_caption, custom_entities, enhanced_text_prompts))
        else:
            stats["token_prompt_hits"] += 1
        return copy.deepcopy(cached_tokens[key])

    def positive_prompts(self, original_caption, custom_entities=False, enhanced_text_prompt=None, tokens_positive=None):
        original = originals[(id(model), "get_tokens_positive_and_prompts")][0]
        if enhanced_text_prompt is not None or tokens_positive is not None or not custom_entities:
            return original(original_caption, custom_entities, enhanced_text_prompt, tokens_positive)
        key = (original_caption if isinstance(original_caption, str) else tuple(original_caption),
               custom_entities, self.test_cfg.get("chunked_size", -1))
        if key not in cached_positive:
            stats["positive_prompt_calls"] += 1
            cached_positive[key] = copy.deepcopy(original(original_caption, custom_entities, enhanced_text_prompt, tokens_positive))
        else:
            stats["positive_prompt_hits"] += 1
        return copy.deepcopy(cached_positive[key])

    language.forward = MethodType(language_forward, language)
    model.get_tokens_and_prompts = MethodType(token_prompts, model)
    model.get_tokens_positive_and_prompts = MethodType(positive_prompts, model)
    try:
        yield stats
    finally:
        for module, name in methods:
            original, overridden = originals[(id(module), name)]
            if overridden:
                setattr(module, name, original)
            else:
                delattr(module, name)


__all__ = ["frozen_text_cache"]
