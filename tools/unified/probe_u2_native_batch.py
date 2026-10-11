#!/usr/bin/env python3
"""Small native FP32 image-batch admissibility probe; no cache protocol change."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from locatemot.unified.runtime.grounding_inference import load_grounding_runtime, run_grounding
    from tools.unified.u2_mot_portability import CHECKPOINT, FOUNDATION_SHA
    from tools.unified.build_u2_candidate_cache import write_json
    torch.set_num_threads(2)
    runtime = load_grounding_runtime(checkpoint=CHECKPOINT, device="cuda:0", max_per_img=900)
    assert runtime.checkpoint_sha256 == FOUNDATION_SHA and not runtime.missing_keys and not runtime.unexpected_keys
    protocol = json.loads((ROOT / "outputs/unified/u2/protocol/mot17_half.json").read_text())
    sequence = protocol["sequences"][0]
    images = [Path(sequence["image_dir"]) / frame for frame in sequence["validation_frames"][:2]]
    data = [runtime.test_pipeline({"img_path":str(path),"img_id":0,"text":protocol["vocabulary"],"custom_entities":True}) for path in images]
    assert data[0]["inputs"].shape == data[1]["inputs"].shape

    def arrays(result):
        predictions = result.pred_instances
        keep = torch.argsort(predictions.scores, descending=True)[:150]
        return {name:getattr(predictions,name)[keep].detach().cpu() for name in ("bboxes","scores","labels")}

    reference = [arrays(run_grounding(runtime,path,protocol["vocabulary"])) for path in images]

    def batch(alter_second=False):
        inputs = [row["inputs"].clone() for row in data]
        if alter_second:
            inputs[1] = 255 - inputs[1]
        packed = {"inputs":inputs,"data_samples":[copy.deepcopy(row["data_samples"]) for row in data]}
        torch.cuda.synchronize(runtime.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = runtime.model.test_step(packed)
        torch.cuda.synchronize(runtime.device)
        return [arrays(result) for result in output],time.perf_counter()-start

    observed, seconds = batch()
    changed, _ = batch(alter_second=True)
    matches = [{name:torch.equal(original[name],batched[name]) for name in original} for original,batched in zip(reference,observed)]
    errors = [{name:float((original[name].float()-batched[name].float()).abs().max()) for name in original}
              for original,batched in zip(reference,observed)]
    causal = {name:torch.equal(observed[0][name],changed[0][name]) for name in observed[0]}
    admissible = all(all(row.values()) for row in matches) and all(causal.values())
    report = {"format":"locatemot-u2-native-batch-admissibility-v1",
              "status":"SMALL_PROBE_PASSES_REQUIRES_FULL_FEATURE_VALIDATION" if admissible else "REJECT_NATIVE_BATCH_FOR_CURRENT_CACHE",
              "reference_only":True,"foundation_sha256":FOUNDATION_SHA,"precision":"native FP32, no autocast",
              "images":[str(path) for path in images],"frames":2,"candidate_cap":150,
              "same_native_preprocessed_input_shape":list(data[0]["inputs"].shape),
              "batch_size":2,"singleton_candidate_equality":matches,"max_same_rank_absolute_error":errors,
              "first_frame_equality_after_only_second_pixels_inverted":causal,"observed_batch_seconds":seconds,
              "fusion_global_softmax_shift_enabled":{name:module.stable_softmax_2d for name,module in runtime.model.named_modules()
                                                       if hasattr(module,"stable_softmax_2d")},
              "applied_to_formal_cache":False,"full_ovmot_vocabulary_batched":False,
              "original_full_audit_restarted_or_modified":False,"tracking_benchmark_measured":False,
              "official_test_labels_read":False,"ready_for_u3":False}
    write_json(ROOT / "outputs/unified/u2/eval/native_batch_admissibility.json",report)
    print(json.dumps(report,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
