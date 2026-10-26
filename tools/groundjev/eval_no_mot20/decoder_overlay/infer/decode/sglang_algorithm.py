"""Keep native decoding/seed42; remove dependence on worker request ordinal.

The pinned upstream generator adds request_index*1_000_003 to its seed. Thus
moving an unchanged input to another worker changes its samples. All native
model, block, temperature, top-p, acceptance and parser code stays upstream.
Only the irrelevant service-local ordinal is held at zero for a request-local
stream. This repair has a separate cache identity and does not overwrite old
order-dependent experiments.
"""
import importlib.util
import os
from pathlib import Path
import sys

source=Path(os.environ['GROUNDJEV_OFFICIAL_SOURCE'])/'infer/decode/sglang_algorithm.py'
spec=importlib.util.spec_from_file_location('infer.decode._official_sglang_algorithm',source)
official=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=official
spec.loader.exec_module(official)


class GAMDecodeV2Block(official.GAMDecodeV2Block):
    @staticmethod
    def _generator(config,forward_batch,request_index,rel_start,step):
        return official.GAMDecodeV2Block._generator(config,forward_batch,0,rel_start,step)
