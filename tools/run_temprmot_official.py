"""Run the released TempRMOT inference without modifying its checkout.

The released model hard-codes a RoBERTa directory under /data_2.  That path is
not writable in this environment, so this wrapper redirects only that exact
path to the already cached, same-named local roberta-base snapshot.  Legacy
torchtext/motmetrics/matplotlib import shims live under the L15 artifact root;
they are import-only and are not used by the model forward path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
REPO = Path(os.environ.get(
    "TEMPRMOT_REPO",
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/temp_rmot",
))
COMPAT = ROOT / "outputs/l15/official_baseline/temp_rmot_v2/compat"
LOCAL_ROBERTA = Path(
    "/home/lwr/.cache/huggingface/hub/models--roberta-base/snapshots/"
    "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)
HARD_CODED = "/data_2/zyn/Data4RMOT/FairMOT/src/roberta_base"

sys.path.insert(0, str(COMPAT))
sys.path.insert(0, str(REPO))


def _redirect_transformers():
    from transformers import RobertaModel, RobertaTokenizerFast
    from transformers import configuration_utils, modeling_utils
    from transformers import tokenization_utils_base

    def redirect_path(path):
        return (str(LOCAL_ROBERTA)
                if str(path).rstrip("/") == HARD_CODED.rstrip("/")
                else path)

    def patch(cls):
        original = cls.from_pretrained.__func__

        def redirected(inner_cls, path, *args, **kwargs):
            return original(inner_cls, redirect_path(path), *args, **kwargs)

        cls.from_pretrained = classmethod(redirected)

    patch(RobertaModel)
    patch(RobertaTokenizerFast)

    # Transformers 4.30 binds cached_file into several modules.  Redirecting
    # those call sites as well makes the mapping robust across multiprocessing
    # spawn and inherited classmethod implementations.
    for module in (configuration_utils, modeling_utils,
                   tokenization_utils_base):
        original_cached_file = module.cached_file

        def cached_file(path, filename, *args, _original=original_cached_file,
                        **kwargs):
            return _original(redirect_path(path), filename, *args, **kwargs)

        module.cached_file = cached_file


def main():
    _redirect_transformers()
    _execute_reference_source()


def _execute_reference_source():
    source = (REPO / "inference.py").read_text()
    # The released script hard-codes eight CUDA processes.  L15 experiments
    # are capped at four GPUs, so change only that process-count constant in
    # the wrapper execution; the reference checkout remains untouched.
    threads = int(os.environ.get("TEMPRMOT_THREADS", "8"))
    limit = os.environ.get("TEMPRMOT_QUERY_LIMIT")
    if limit:
        source = source.replace(
            "    thread_num = 8\n",
            f"    seq_nums = seq_nums[:{int(limit)}]\n"
            "    thread_num = 8\n", 1)
    source = source.replace("    thread_num = 8\n",
                            f"    thread_num = {threads}\n", 1)
    code = compile(source, str(REPO / "inference.py"), "exec")
    # Execute in the actual multiprocessing main-module globals.  Spawned
    # children import this wrapper as __mp_main__, define sub_processor from
    # the same source, and do not enter inference.py's main guard.
    exec(code, globals(), globals())


if __name__ == "__main__":
    main()
elif __name__ == "__mp_main__":
    _redirect_transformers()
    _execute_reference_source()
