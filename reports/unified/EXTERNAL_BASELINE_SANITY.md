# External baseline sanity — U0

The bounded sanity command completed without launching training, reading any Refer-KITTI official-test labels, or constructing an external dataset. It compiled the selected source files and invoked each advertised entrypoint with `--help`.

- Bytecode compilation: **10/10** repository checks passed.
- Argument-parser/import help: **1/10** completed successfully (iKUN only on this host).
- `training_launched=false`, `official_test_labels_read=false`, `screening_gt_used=false`.

## Results

| Repository | Compile | Help | First actionable environment result |
|---|---:|---:|---|
| MOTR | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/MOTR/main.py", line 23, in <module> import datasets File "/data2/user/reference_repos/MOTR/datasets/__init__.py", line 14, in <module> from .coco import build as build_coco File "/d ... e_repos/MOTR/util/misc.py", line 34, in <module> from torchvision.ops.misc import _NewEmptyTensorOp ImportError: cannot import name '_NewEmptyTensorOp' from 'torchvision.ops.misc' (/usr/local/lib/python3.10/dist-packages/torchvision/ops/misc.py) |
| TransRMOT | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/TransRMOT/main.py", line 24, in <module> import datasets File "/data2/user/reference_repos/TransRMOT/datasets/__init__.py", line 14, in <module> from .coco import build as build_coc ... os/TransRMOT/util/misc.py", line 34, in <module> from torchvision.ops.misc import _NewEmptyTensorOp ImportError: cannot import name '_NewEmptyTensorOp' from 'torchvision.ops.misc' (/usr/local/lib/python3.10/dist-packages/torchvision/ops/misc.py) |
| TempRMOT | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/TempRMOT/main.py", line 25, in <module> import datasets File "/data2/user/reference_repos/TempRMOT/datasets/__init__.py", line 15, in <module> from .refer_kitti import build as buil ...  in <module> from .transrmot_pro import build as build_rmot File "/data2/user/reference_repos/TempRMOT/models/transrmot_pro.py", line 33, in <module> from torchtext.data.utils import get_tokenizer ModuleNotFoundError: No module named 'torchtext' |
| DKGTrack | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/DKGTrack/main.py", line 25, in <module> import datasets File "/data2/user/reference_repos/DKGTrack/datasets/__init__.py", line 15, in <module> from .refer_kitti import build as buil ... DKGTrack/models/__init__.py", line 10, in <module> from .transrmot_pro import build as build_rmot File "/data2/user/reference_repos/DKGTrack/models/transrmot_pro.py", line 23, in <module> import spacy ModuleNotFoundError: No module named 'spacy' |
| FlexHook | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/FlexHook/main.py", line 18, in <module> from config import get_config File "/data2/user/reference_repos/FlexHook/config.py", line 5, in <module> from yacs.config import CfgNode as CN ModuleNotFoundError: No module named 'yacs' |
| iKUN | pass | pass | none |
| OVTR | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/OVTR/ovtr/main.py", line 20, in <module> from util.events import EventStorage, TensorboardXWriter File "/data2/user/reference_repos/OVTR/ovtr/util/events.py", line 10, in <module> from fvcore.common.file_io import PathManager ModuleNotFoundError: No module named 'fvcore' |
| COVTrack | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/COVTrack/tools/train.py", line 13, in <module> from mmcv import Config, DictAction ImportError: cannot import name 'Config' from 'mmcv' (/data2/user/LocateMOT/.venv/lib/python3.10/site-packages/mmcv/__init__.py) |
| ReferDINO | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/ReferDINO/main.py", line 6, in <module> from trainer import Trainer File "/data2/user/reference_repos/ReferDINO/trainer.py", line 8, in <module> import wandb ModuleNotFoundError: No module named 'wandb' |
| Open-GroundingDino | pass | fail | Traceback (most recent call last): File "/data2/user/reference_repos/Open-GroundingDino/main.py", line 15, in <module> from util.logger import setup_logger File "/data2/user/reference_repos/Open-GroundingDino/util/logger.py", line 6, in <module> import colorlog ModuleNotFoundError: No module named 'colorlog' |

The failures are dependency or version-contract findings: legacy torchvision symbols (MOTR/TransRMOT), repository-local imports after argument parsing (TempRMOT/DKGTrack), missing `yacs` (FlexHook), missing `fvcore` (OVTR), MMCV 2.x lacking the legacy `Config` API (COVTrack), missing `wandb` (ReferDINO), and missing `colorlog` (Open-GroundingDino). They are preserved as evidence; no package substitution or external metric claim was made.

The legal-scope guard and LocateMOT-U skeleton smoke are recorded separately in `outputs/unified/skeleton_smoke.json` and `outputs/unified/parameter_sharing_smoke.json`.
