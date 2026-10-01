# LocateMOT-U U0R reproducibility audit

Date: 2026-10-01

The working branch is `codex/u0r-u1r-foundation-repair-20261001`.  It starts at
the restored commit `014b8a5ec81bb83c5993b56da171089f8a440a11` and contains the
small repair commit that makes the unified data package explicit in Git.  The
first fresh-clone attempt exposed the reason a local checkout was not
reproducible: the repository-wide `data/` ignore rule also ignored
`locatemot/unified/data/`, even though the package is source code.  The ignore
exception was corrected and the eight package files were force-added to the
repair commit.

The remote branch was cloned into
`/data2/user/locatemot_u_repro_check_20261001_r1`.  Dataset, weight, third-party,
and virtual-environment paths were linked only for the local smoke; no ignored
dataset or weight artifact was committed.  The clone resolved at remote commit
`26f9c989c8b13773d681bbe6962b77e73a0026bd` and passed:

```text
import locatemot.unified
import locatemot.unified.data.legal_scope
import locatemot.unified.models.unified_spec_track
python tools/unified/smoke_skeleton.py
python tools/unified/u1_foundation_smoke.py
```

The skeleton smoke is finite and the foundation smoke returns 300 finite
predictions on legal V1 video `0004`, frame `000063`.  The latter uses the
shared official MMDetection test pipeline and does not open official-test
labels.  The legal scope source is
`locatemot/unified/data/legal_scope.py`; the frozen v2 split and its SHA are
recorded separately in `LEGAL_SPLIT_AUDIT.md`.

Decision: fresh-clone reproducibility **PASS** after the ignore-rule repair.
The initial failed clone is retained as a diagnosis, not hidden; the repair is
the smallest source-control change that explains and fixes it.
