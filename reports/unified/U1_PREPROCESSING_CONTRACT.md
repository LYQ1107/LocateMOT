# LocateMOT-U U1 preprocessing contract

Date: 2026-10-01

All foundation measurements call the shared runtime in
`locatemot/unified/runtime/grounding_inference.py`.  Its test path is the
MMDetection path: `LoadImageFromFile`, `FixScaleResize(scale=(800, 1333),
keep_ratio=True)`, `PackDetInputs`, and `DetDataPreprocessor`.  The
preprocessor performs the BGR-to-RGB conversion expected by this config, and
MMDetection restores postprocessed boxes to the original image coordinates.

The equivalence check selected ten deterministic legal development images from
V1 and V2.  It compared the direct official `inference_detector` call with the
shared wrapper using one fixed, query-independent vocabulary:

```text
car . van . truck . bus . tram . pedestrian . person . cyclist . bicycle . motorcycle .
```

Every image produced 300 finite predictions.  The maximum absolute box
difference was `0.0` pixels and the maximum score difference was `0.0`, below
the registered tolerances of `1e-4` and `1e-5`.  The exact rows and runtime
metadata are in `outputs/unified/u1_preprocessing_equivalence.json`.

Decision: preprocessing equivalence **PASS**.  The former hand-built CHW
smoke is no longer used as evidence for a formal proposal or grounding metric.
No official-test labels or screening ground truth were read.
