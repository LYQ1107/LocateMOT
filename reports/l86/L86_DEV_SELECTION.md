# L86 internal dev checkpoint selection

Cheap dev scoring is stored at
`outputs/l86/eval/dev_cheap_attempt2/`. The first attempt is retained as an
implementation failure caused by an absent `FrameExample.candidate_count`;
the minimal fix used the native row-offset length and did not change metrics
or data protocol. The corrected run scored all 20 even checkpoints over 138
video-disjoint internal dev groups, producing 9,960 complete candidate-row
records.

Full-video dev HOTA was unavailable and was not substituted with validation.
The fixed Rule-B selection tuple was applied to dev only:

```text
(lower target-bag hard violation,
 higher target-bag hit@1,
 higher multi-target exact,
 lower inactive false acceptance,
 higher candidate recall,
 earlier epoch)
```

It selected:

- checkpoint: `outputs/l86/train/joint40/checkpoint_l86_epoch014.pt`
- epoch/step: 14 / 826
- SHA256: `b9a6d659e6b5315696370f5a8350f1abce1716eec8ccd8e12244f339b3e26be5`
- rule: B
- candidate threshold: `0.75`
- presence threshold: `0.0`
- NULL margin: `0.0`
- selection tuple:
  `[0.8108974359, -0.4198717949, -0.0, 0.5918367347,
  -0.3386666667, 14]`

The selection was frozen before fixed calibration/validation labels were
loaded. No screening or official-test labels were read.
