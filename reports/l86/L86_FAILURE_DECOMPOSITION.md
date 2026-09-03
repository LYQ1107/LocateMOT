# L86 failure decomposition and evidence decision

## Status

`full_rmot_hota_complete_improved`

L86 completed the registered 40-epoch RMOT-only training, fixed semantic
evaluation, legal internal full-video inference and TrackEval. It improved the
internal full-video HOTA descriptor over L85 in both domains, but it did not
pass the deployable fixed semantic gate and is not ordinary-level RMOT proof.

## Semantic failure

The selected epoch14/Rule-B output reduced volume and hard-negative violation,
but suppressed too many held-out positives. Validation recall was `.3225806`
versus L29 `.7333333`; multi-positive recall was `.2638889` versus `.8194444`.
The hard violation improved from `.9166667` to `.8461538`, precision was
`.1176471`, FP/frame was `3.125`, predictions/positive was `2.7419`, and
inactive false acceptance was `.8333333`. Thus the passing hard/precision/
volume columns are not sufficient: recall and multi-positive preservation fail
simultaneously. The final frozen rule retained complete candidate rows and did
not use top-k, NMS, candidate deletion or post-hoc NULL suppression.

The first actionable bottleneck is **held-out query-to-target emission and
multi-positive/presence calibration**. Candidate coverage is not a sufficient
explanation: the preceding candidate/oracle audit had unit coverage about
`.8385`, target micro coverage about `.8792`, and the GT-privileged oracle
TrackEval ceiling was much higher than the learned result. This does not prove
the learned semantic representation is adequate; it identifies the remaining
failure after the faithful target-bag and temporal repair.

## Full-video result

The HOTA gain is real for the declared internal protocol, not a semantic-gate
pass. V1 HOTA rose from `25.0548` to `29.1663` (`+4.1115`); V2 rose from
`17.2924` to `21.6467` (`+4.3543`). V1 DetA/DetPr/DetRe improved, while AssA
slightly decreased. V2 DetA, AssA and DetPr improved, while DetRe decreased
slightly. IDSW decreased in both domains. The HOTA gain is therefore best
described as a full-video emission/track-interaction improvement under the
frozen rule, not evidence that expression correspondence has reached ordinary
RMOT quality.

## Preserved failures and limitations

- The first cheap-dev attempt failed only because of a missing data-object row
  count attribute; it is retained and the corrected attempt is authoritative.
- The first fixed semantic attempt failed only at the existing
  `sidecar_candidate_gt` field name; it is retained and the corrected attempt
  is authoritative.
- The first oracle seqmap attempt is retained; the repaired seqmap attempt was
  used for the oracle TrackEval comparison.
- The TrackEval checkout has no verifiable Git HEAD. This is recorded, not
  silently treated as an official revision.
- Fine-grained token/span-to-region and static/motion alignment are
  `UNALIGNED`.

## Boundary and one next action

No screening labels, official-test labels, TrackEval production path, ordinary
MOT, OVMOT, TAO, UIDM or legacy bank/checkpoint were changed. Do not launch
L86-R1, alter the threshold/loss/tracker, or enlarge training automatically.
The unique next action is **supervisor review of the semantic-gate failure
against the improved internal HOTA and oracle ceiling, followed by one newly
authorized RMOT hypothesis**.
