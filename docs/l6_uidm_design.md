# Stage L6 UIDM Architecture Design

Date: 2026-08-11

## 1. One-sentence method

**We model heterogeneous MOT as learning a shared causal identity dynamics
process over a set of interacting trajectories**: one checkpoint maintains a
persistent memory per object, synchronises all objects every frame, decodes
the identity transition (continue / NEW / NO-MATCH), and updates its own
states from its own associations (model-in-the-loop).

## 2. Why each component exists (traceable to L1–L5 evidence + audit)

| Component | Scientific reason | Evidence / audit source |
|---|---|---|
| Per-track persistent memory `h_i^t` | identity is a temporal process, not a single-frame embedding | L1-B negative (universal ReID fails); Samba hidden state; MOTIP trajectory history; L5 positive (temporal identity helps) |
| Anchor memory `a_i` | first/reliable observation must not be overwritten by noisy frames | L1D pbd_anchor_cos feature; MOTIP in-context history |
| Frame-level set interaction over track+candidate tokens | MOT is a set of interacting sequences; competition must be inside the temporal model | Samba synchronized states; MOTIP self-attention among current objects; L3 router failure (no dataset shortcut) |
| Learned transition decoder (pair logits + NO_MATCH + NEW + alive) | association is an identity state transition, not a fixed weighted formula | L5: fixing only association score increases BDD IDSW (lifecycle must be learned); CO-MOT: birth is a first-class target |
| Motion/Kalman/IoU/PBD as **evidence inputs** (not fixed formula) | heterogeneous domains need learned cue reliability | L3: per-domain oracle mixes C1/L1DK/EGRA; MOT17/MOT20 motion-sensitive |
| Causal masking / no future leak | online inference must equal training | MOTIP causal cross-attn mask; AGENTS online-causal constraint |
| Tracking-level loss (soft IDSW/FP/FN penalty + box smoothness) | row-CE alone (98% acc in L5) does not fix persistent drift | L5 finding; UniTrack hinge criterion |
| Lifecycle (alive logit, memory decay, birth, termination) | BDD IDSW rose when only association was fixed | L5 BDD IDSW 11042→12399 |

## 3. UIDM-Base structure

```
Frame t candidates (boxes, PBD 2048, gen)
        │
        ├─ Candidate Encoder (PBD L2-norm + normalized box + gen + margins)
        │        c_j^t ∈ R^d
        │
Track states h_i^{t-1}, anchor_i, last box, Kalman prediction, gap, age, hits
        │
        └─ Track token = h_i^{t-1} + TrackEvidenceMLP(track_feats_i)
                │
        Set-of-sequences interaction (causal, per-frame):
        TransformerEncoder over [track tokens (T) + candidate tokens (N)]
                │  s_i' (track), c_j' (candidate)
        ┌───────┼─────────────────────────────┐
        │       │                             │
   Pair head  NoMatch head                New head / Alive head / Motion head
   [s_i',c_j', ─ s_i' → no_match_i         c_j' → new_j
    pair_feats]                            s_i' → alive_i', pred box
        │
   Extended transition logits:
      rows    = T × (N candidates + NO_MATCH)
      columns = N × (T tracks + NEW)
        │
   Structured one-to-one assignment
   (training: soft CE + straight-through sampling;
    inference: Hungarian/LSA)
        │
   Memory update (differentiable):
      matched:  h_i' = UpdateCell(h_i, c_j', s_i')
      no-match: h_i' = DecayCell(h_i, gap)
      birth:    h_i  = InitCell(c_j'), anchor = c_j'
      death:    alive logit < threshold or age > MAX_AGE
        │
   Next frame
```

## 4. Identity representation

- NOT a universal embedding.  Identity lives in:
  - `h_i^t`: recurrent state (who/how moving/reliable),
  - `anchor_i`: first-observation token (permanent identity anchor),
  - competition context from set interaction (who else is present).
- GT identity is used **only as a training label** (which candidate belongs
  to which track), never as a model input or dataset-specific signal.

## 5. Lifecycle formulation (learned)

- Row target: correct candidate if the track's identity is observed this
  frame, else NO_MATCH.
- Column target: the track whose identity matches the candidate, else NEW.
- Alive target: 1 while the identity was observed within MAX_AGE=30 frames,
  else 0.  The model predicts `alive_i` after each update; at inference a
  track is removed when `alive` drops below threshold or age > MAX_AGE.
- Memory decay: unmatched tracks update via a learned DecayCell; decay
  magnitude is controlled by gap and state uncertainty.

## 6. Training objective (tracking-level, not row-CE-only)

1. `L_row` / `L_col`: cross-entropy on the extended transition matrix.
2. `L_life`: BCE on NEW / NO_MATCH / alive.
3. `L_motion`: L1 between the motion-head predicted box and the matched
   candidate box (only for matched tracks).
4. `L_switch`: soft margin ranking — for each candidate, the correct track
   (or NEW) must beat every incorrect track by a margin; vectorised
   UniTrack-style hinge that directly penalises soft ID switches / FP.
5. `L_smooth`: small L1 on consecutive matched box centers (temporal
   consistency; UniTrack temporal component, weight small so Dance's fast
   motion is not crushed).

Total: `L = L_row + L_col + 0.3 L_life + 0.3 L_motion + 0.5 L_switch +
0.05 L_smooth`.

## 7. Model-in-the-loop training

- Each clip (H=16 frames) is rolled out by the model itself: states at
  frame t+1 come from the model's states at frame t.
- Scheduled sampling: teacher forcing (GT transitions) for the first
  ~15–20% of steps, then anneal to ~40% teacher / 60% student rollout.
- Student memory updates use hard assignments (Hungarian) with the
  **selection index detached but the selected candidate token
  differentiable** (straight-through); truncated BPTT with state detach at
  clip boundaries.
- Clip starts with an empty state table (matches full-video inference from
  frame 0).

## 8. Parameter budget

UIDM-Base: d=320, 4 set layers, FFN=1280, 8 heads, PBD MLP 2048→320.
Estimated ≈ 13–16M params (architecture-driven; no param sweep).
UIDM-Large (only if Base clearly underfits): d=384, 6 layers ≈ 30M.

## 9. Explicitly NOT in UIDM

- No dataset ID / regime router / dataset-specific head (L3 shortcut).
- No universal ReID cosine (L1-B failure).
- No fixed residual on L1DK (L5 showed lifecycle must be learned).
- No future-utility oracle (L2 headroom low).
- No detector training / no MOTSynth.
