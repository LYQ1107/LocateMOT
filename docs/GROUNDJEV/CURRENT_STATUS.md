# Current status — 2026-10-14

GroundAnything DLM + real pretrained OneJev completed four causal MOT17 frames
302–305, producing native candidates, typed decision probabilities, global
assignment, identity memory, prediction logs and a verified four-frame ID video.
This is a functional check, not a full-video/dataset result.

A same-candidate Simple control completed with unchanged TrackEval:

| Four-frame experiment | HOTA | DetA | AssA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained OneJev Full | .42163 | .65004 | .28169 | .42748 | 60 | 16 | 24 |
| Simple IoU reference | .74881 | .64230 | .89516 | .83650 | 1 | 17 | 24 |

The generic pretrained model currently fragments identities:62 persistent
IDs after four frames. Keep this negative result. It motivates tracking SFT;
it does not establish a paper-scale conclusion or a global NO_GO.

## Running work

The generic-greedy DanceTrack attempt stopped at frame21 after20 completed
frames: duplicated boxes plus a trailing separator are rejected even by the
official scoring parser. Its failed prediction/log/memory/resources remain.
The generic client lacked the official dense task request contract. A targeted
frame21 probe with the official dense tier produces5 scoring-valid boxes,
while still retaining its strict duplicate-coordinate diagnostic.

One new blocking owner runs the complete DanceTrack0004 validation video
(1,203 real frames) on physical GPU6, followed by a Simple control using the
exact same candidates and official HOTA/CLEAR/Identity evaluation. It records
per-frame stage progress, native decisions, latest causal memory, video,
actual native pipeline FPS and internally observed project GPU process peaks.
No full-video result exists yet. The new run is an explicit request-policy
repair: outer/inner temperature.1/top-p.95, official B32/subblock4/4 denoise
steps/entropy.8, sampling seed42 unchanged from the live server. Model and
DLM algorithm unchanged; parser and metric gates unchanged. The old20 frames
are not mixed into this distinct protocol run.

A second blocking owner runs the official OneJev multimodal100-step full
language-model SFT check on physical GPU4, frozen vision. It loads all474
pretrained entries and trains752.39M of852.99M parameters. Through observed step40, losses remain finite and gradients nonzero; peak
allocator15.2GiB. A full step25 checkpoint has been saved; reload/finished
training evidence is still pending. Training is still
running, so no completed SFT checkpoint/result is claimed.

The teacher curriculum has18 legal16-observation clips from MOT17-02,
DanceTrack0001 and MOT20-01,8,964 official decision rows and59,665,520bytes
of historical crop sheets. Identity aliases are randomized per clip. All
rows fit the official16384-token plan (max7858); none dropped. Noul labels
were corrected to official yes/no before training. Teacher history is only
initial curriculum; false-candidate/termination/long-gap and actual model
rollout coverage must be completed before formal500–1000step adaptation.

GroundJEV currently uses two physical GPUs4and6 within allowlist4/5/6/7.
UDMT and other-user processes remain untouched. Temporary feature budget25GiB;
no full-dataset features are cached.

## Failure evidence and repair

Frame305 initially failed strict client syntax (coordinate ordering/repeated
label blocks). Three identical retries reproduced it. The pinned official
scoring parser extracts35 geometrically valid boxes; the explicit
`official_scoring` policy calls that function unchanged. Strict wire validity
stays false in the log; every raw output/error and coordinate is retained.
A real-failure regression matches all official scorer labels/coordinates and
rejects truncated generations. Model, decoder, sampling and metrics unchanged.

The original runner cleanup waited on ffmpeg with its input pipe still open.
The exact owned encoder received SIGTERM then SIGKILL after its10s timeout;
closing stdin before waiting fixes cleanup and preserves the first failure.
Completed native decision replay exactly restored frames302–304 without GT
or extra model calls. A request-builder variable-name error in the first
resume was repaired and its failed run retained; the second resume completed.
Legacy failed-attempt/interruption timing is incomplete, so its complete
end-to-end FPS is null. Simple replay FPS is not native end-to-end speed.

## Preserved foundations

Old U2 owners/queues stopped and39.89GiB regenerable features removed; original
data/checkpoints/results remain. Independent worktree
`/data2/user/LocateMOT_GroundJEV`, branch`codex/groundjev-4gpu-20261013`,
inherited freshly fetched U2`51614cb`.

Official source revisions: GroundAnything8a3b43a1…, OneJev2470c759….
Both official weight bundles downloaded directly, matching pinned LFS digests.
GroundAnything uses its own Python3.12/Torch2.9.1/custom SGLang environment;
OneJev uses Python3.10/Torch2.10/Transformers5.17.0. Correcting OneJev's wrong
initial5.5.4 pin fixed the original fork check without changing its.05 tolerance.
Native GroundAnything hidden-feature export is pending; historical visual
memory currently holds real anchor/long/recent crops, explicitly labeled.

Old U2 has no trained tracking checkpoint. Its preserved perception recall
is not MOT HOTA. Full pilot remains incomplete: full-video results, completed
SFT and before/after metrics, RMOT/OVMOT loops, native features and further
controls/adaptation still require real work. Pilot completion will not end
the unified tracking research goal.
