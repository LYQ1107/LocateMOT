# Latest training status — 2026-10-14

Official real-image OneJev full language-model100-step correctness check is
running on physicalGPU4.474 pretrained tensors loaded;752.39M parameters
trainable, vision frozen. First5steps: loss2.4260/2.1067/2.0229/1.3239/1.7547,
nonzero gradient norms215.007/190.773/172.794/212.342/128.342; allocator peak
15.1GiB. These are online training logs, not completed training or tracking
performance. Save every25steps; validate/reload after the blocking owner ends.

Initial teacher curriculum:18 legal16-frame-observation clips,8,964rows,
59,665,520bytes history sheets. Actual max7858tokens vs16384cap,0rows dropped;
maximum18Choice options vs255API limit. Corrected1728 Noul target labels to
yes/no before training. Seed0/lr5e-6 unchanged. Real detector/model rollout,
negative detections and formal500–1000step stage are still pending.

Four-frame same-candidate before-SFT result is negative: Full HOTA.42163,
IDF1.42748,60IDSW; Simple HOTA.74881,IDF1.83650,1IDSW. This small functional
scope is not the full-video before/after evaluation. Complete native
DanceTrack0004 baseline and identical-candidate control are running onGPU6.

## Earlier setup history

# Training log

2026-10-13: no GroundJEV training run yet. Old U2 selected foundation and
completed evaluation receipts preserved. GroundJEV dependency setup started;
system Python lacks ensurepip, so bootstrap uses pip targeting the new
environment without modifying LocateMOT's environment.

Setup outcomes: Python 3.12.15 installed; isolated uv setup cache fixed a
300-second lock timeout. OneJev and custom SGLang dependency installs remain
active. Model download transport failures are recorded in CURRENT_STATUS;
completed byte prefixes remain. No training optimizer or GPU inference has
been launched yet.

OneJev install completed in a separate Python3.10 environment using copied,
ABI-matched local Torch2.10 CUDA dependencies and freshly installed
Transformers5.5.4/qev/peft/accelerate. All69 packages compatible; CUDA check on
visible physical GPU6 succeeds. Slow unused Python3.12 OneJev install stopped
after that verification. GroundAnything retains its own Python3.12/Torch2.9.1
recipe and official cuDNN override; remaining package stages use verified
local PyPI wheel transport. Weight-dependent real-frame check is queued;
training steps remain zero.

2026-10-14: actual OneJev weight loading (474 entries) initially failed the
official fork self-check under Transformers5.5.4 (.0857 probability error vs
.05 tolerance). Checkpoint metadata declares5.17.0; upgraded only OneJev's
isolated runtime and retained the check/tolerance. Same checkpoint then passes
auto validation and answers real MOT-frame Choice/Noul/Score questions, native
probabilities retained. Request .286737s, allocator peak3,119,928,832 bytes;
not a video benchmark. GroundAnything's separate official serve setup finished
and passed its cuDNN compatibility check. DLM weights were pending at that check; optimizer steps were0 then.
