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
and passed its cuDNN compatibility check. DLM weights pending; optimizer steps0.
