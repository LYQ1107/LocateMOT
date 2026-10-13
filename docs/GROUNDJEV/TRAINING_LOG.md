# Latest training status — 2026-10-14

100-step official full language-model check completed400examples/1509.262s,
peak15.3GiB. All474weights saved; three BF16-aligned MLP probes changed,
two real multimodal reloads agree exactly. Original fork check unchanged.
Same-candidate V1 HOTA worsens.42163 to.32834, IDSW60to75. Preserve the
negative checkpoint and do not equate lower training loss with tracking gains.

Teacher context was limited to18options and had no wrong detections/model
states, unlike production35options. V2 crop/motion evidence collapsed due
to ambiguous NO_MATCH; V3 semantic repair retains substantial missing outputs.
All controls/results remain in the registry.

Two actual V3 actor rollouts completed96legal observations and13,501unique
supervised rows. They include526 contaminated history observations and
explicit synthetic false fragments; GT IDs never correct the actor. GT
proposals remain a declared curriculum source, not GroundAnything predictions.
Mix both branches with4x identity repeats:20,881rows. Official plan max22817
of32768tokens,0dropped.500-step finite full-language SFT is running onGPU4
from checkpoint100, fresh optimizer as an explicit curriculum stage, unchanged
seed0/lr5e-6. First2steps finite loss/gradient, peak18.1GiB, about76s/step.
Save every125steps, then validate/reload and run the exact before/after protocol.

Actual DLM-proposal rollout and the other task/adaptation studies remain
required; this finite stage does not by itself finish the pilot or full goal.

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
