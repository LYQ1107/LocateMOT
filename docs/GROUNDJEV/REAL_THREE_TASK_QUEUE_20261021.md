# Corrected three-task OneJev queue — 2026-10-21

`scripts/groundjev/queue_real_three_task_joint_v1.py` is a new resumable
owner for the corrected 131,990-row source manifest. It does not touch the
paused historical `full_joint_source_epoch1_resume_queue_v2` owner or its
step-3,000 checkpoint.

The owner gates on terminal G2 Clean, G1 GPU5 adaptation, bundle export,
independent held-out evaluation, and an exclusive four-GPU window (physical
4/5/6/7, at least 32 GiB free on each). It then launches one distributed
OneJev epoch from the registered native step-600 checkpoint in the dedicated
`.venv-onejev-py310` environment. Its OVMOT component is explicitly the legal
Base teacher-control (215 group + 215 leaf rows); the native TAO archive is
not mislabeled as training data. A CPU preflight found 2,099 rows above the
registered 32,768-token limit (maximum 96,734). The first bounded course
therefore expects 129,891 exposed rows and records the 2,099 overlength rows
explicitly; its terminal receipt cannot be reported as full-source exposure.
A separate compacting or long-row course is required before making a
full-source training claim.

The queue is a reproducible comparison run, not the final online GroundJEV
architecture. Its state records `full_goal_complete=false` until the coupled
single-backbone path and video-level held-out metrics are independently
verified.
