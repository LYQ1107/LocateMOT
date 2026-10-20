# Research log append — runtime config parser fix

The dedicated OneJev parser initially rejected four audit-only metadata keys
that had been placed in the runtime training JSON. Those keys were removed
from `real_three_task_joint_v1_runtime.json` and remain recorded in the queue
state and research documents. The registered `.venv-onejev-py310` parser now
accepts the config and resolves the corrected source manifest, one epoch, and
global 16-row accumulation. No optimizer update was started during the fix.
