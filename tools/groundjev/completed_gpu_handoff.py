"""Validate a completed owned GPU queue's recorded release before reuse."""
import json
from pathlib import Path


def validate_release(receipt, physical_gpu):
    from tools.groundjev.run_unified600_controls import alive
    from tools.groundjev.service_process_group import members
    result = json.loads(Path(receipt).read_text())
    assert result['status'] == 'VERIFIED_COMPLETED_GPU_OWNER_RELEASE'
    assert result['physical_gpu'] == physical_gpu
    assert not alive(result['upstream_owner_identity'])
    assert result['all_owned_groups_gone']
    assert all(not members(group) for group in result['owned_process_groups'])
    return result
