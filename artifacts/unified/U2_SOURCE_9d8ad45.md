# U2 temporal supervision and MOT-only baseline source

Local original-history commit `9d8ad45dc54146b5fadbbfd3638bfe441577fd78` has tree `bf713e5bd3902537f5a5607d8b35eafdc63536dc` and parent `10bb698f21cd69a1bda3e632fed46193259dd9ff`. The incremental bundle `u2_source_history_9d8ad45.bundle` preserves this source commit on top of the preceding published source history.

Original KITTI F0 audit completed on all 3,472 frames / 15,875 eligible GT units: Top150/Top300 recall .928503937/.940535433. Actual persistent-state augmentation, registered temporal losses, physical frame gaps and causal expiry are implemented. Eight-GPU BF16, ordinary/OVMOT temporal contracts, original train-video GT alignment and resume/selection checks pass; development failures and repairs are recorded.

The exact 6,000-step MOT-only baseline is preregistered and queued behind complete perception selection, ordinary feature cache and all-five simple metrics. No formal tracking metric, learned MOT checkpoint or joint OVMOT benchmark is claimed. U2 is not ready for U3. Immutable U1 references, original source scopes and all gates remain unchanged.
