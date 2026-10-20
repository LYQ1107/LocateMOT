# G2 independent held-out manifest preflight — 2026-10-20

All four registered independent manifests passed task, frame-count, frame-order, and local-image existence checks. The queue owner separately performs the exact normalized image-path overlap audit against G2 Clean TRAIN before inference.

- `mot20_05_validation1657`: mot, 1657 frames, frames 1659–3315, SHA256 `2be2233c2e66323b22ccdb1b9b386646c6b792d9de5e77554cc1143b80ba42a6`
- `rmot_single0007`: rmot, 800 frames, frames 0–799, SHA256 `79e4286fc904209df00a54a5ccc58171e4f267d915a2256e80e5c3db5df5d32c`
- `rmot_multi0007`: rmot, 800 frames, frames 0–799, SHA256 `fc09932ae678be52e6091178a08b565e41a7895d5ff76192d50b89caa27e5e14`
- `ovmot_tao105`: ovmot, 40 frames, frames 1–1171, SHA256 `522cde4bfc776b7b8a18807a5caee30330c322eb9dd8df86fc472d150db9e261`

This receipt validates inputs only; it is not a model or tracking metric result.
