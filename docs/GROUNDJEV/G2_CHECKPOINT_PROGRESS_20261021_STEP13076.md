# G2 Clean checkpoint progress — event step 13076

- Observed: `2026-10-21T06:40:39.346423+08:00`
- Event stream: step `13076/18227` (event time `2026-10-21 06:40:32 +0800`)
- New adapter checkpoint: `adapter-step013000.pt`, `469243990` bytes, SHA-256 `08cf28930eced61b2086785494728221e113ac75cd7dc6aefe02ab1bc2ebfe59`
- Latest full recovery state: step `13000`, `469247597` bytes, SHA-256 `4a79cbf3e64e69ffc56aaf964754d0209382b87c2124d46f13506325ca52fd22`
- Final receipt: absent; training remains active

The full-state checkpoint remains the protected recovery point while the event stream continues. This is checkpoint/training-health evidence only and does not claim MOT/RMOT/OVMOT performance.
