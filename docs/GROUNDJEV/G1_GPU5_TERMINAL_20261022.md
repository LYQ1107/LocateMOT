# GroundAnything G1 GPU5 terminal evidence

The registered G1 GPU5 owner reached
`G1_GPU5_ADAPTATION_AND_CONTROLS_COMPLETE`. Native Ground Tracking200
training completed 200 steps on physical GPU5. The run visited all registered
training pairs, and reloading the latest checkpoint produced a maximum
difference of `0.0`.

The projector delta contains `16,211,750` changed BF16 elements and is recorded
at `training/projector_delta.pt` (SHA256
`e65020bfc62dc4440edb3138b74e53b18c0d36283ef1fd4f0a25dbb45a337ed1`). The
recoverable training state is `training/latest_training_state.pt` (SHA256
`60c41cfb46f36d20d49512e90dd4c7764ab614e4e1870604d98bc4f61be38808`). The
exported bundle is
`models/GroundAnything-DLM-TrackProjG1-GPU5-V2`, with 569 native projector
tensors and SHA256
`0efd9d7ddb241d588be85cb6b08853de5dfd18011aa477932c0580bab3265a25`.

The isolated controls completed 12 ordered units over 96 registered native
frames: MOT dense4, RMOT refer-single8, and RMOT refer-multi12. Repeated runs
used the same raw wire candidates and outputs for all three scopes. This
receipt verifies adaptation, checkpoint reload, and control determinism; it
does not claim a full-dataset benchmark improvement.
