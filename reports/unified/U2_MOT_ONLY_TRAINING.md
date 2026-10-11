# LocateMOT-U U2 MOT-only training

The exact shared-core MOT-only run (6000 optimizer steps, 2/4/6/8-frame
curriculum, frozen U1 foundation) was not launched.  The U2-P portability
gate must pass on BDD100K, DanceTrack, MOT17, MOT20, and KITTI before candidate
cache construction and training.  There is no MOT-only checkpoint or metric
to select.
