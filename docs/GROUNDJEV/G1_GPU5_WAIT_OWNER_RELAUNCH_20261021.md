# G1 GPU5 wait-owner relaunch

The previous G1 owner PID `631075` had exited while its state still recorded
`WAITING_G2_CLEAN_SUCCESS`. Its log contained only the initial wait-state
record; there was no training directory, checkpoint, optimizer update or GPU5
launch. The owner was restarted with the same registered G2 dependency, output
paths and runtime model under PID `642505`.

The new owner is live at `WAITING_G2_CLEAN_SUCCESS`. GPU5 remains reserved for
the protected G2 process until the exact terminal G2 receipt appears. This is a
recoverability repair only and carries no G1 metric claim.
