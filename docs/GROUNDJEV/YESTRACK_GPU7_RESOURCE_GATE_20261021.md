# YesTrack GPU7 resource gate

- Checked: `2026-10-21T06:23:23.820164+08:00`
- Queue status: `WAITING_GPU7_EXCLUSIVE`
- Gate: no compute PIDs and at least 32768 MiB free.
- Result: **blocked by an external process**.
- GPU7 query: `7, NVIDIA A800-SXM4-80GB, 81920, 16225, 64929, 0`
- External process: PID `4008192`, user `ljh`, command `python -u train.py --method PRISM --dataset_name cub ...`, approximately 16.2 GiB and 100% utilization.
- Action: left untouched; the YesTrack owner continues waiting and will launch only after GPU7 is exclusive.

This is a resource-gate receipt only. It is not a YesTrack training or metric result.
