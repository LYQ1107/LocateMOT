# G1 GPU5 formal adaptation evaluation — 2026-10-26

G1 projector adaptation completed 200 optimizer steps on physical GPU5, visited all 56 registered pairs, passed checkpoint reload (`max_difference=0`), and exported a separate 569-tensor bundle without overwriting the base model.

The independent official receipts cover MOT20-05 (1,657 frames), DanceTrack `dancetrack0004` (1,203), AnimalTrack `chicken_1` (306), RMOT `0007` single-expression (800), and TAO OVMOT video63 (40 annotation frames). G1 HOTA/IDF1 are respectively 0.35023/0.29933 on MOT20-05, 0.10020/0.05118 on DanceTrack, 0.46612/0.49924 on AnimalTrack, and 0.03778/0.00894 on RMOT single. OVMOT rejects all 40 frames and reports TETA=0.

The same-scope G0 comparison is mixed: G1 is higher on MOT20-05 and AnimalTrack, lower on DanceTrack and RMOT single. The result is a verified adaptation and held-out measurement, not a universal performance-improvement claim. The 96-frame original/projector/G1/repeat controls are deterministic and retained separately.
