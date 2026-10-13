# Training log

2026-10-13: no GroundJEV training run yet. Old U2 selected foundation and
completed evaluation receipts preserved. GroundJEV dependency setup started;
system Python lacks ensurepip, so bootstrap uses pip targeting the new
environment without modifying LocateMOT's environment.

Setup outcomes: Python 3.12.15 installed; isolated uv setup cache fixed a
300-second lock timeout. OneJev and custom SGLang dependency installs remain
active. Model download transport failures are recorded in CURRENT_STATUS;
completed byte prefixes remain. No training optimizer or GPU inference has
been launched yet.
