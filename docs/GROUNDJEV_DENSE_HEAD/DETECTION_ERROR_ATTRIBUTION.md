# Detection error attribution

Same1,607 external frames and GT, native final G2 cache vs each frozen readout. Thresholds were fixed on internal DEV. Coverage transitions below do not imply net improvement: consult FP/precision/duplicate and one-to-one metrics in PILOT_RESULTS.json.

|Model / dataset|GT|Both covered|Newly recovered native misses|Lost native coverage|Both missed|Recovered after native parser rejection|
|---|---:|---:|---:|---:|---:|---:|
|global/g0/animaltrack|17416|1454|436|10316|5210|195|
|global/g0/sportsmot|8444|1045|8|7289|102|0|
|global/g2/animaltrack|17416|5066|1747|6704|3899|590|
|global/g2/sportsmot|8444|2287|27|6047|83|0|
|spatial_prior_v2/g0/animaltrack|17416|5849|2109|5921|3537|724|
|spatial_prior_v2/g0/sportsmot|8444|2545|15|5789|95|0|
|spatial_prior_v2/g2/animaltrack|17416|5844|1941|5926|3705|711|
|spatial_prior_v2/g2/sportsmot|8444|2365|17|5969|93|0|

Per-category/size transitions and changed-target examples are in the JSON. Direct numeric box output has no language grammar/parser step; class/category understanding and detection accuracy remain separate requirements. No AR regeneration, GT candidate construction or external parameter tuning was performed.
