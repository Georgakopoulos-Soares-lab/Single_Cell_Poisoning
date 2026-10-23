# BIORAG_* settings behind each 03a/03b tag

Several defence steps take their settings from environment variables and
write one table per `BIORAG_TAG`. Below is each tag's settings, copied from
the settings line of its run logs in `v02/2_results/GSE136831/logs/`
(line 4 for 03a_01 and 03b_01, line 1 for 03a_02 and 03a_03). Each log is
the authority; this table was transcribed by hand from them.

For example, to re-run one shard of 03a_01's `v3reluni` tag:

    BIORAG_ATTACKS=relabel BIORAG_SELECTIONS=uniform \
    BIORAG_DOSES=0.002,0.005,0.01,0.025,0.05,0.1,0.25,0.5,0.7,0.8,0.85,0.9,1,1.25 \
    BIORAG_TAG=v3reluni BIORAG_SAVE_RETRIEVAL=1 BIORAG_SHARD=0 BIORAG_N_SHARDS=3 \
    python v02/1_scripts/03_a_defence_vs_insert/03a_01_no_access_layer_matrix.py

Run every shard, from 0 to N_SHARDS-1. The shards are then merged with
`v02/1_scripts/00_pipeline/merge_shards.py`. **This README has not checked
that merge's exact command.**

Dose lists used below:
- **D14** = 0.002,0.005,0.01,0.025,0.05,0.1,0.25,0.5,0.7,0.8,0.85,0.9,1,1.25
- **D13** = D14 without 1.25
- **D9** = 0.005,0.025,0.1,0.5,0.7,0.8,0.85,0.9,1
- **D7** = 0.005,0.025,0.5,0.7,0.8,0.85,0.9

## 03a_01 (`03_a_defence_vs_insert/03a_01_no_access_layer_matrix.py`)

All with `BIORAG_N_SHARDS=3`. The `v3*` tags also set
`BIORAG_SAVE_RETRIEVAL=1`.

| Tag | ATTACKS | SELECTIONS | DOSES | FLOOD_ANCHORS | LABEL_CHANGE |
|---|---|---|---|---|---|
| dupcon, v3dupcon | duplicate | concentrated | D14 | - | - |
| v3dupconkeep | duplicate | concentrated | D14 | - | keep |
| flo64, v3flo64 | flood | hub | D14 | 64 | - |
| flo256, v3flo256 | flood | hub | D14 | 256 | - |
| flo1024, v3flo1024 | flood | hub | D14 | 1024 | - |
| v3flo1024keep | flood | hub | D14 | 1024 | keep |
| flo4096, v3flo4096 | flood | hub | D14 | 4096 | - |
| flo16k | flood | hub | D7 | 16384 | - |
| v3flo16384 | flood | hub | D14 | 16384 | - |
| intuni, v3intuni | interpolate | uniform | D14 | - | - |
| relhub | relabel | hub | D7 | - | - |
| v3relhub | relabel | hub | D14 | - | - |
| reluni, v3reluni | relabel | uniform | D14 | - | - |
| v3relunikeep | relabel | uniform | D14 | - | keep |

Four further 03a_01 logs, `03a_01_dose_shard{0..3}of4_20260930_211732.log`,
ran with `BIORAG_DOSES=0.002,0.01,0.05,0.1,1.25` over 4 shards. Their
`BIORAG_TAG` is not on the line that was read; see the logs.

## 03b_01 (`03_b_defence_vs_modify/03b_01_with_access_layer_matrix.py`)

All with `BIORAG_PHASE=graph` and `BIORAG_N_SHARDS=2`. The `v3*` tags also
set `BIORAG_SAVE_RETRIEVAL=1`.

| Tag | TYPES | DOSES |
|---|---|---|
| modhub | shifted:hub,synthesised:hub | D9 |
| v3modhub | shifted:hub,synthesised:hub | D13 |
| moduni, v3moduni | shifted:uniform,synthesised:uniform | D13 |
| v3modkeep | shifted:hub:keep,shifted:uniform:keep | D13 |

## Other steps

- **03a_02:** `BIORAG_ALL_CELLTYPES=1 BIORAG_TAG=v3`, 2 shards.
- **03a_03:** `BIORAG_TAG=v3`, 2 shards.
- **03a_09** (`03a_09_both_intervals.py`): `BIORAG_TAG=<tag>`, single
  process, one run per tag of 03a_01 and 03b_01 above. Also run for
  `v3moduni_rep2`, and for `v3` (the cached_clean pair).
- **03b_03 (repeatability)** uses `BIORAG_REP` and `BIORAG_TAG`. Its
  settings are not transcribed here.
