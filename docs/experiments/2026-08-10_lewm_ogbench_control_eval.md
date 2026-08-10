# LeWM visual control policies evaluated with the OGBench protocol

Date: 2026-08-10  
Server: Yingbo Cloud (`/root/data/yyf`)  
Checkpoint step: 100,000

## Evaluation protocol

All policies were evaluated with the same dataset-based control protocol:

- 50 evaluation episodes
- seed 42
- goal offset 25 environment steps
- evaluation budget 50 actions
- deterministic policy sampling (`temperature=0`)
- initial state and reachable goal sampled from the same offline trajectory

The policies use the OGBench GCIQL/HIQL implementation with the
`impala_small` visual encoder. The tasks and offline visual datasets come from
the LeWM control benchmark. Stable World Model supplies the vectorized world
and dataset-based evaluator.

## Results

| Task | Policy | Successes | Success rate | Evaluation time |
|---|---|---:|---:|---:|
| PushT | OGBench GCIQL | 45 / 50 | 90% | 18.30 s |
| Cube | OGBench GCIQL | 44 / 50 | 88% | 56.90 s |
| Cube | OGBench HIQL | 42 / 50 | 84% | 68.74 s |
| PushT | OGBench HIQL | 33 / 50 | 66% | 18.11 s |

These are single-seed, 50-episode estimates. The difference between methods
should not be treated as statistically conclusive without evaluation over
additional seeds.

## Checkpoints and raw results

All paths below are on Yingbo Cloud.

| Task / policy | Checkpoint | Raw result JSON |
|---|---|---|
| PushT GCIQL | `/root/data/yyf/lewm-runs/OGBench/lewm-pusht-visual-gciql-bs256-100k/sd000_20260809_021452/params_100000.pkl` | `/root/data/yyf/lewm-runs/OGBench/lewm-pusht-visual-gciql-bs256-100k/sd000_20260809_021452/eval_ff/pusht_gciql_step100000.json` |
| Cube GCIQL | `/root/data/yyf/lewm-runs/OGBench/lewm-cube-visual-gciql-bs256-100k/sd000_20260809_015848/params_100000.pkl` | `/root/data/yyf/lewm-runs/OGBench/lewm-cube-visual-gciql-bs256-100k/sd000_20260809_015848/eval_ff/cube_gciql_step100000.json` |
| Cube HIQL | `/root/data/yyf/lewm-runs/OGBench/lewm-cube-visual-hiql-bs256-100k/sd000_20260809_015848/params_100000.pkl` | `/root/data/yyf/lewm-runs/OGBench/lewm-cube-visual-hiql-bs256-100k/sd000_20260809_015848/eval_ff/cube_hiql_step100000.json` |
| PushT HIQL | `/root/data/yyf/lewm-runs/OGBench/lewm-pusht-visual-hiql-bs256-100k/sd000_20260809_021452/params_100000.pkl` | `/root/data/yyf/lewm-runs/OGBench/lewm-pusht-visual-hiql-bs256-100k/sd000_20260809_021452/eval_ff/pusht_hiql_step100000.json` |

Evaluation launcher logs and exit codes are stored under:

`/root/data/yyf/lewm-runs/evals/gpu567_20260810/`

The four evaluations all exited with status 0. Videos are stored in each
checkpoint directory under `eval_ff/videos/`.

## Reproduction Bash

- `scripts/0810_yb_eval_lewm_gpu567.sh` evaluates PushT GCIQL, Cube HIQL, and
  PushT HIQL concurrently on GPUs 5, 6, and 7.
- `scripts/0810_yb_eval_lewm_cube_gciql.sh` evaluates Cube GCIQL on GPU 7.

Both launchers call `/root/data/yyf/ogbench/scripts/eval_lewm.sh`, which pins
the evaluation protocol above.
