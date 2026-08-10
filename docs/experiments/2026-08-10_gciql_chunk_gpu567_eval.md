# StableWM GCIQL-chunk control evaluation

Date: 2026-08-10  
Server: Yingbo Cloud (`/root/data/yyf/stable-worldmodel`)  
Training checkpoint: epoch 10

## Evaluation protocol

- 50 evaluation episodes
- seed 777 and an independent repeat with seed 42
- goal offset 25 environment steps
- evaluation budget 50 actions
- DINO visual encoder
- observation history 3
- action chunk length 5
- deterministic feed-forward chunk policy

The evaluator samples a reachable future goal from the same offline trajectory
as the initial state. Each policy inference predicts a five-action chunk, which
is executed open-loop before the next policy query.

## Results

| Task | Seed | Successes | Success rate | Evaluation time |
|---|---:|---:|---:|---:|
| Reacher | 777 | 0 / 50 | 0% | 46.83 s |
| Reacher | 42 | 1 / 50 | 2% | 47.00 s |
| PushT | 777 | 0 / 50 | 0% | 28.85 s |
| PushT | 42 | 0 / 50 | 0% | 25.10 s |
| OGBench Cube | 777 | 46 / 50 | 92% | 61.62 s |
| OGBench Cube | 42 | 48 / 50 | 96% | 59.50 s |

All evaluations exited with status 0. The evaluator reconstructed each
checkpoint as `action_block=5` and `history_len=3`; these numbers are therefore
policy outcomes rather than process failures. Repeating Reacher and PushT with
an independent seed confirmed the result: pooled across 100 episodes, Reacher
achieved 1% and PushT achieved 0%. These checkpoints should be treated as
failed or near-failed policies and investigated separately. OGBench Cube was
stable across seeds, achieving 94% pooled success over 100 episodes.

## Checkpoints and raw results

All paths below are on Yingbo Cloud.

| Task | Policy checkpoint | Raw result |
|---|---|---|
| Reacher | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/reacher/checkpoints/gciql_chunk_reacher_dino_bs256_e10_policy/weights_epoch_10.pt` | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/reacher/checkpoints/gciql_chunk_reacher_dino_bs256_e10_policy/gciql_chunk_reacher_offset25_budget50_seed777_results.txt` |
| PushT | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/pusht/checkpoints/gciql_chunk_pusht_dino_bs256_e10_policy/weights_epoch_10.pt` | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/pusht/checkpoints/gciql_chunk_pusht_dino_bs256_e10_policy/gciql_chunk_pusht_offset25_budget50_seed777_results.txt` |
| OGBench Cube | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/ogbench_cube/checkpoints/gciql_chunk_ogbench_cube_dino_bs256_e10_policy/weights_epoch_10.pt` | `/root/data/yyf/stable-worldmodel/runs/gciql_chunk/ogbench_cube/checkpoints/gciql_chunk_ogbench_cube_dino_bs256_e10_policy/gciql_chunk_ogbench_cube_offset25_budget50_seed777_results.txt` |

The seed-42 rerun results are stored next to the corresponding checkpoints as
`gciql_chunk_reacher_offset25_budget50_seed42_results.txt`,
`gciql_chunk_pusht_offset25_budget50_seed42_results.txt`, and
`gciql_chunk_ogbench_cube_offset25_budget50_seed42_results.txt`.

Launcher logs, exit statuses, and the manifest are stored under:

`/root/data/yyf/stable-worldmodel/runs/gciql_chunk/eval_gpu567_20260810/`

Logs and exit statuses are separated into `seed_777/` and `seed_42/`
subdirectories for the rerun tasks.

Videos are stored next to each policy checkpoint.

## Reproduction Bash

Run:

```bash
cd /root/data/yyf/stable-worldmodel
bash scripts/0810_yb_eval_gciql_chunk_gpu567.sh
```

The launcher maps Reacher, PushT, and OGBench Cube to GPUs 5, 6, and 7,
respectively. On Yingbo Cloud it reuses the offline MuJoCo, dm_control, pygame,
and OGBench packages from `/root/data/yyf/ogbench/.venv`.
