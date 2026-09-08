# LDP–OGBench Experiment Integrity Audit

- Audit snapshot: 2026-09-08 08:27–08:36 CST
- Verdict: **WARN**
- Acceptance: **provisional**
- Review independence: **same-family**

The eight launched jobs are genuine training runs over eight distinct offline
dataset configurations (four visual manipulation environments crossed with
`play` and `noisy`).  They were all still in VAE training at the audit snapshot,
so there are no admissible formal LDP or evaluation results yet.

## A. Ground-truth provenance — PASS

Training goals are stored segment-final dataset frames, not model predictions.
Evaluation goals and success are supplied by the OGBench simulator.  The source
datasets independently passed shape, action range, episode-boundary, reward
(`-1 ... -1, 0`), and source-alignment checks.

Artifact identity was initially bound mainly by paths and byte sizes.  The final
completion audit must retain exact adapter, upstream LDP, and official OGBench
commits together with artifact hashes.

## B. Score normalization — PASS

Success rate is the arithmetic mean of raw simulator success booleans, with raw
per-episode outputs retained.  No score is normalized by the evaluated model's
own predictions.  VAE PSNR uses the fixed `[-1, 1]` image range.

## C. Result existence — WARN

Seven matrix smoke results and one earlier cube-single smoke exist and contain
real end-to-end VAE, latent encoding, LDP, and simulator evaluation evidence.
At the snapshot, all eight formal jobs were live but had no `VAE_COMPLETE`, full
latent file, final LDP checkpoint, or formal `results.json`.

## D. Executed-path audit — WARN

The current-plus-final-goal condition is used in both diffusion training and
sampling, and the sampled plan feeds the inverse-dynamics model.  The audit
found missing loader-side terminal/source-episode validation, a weak environment
marker check, and overly broad log anomaly matching.  These have been repaired
in the pending protocol update and require remote smoke verification.

## E. Scope — WARN

The old evaluator directly called `gymnasium.make`, fixed `reward_task_id=2`,
and overrode every horizon to 50.  Official OGBench instead uses
`ogbench.make_env_and_datasets(..., env_only=True)`, evaluates task IDs 1–5,
and registers native horizons 200/500/1000/750 for cube-single/cube-double/
cube-triple/scene.  Therefore the prior smoke results are custom protocol checks,
not official OGBench aggregates.

The repaired formal scope is:

- 8 offline dataset configurations, one per GPU;
- 5 official evaluation tasks per dataset configuration;
- 10 episodes per task (50 episodes per configuration);
- one training seed, so no multi-seed robustness claim.

## F. Evaluation type — PASS

Classification: `simulation_only`, with benchmark-provided simulator ground
truth.  There is no human evaluation or model-generated reference score.

## Required completion gates

1. Run the repaired five-task/native-horizon evaluator against the pinned clean
   official OGBench checkout.
2. Require step-300k VAE checkpoints, complete latent files with train-only
   normalization bounds, step-500k LDP checkpoints, and 10 episodes for each of
   five official tasks on all eight dataset configurations.
3. Validate raw arrays, per-task and overall success-rate consistency, finite
   metrics, provenance, hashes, and fatal log anomalies with the tracked
   completion auditor.
4. Do not claim robustness or final benchmark performance until these gates pass.
