---
title: GCHIQL-Chunk
summary: Hierarchical GCHIQL with a chunk-conditioned low-level critic.
sidebar_title: GCHIQL-Chunk
---

# Goal-Conditioned Hierarchical IQL with Q-Chunking

GCHIQL-Chunk extends GCHIQL with the low-level Q-Chunking component of HiQC.
The high-level policy continues to predict latent waypoints and uses the
GCHIQL value-difference AWR weight.  The low-level policy predicts an action
chunk and is weighted by an explicit chunk critic.

For chunk length $k$, the low-level critic receives the complete dataset
action sequence $a_{t:t+k}$ and the same latent subgoal $z$ used by the
low-level policy:

$$
Q_L(s_t,a_{t:t+k},z) \leftarrow
R_t^{(k)} + \gamma^k V_L(s_{t+k},z).
$$

With the default negative goal-reaching reward, the implementation uses

$$
R_t^{(k)}=-\sum_{i=0}^{k-1}\gamma^i
$$

for a non-terminal chunk.  The low-level actor then uses

$$
w_L=\min\left(\exp\left(\alpha_L
[Q_L(s,a_{t:t+k},z)-V_L(s,z)]\right),100\right).
$$

The implementation uses a Gaussian chunk policy, matching the existing
GCHIQL baseline, rather than HiQC's optional flow-matching policy.
It also retains GCHIQL's shared twin value/representation module for the
high- and low-level objectives; only the low level adds an explicit critic.
For pixel observations, `low_actor_rep_grad=true` allows the low-level actor
loss to update this shared subgoal representation and is enabled by default.

## Temporal units

`frameskip` is the action chunk length in raw environment frames.  Dataset
actions are kept dense and reshaped to `frameskip * action_dim`, while sampled
observations are `frameskip` frames apart.  `subgoal_steps` is measured in
these sampled chunk steps.  Thus the default `frameskip=5` and
`subgoal_steps=10` give a physical high-level horizon of 50 raw frames.

The default configuration is
`scripts/train/config/gchiql_chunk.yaml`.  Launch the standard four-task suite
through the workspace-controlled bash entry point:

```bash
bash scripts/launch_gchiql_chunk_four_tasks.sh
```
