title: GCHIQL
summary: Goal-conditioned hierarchical implicit Q-learning in stable-worldmodel.
sidebar_title: GCHIQL
---

# Goal-Conditioned Hierarchical IQL

GCHIQL ports the core of
[HIQL](https://arxiv.org/abs/2307.11949) to the visual, transformer-based
goal-conditioned stack used by stable-worldmodel. The implementation follows
the current OGBench HIQL stabilization choices while retaining the module,
configuration, logging, and checkpoint style of the existing GCIQL baseline.

The main components are:

1. a learned, length-normalized subgoal representation
   $z_t^g=\phi(s_t,g)$;
2. twin action-free values $V_1(s,z)$ and $V_2(s,z)$ with EMA target copies;
3. a high-level policy $\pi^h(z_{t+k}\mid s_t,g)$;
4. a low-level policy $\pi^\ell(a_t\mid s_t,z_{t+k})$.

The HIQL paper writes the representation as $\phi(g)$ in the main text and
also evaluates the state-conditioned variant $\phi([g,s])$ in the appendix.
The OGBench implementation supplied with this project uses
$\phi([s,g])$. GCHIQL follows this OGBench variant because it is the requested
executable reference. Representations are normalized to
$\|\phi(s,g)\|_2=\sqrt{d_z}$, exactly as in that implementation.

## Hierarchical relabeling

For a sampled trajectory and hierarchy horizon $k$, GCHIQL constructs four
goal views.

- The value goal $g^V$ is sampled from the configured mixture of the current
  state, a geometric future state, a uniform future state, and a random state.
- The low-level goal is

  $$g_t^\ell=s_{\min(t+k,T)}.$$

- For a same-trajectory high-level goal $g^h=s_{t_g}$, the high-level target is

  $$\tilde{s}_{t+k}=s_{\min(t+k,t_g)}.$$

- For a random high-level goal, the reachable supervision target remains

  $$\tilde{s}_{t+k}=s_{\min(t+k,T)}.$$

The last rule is important: a random final goal may be outside the current
trajectory, but the high-level policy is still supervised with a reachable
$k$-step waypoint. The main clip contains `history_size + td_offset` states;
the aligned $k$-step goals are loaded separately by episode index and clipped
to the terminal state, as in OGBench HIQL. Thus transitions near the end of an
episode remain available instead of being discarded for lacking a full
hierarchy horizon.

For the default negative goal-reaching reward,

$$
r(s_t,g)=
\begin{cases}
0,&s_t=g,\\
-1,&s_t\ne g,
\end{cases}
\qquad
m(s_t,g)=\mathbb{1}[s_t\ne g].
$$

Equality is tested on the raw pixels and, when enabled, proprioception. This
avoids false mismatches caused by nondeterministic floating-point encoders.

## Twin value objective

Let $V_i$ be online value head $i$ and $\bar V_i$ its EMA teacher. Define

$$
y_i=r_t+\gamma m_t\bar V_i(s_{t+1},g),
$$

and the conservative target used only to choose the expectile side,

$$
y_{\min}=r_t+\gamma m_t
\min_{i\in\{1,2\}}\bar V_i(s_{t+1},g).
$$

The detached advantage is

$$
\bar A_V=y_{\min}
-\frac{\bar V_1(s_t,g)+\bar V_2(s_t,g)}{2}.
$$

For $L_\tau(u;A)=|\tau-\mathbb{1}[A<0]|u^2$, the value loss is

$$
\mathcal L_V=
\mathbb E\left[
L_\tau(y_1-V_1(s_t,g);\bar A_V)
+L_\tau(y_2-V_2(s_t,g);\bar A_V)
\right].
$$

Separating the teacher advantage used for the asymmetric weight from the two
online residuals is the double-estimation stabilization used in OGBench HIQL.
The whole value module, including $\phi$, is copied into the EMA teacher.

## Low-level actor

The low-level actor uses the same learned value to compare the dataset next
state against the current state for the nearby $k$-step goal:

$$
A^\ell_t=
\frac{1}{2}\sum_{i=1}^{2}
\left[V_i(s_{t+1},g_t^\ell)-V_i(s_t,g_t^\ell)\right].
$$

The advantage is detached, exponentiated, and clipped:

$$
w^\ell_t=\min\left(\exp(\beta_\ell A^\ell_t),100\right).
$$

For the Gaussian low-level policy, GCHIQL minimizes

$$
\mathcal L_\ell=
-\mathbb E\left[
w^\ell_t\log\pi^\ell
\left(a_t\mid s_t,\phi(s_t,g_t^\ell)\right)
\right].
$$

`low_actor_rep_grad` controls whether this loss may update $\phi$. It defaults
to `true` here because this implementation is pixel-based; OGBench identifies
this gradient path as crucial for maintaining useful visual subgoal
representations. The value loss always updates $\phi$.

## High-level actor

The high-level actor evaluates the sampled waypoint against the final goal:

$$
A^h_t=
\frac{1}{2}\sum_{i=1}^{2}
\left[V_i(\tilde{s}_{t+k},g^h)-V_i(s_t,g^h)\right],
$$

$$
w^h_t=\min\left(\exp(\beta_h A^h_t),100\right),
$$

and predicts the latent representation of that waypoint:

$$
\mathcal L_h=
-\mathbb E\left[
w^h_t\log\pi^h
\left(\phi(s_t,\tilde{s}_{t+k})\mid s_t,g^h\right)
\right].
$$

The high-level regression target is detached, as in OGBench HIQL. Both policy
heads use Gaussian likelihoods with fixed unit standard deviations, matching
OGBench HIQL's default `const_std=true`. At inference, `temperature` scales
these standard deviations when sampling.

The joint training objective is

$$
\mathcal L_{\mathrm{GCHIQL}}
=\mathcal L_V+\mathcal L_\ell+\mathcal L_h.
$$

## Inference

At inference time, no planner or Q-function is required:

$$
\hat z\sim\pi^h(\cdot\mid s,g),
\qquad
\hat z\leftarrow\sqrt{d_z}\frac{\hat z}{\|\hat z\|_2},
\qquad
a\sim\pi^\ell(\cdot\mid s,\hat z).
$$

`GCHIQL.get_action` implements this high-to-low composition and returns the
action for the last frame of the history window. Setting `sample=False`
selects both Gaussian means; `sample=True` samples both levels.

## Tensor mapping

With batch size $B$, history $H$, patches $P$, encoder dimension $D$, action
dimension $A$, and representation dimension $R$:

| Quantity | Shape |
|---|---|
| observation embedding | `(B, H, P, D)` |
| value/high final goal | `(B, 1, P, D)` |
| aligned low goal | `(B, H, P, D)` |
| aligned high target | `(B, H, P, D)` |
| $\phi(s,g)$ | `(B, H, R)` |
| each value head | `(B, H, 1)` |
| low-level mean | `(B, H, A)` |
| high-level mean | `(B, H, R)` |

The image encoder is DINOv2-small by default and frozen, matching GCIQL. A
trainable ViT-tiny can be selected with `encoder_type=vit_tiny`.

## Training

Before launching a GPU job in this workspace, inspect physical GPUs 4--7 and
select the least-loaded suitable device:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv -i 4,5,6,7
CUDA_VISIBLE_DEVICES=4 uv run python scripts/train/gchiql.py
```

`CUDA_VISIBLE_DEVICES=4` makes physical GPU 4 appear as local device 0 to the
trainer. Change the physical ID only after inspecting GPUs 4--7.

The default configuration is in `scripts/train/config/gchiql.yaml`. The most
important task-dependent settings are `subgoal_steps`, `rep_dim`,
`low_alpha`, `high_alpha`, and the two goal-probability mixtures.
`dinowm.td_offset` must remain 1 because both HIQL actor advantages use the
one-transition successor $s_{t+1}$.
