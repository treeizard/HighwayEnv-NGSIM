# GAIL Methodology and Implementation

This note describes the methodology used for the project GAIL testing system. It is intended to complement the mathematical background in `Cognitive Driving Simulator and GAIL (May 2026)` by making the link from the GAIL objective to this repository's implementation explicit.

## Methodological Aim

The testing system trains an imitation policy for NGSIM-style highway driving without manually specifying a dense driving reward. Instead, expert demonstrations are replayed through `NGSimEnv`, the learner policy is rolled out in the same simulator, and a discriminator/critic provides the reward signal used to update the policy. This follows the central idea of Generative Adversarial Imitation Learning (GAIL): match the learner occupancy measure to the expert occupancy measure rather than only matching isolated expert actions.

The core implementation lives in:

- `scripts_gail/train_simple_ps_gail.py`
- `scripts_gail/ps_gail/`
- `highway_env/imitation/expert_dataset.py`
- `highway_env/envs/ngsim_env.py`
- `highway_env/ngsim_utils/expert/ngsim_expert_mixin.py`

## Formal Problem Setup

The simulator is treated as a discounted Markov decision process:

$$
M = (S, A, P, p_0, \gamma, r),
$$

where `S` is the driving state space, `A` is the action space, `P(s' | s, a)` is the simulator transition model, `p_0` is the initial scenario distribution, and `\gamma` is the discount factor. In this project the true expert reward is unknown, so the system learns from demonstration trajectories:

$$
D_E = \{\tau_i\}_{i=1}^{N}, \qquad
\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T).
$$

The discounted occupancy measure of a policy `\pi` is:

$$
\rho_{\pi}(s,a) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr_{\pi}(s_t=s, a_t=a).
$$

Occupancy matching is the key reason to use GAIL instead of plain behavior cloning. If `\rho_\pi \approx \rho_E`, then the learned policy visits similar states and takes similar actions over time, including under states induced by its own rollouts. This directly targets the distribution-shift problem that behavior cloning suffers from.

## Classical GAIL Objective

GAIL formulates imitation as a two-player minimax problem between a policy `\pi_\theta` and discriminator `D_\phi`. With `D_\phi(s,a)` interpreted as the probability that a transition came from the expert, the binary-classification version is:

$$
\min_{\pi_\theta} \max_{D_\phi}
\mathbb{E}_{(s,a)\sim\rho_E}[\log D_\phi(s,a)]
+ \mathbb{E}_{(s,a)\sim\rho_{\pi_\theta}}[\log(1-D_\phi(s,a))]
- \lambda H(\pi_\theta).
$$

For a fixed policy, the optimal discriminator is:

$$
D^*(s,a) =
\frac{\rho_E(s,a)}{\rho_E(s,a) + \rho_\pi(s,a)}.
$$

Substituting `D^*` gives a Jensen-Shannon-style divergence between expert and learner occupancy measures. This is the connection to GANs: the policy is the generator, and the discriminator compares expert occupancy samples against learner occupancy samples.

The practical reward used in binary GAIL is commonly:

$$
r_D(s,a) = -\log(1-D_\phi(s,a)).
$$

In code, if `D_\phi(s,a)=\sigma(f_\phi(s,a))`, then:

$$
-\log(1-D_\phi(s,a)) = \mathrm{softplus}(f_\phi(s,a)).
$$

This is exactly the BCE reward branch in `discriminator_reward()`.

## Implementation Choice: WGAN-GP by Default

The repository keeps the BCE GAIL pathway, but the default configuration uses a Wasserstein critic with gradient penalty:

```python
discriminator_loss: str = "wgan_gp"
```

In the WGAN-GP setting, the discriminator is interpreted as a critic `f_\phi(x)` rather than a probability. The critic objective implemented here is:

$$
L_D =
\mathbb{E}_{x\sim\rho_\pi}[f_\phi(x)]
- \mathbb{E}_{x\sim\rho_E}[f_\phi(x)]
+ \lambda_{gp}\mathbb{E}_{\hat{x}}
\left(\lVert \nabla_{\hat{x}}f_\phi(\hat{x}) \rVert_2 - 1\right)^2,
$$

where `\hat{x}` is sampled on straight lines between expert and generator features. The policy reward is then the critic score:

$$
r_D(x) = f_\phi(x).
$$

This is a pragmatic stability choice for the testing system. Classical BCE GAIL can saturate when the discriminator becomes confident early; WGAN-GP gives a smoother ranking signal and explicit Lipschitz regularization. The implementation follows the WGAN-GP idea of replacing weight clipping with a gradient penalty.

## Expert Data Pipeline

Expert demonstrations are not read as detached CSV rows. They are replayed through the same `NGSimEnv` environment used for training. This is important because it makes the expert observations and actions match the simulator's actual observation/action contract.

The replay collection pathway is:

1. `build_env_config()` enables `expert_test_mode=True` and `truncate_to_trajectory_length=True`.
2. `build_expert_dataset()` samples fixed NGSIM episodes and runs them in the simulator.
3. `NGSimEnv` initializes a `PurePursuitTracker` for each expert vehicle.
4. `_resolve_expert_action()` converts the replay target into either continuous normalized control or a discrete meta-action.
5. `env.step()` exposes expert actions in the `info` dictionary.
6. The dataset stores observations, actions, next observations, rewards, done flags, vehicle ids, timesteps, and metadata.

The code path is:

- `highway_env/imitation/expert_dataset.py:115` builds replay-mode environment config.
- `highway_env/imitation/expert_dataset.py:391` collects and saves expert datasets.
- `highway_env/envs/ngsim_env.py:906` initializes `PurePursuitTracker`.
- `highway_env/ngsim_utils/expert/ngsim_expert_mixin.py:637` resolves expert actions.
- `highway_env/envs/ngsim_env.py:1323` exposes expert actions in `info`.

This design supports both `per_vehicle` datasets for transition-level imitation and `scene` datasets for multi-agent or scene-level discriminators.

## Feature Representation

The policy observation is derived from the environment observation using:

- `flatten_agent_observations()`
- `policy_observations_from_flat()`

For the default discrete PS-GAIL configuration, the discriminator does not consume the discrete meta-action directly. Instead, it uses:

$$
x_t = [o_t,\; \tilde{p}_t,\; v_t],
$$

where `o_t` is the policy observation and `[\tilde{x}_t,\tilde{y}_t,v_t]` is the normalized trajectory state. Relative trajectory coordinates are the default, so trajectory position is measured relative to the start of each controlled vehicle trajectory. This prevents the discriminator from simply memorizing absolute map location.

The continuous-action pathway can instead use action-conditioned features:

$$
x_t = [o_t,\; a_t].
$$

The implementation anchors are:

- `scripts_gail/ps_gail/data.py:71` builds `[policy_observation, trajectory_state]` discriminator features.
- `scripts_gail/ps_gail/data.py:414` loads policy and discriminator data from expert files.
- `scripts_gail/ps_gail/data.py:503` loads full transition data for action-conditioned or BC-regularized variants.
- `scripts_gail/ps_gail/training/rewards.py:286` builds action-conditioned continuous features.

Optional feature standardization is enabled by default for discriminator inputs. The config clips standardized features to avoid unstable outliers.

## Models

The policy is an actor-critic network. The default is an MLP policy, with optional transformer and recurrent-transformer variants:

- `scripts_gail/ps_gail/models.py:162` defines `SharedActorCritic`.
- `scripts_gail/ps_gail/models.py:252` defines `TransformerActorCritic`.
- `scripts_gail/ps_gail/models.py:395` defines `RecurrentTransformerActorCritic`.
- `scripts_gail/ps_gail/models.py:707` selects the model in `make_actor_critic()`.

The discriminator family is:

- `scripts_gail/ps_gail/models.py:784` for transition/trajectory MLP discrimination.
- `scripts_gail/ps_gail/models.py:807` for scene snapshot discrimination.
- `scripts_gail/ps_gail/models.py:811` for sequence discrimination using a GRU over trajectory-feature windows.

The default discriminator hidden layout is configured as:

```python
discriminator_hidden_sizes: str = "128,128,64"
```

## Round-by-Round Training Algorithm

The training loop in `scripts_gail/train_simple_ps_gail.py` performs:

1. Load expert feature samples and metadata.
2. Construct the training environment.
3. Build the actor-critic policy and discriminator/critic.
4. For each training round:
   - collect current-policy rollouts from `NGSimEnv`;
   - construct generator discriminator features;
   - update the discriminator/critic against expert features;
   - convert discriminator output into rollout rewards;
   - compute returns and advantages;
   - update the policy with PPO;
   - log metrics, evaluate, and checkpoint.

The main implementation anchors are:

- `scripts_gail/train_simple_ps_gail.py:644` starts training setup.
- `scripts_gail/train_simple_ps_gail.py:1049` enters the training-round loop.
- `scripts_gail/train_simple_ps_gail.py:1079` collects policy rollouts.
- `scripts_gail/train_simple_ps_gail.py:1112` updates the discriminator.
- `scripts_gail/train_simple_ps_gail.py:1146` refreshes adversarial rollout rewards.
- `scripts_gail/train_simple_ps_gail.py:1161` updates the policy.
- `scripts_gail/train_simple_ps_gail.py:1563` runs validation.
- `scripts_gail/train_simple_ps_gail.py:1745` runs final test evaluation.

## Discriminator Update

`update_discriminator()` supports two objectives.

For BCE GAIL, the discriminator receives expert labels near `0.9` and generator labels near `0.1`. With `D_\phi(x)=\sigma(f_\phi(x))`, the implemented soft-label BCE loss is:

$$
L_D^{BCE}
=
-\mathbb{E}_{(x,y)}
\left[
y\log D_\phi(x) + (1-y)\log(1-D_\phi(x))
\right]
+ \frac{k}{2}\mathbb{E}[(D_\phi(x)-0.5)^2],
$$

where `y=y_E` for expert samples, `y=y_\pi` for generator samples, and the final term is an optional confidence penalty controlled by `cgail_k`.

For WGAN-GP, the critic minimizes:

$$
L_D^{WGAN}
= \mathbb{E}_{x\sim\rho_\pi}[f_\phi(x)]
- \mathbb{E}_{x\sim\rho_E}[f_\phi(x)]
+ \lambda_{gp}\mathbb{E}_{\hat{x}}
\left(\lVert \nabla_{\hat{x}}f_\phi(\hat{x}) \rVert_2 - 1\right)^2.
$$

Code anchors:

- `scripts_gail/ps_gail/training/discriminator.py:43` computes the WGAN-GP penalty.
- `scripts_gail/ps_gail/training/discriminator.py:226` updates the discriminator/critic.
- `scripts_gail/ps_gail/training/discriminator.py:134` optionally selects hard expert/generator examples.

## Reward Construction

After discriminator training, rollout rewards are recomputed from the current discriminator/critic:

$$
r_t =
\mathrm{clip}\left(
\mathrm{shape}(r_D(x_t) + \alpha_s r_s(x_t) + \alpha_q r_q(x_{t:t+K}))
+ r_{env\_penalty,t}
+ r_{challenge,t}
\right).
$$

The simple transition discriminator gives `r_D`. Optional scene and sequence discriminators can add scene-level and temporal-window rewards. Collision and off-road penalties remain explicit safety terms, which makes the testing system less likely to accept policies that fool the discriminator while breaking simulator safety constraints.

Code anchors:

- `scripts_gail/ps_gail/training/rewards.py:68` converts discriminator output to adversarial reward.
- `scripts_gail/ps_gail/training/rewards.py:322` shapes, scales, normalizes, and clips adversarial rewards.
- `scripts_gail/ps_gail/training/rollouts.py:242` combines transition, scene, sequence, safety, and challenge rewards.

## Policy Update with PPO

The policy is updated using PPO rather than the original TRPO-style optimizer. PPO is used because it keeps the trust-region intuition of limiting policy changes while being simpler to implement and easier to batch on this codebase's rollout tensors.

For each sampled rollout transition, the likelihood ratio is:

$$
r_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}.
$$

The clipped PPO actor objective is:

$$
L^{CLIP}(\theta)
= \mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat{A}_t,\;
\mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right].
$$

The implementation minimizes the negative actor objective plus value loss minus entropy:

$$
L_{\pi}
= -L^{CLIP}
+ c_v\lVert V_\theta(s_t)-\hat{R}_t\rVert^2
- c_H H(\pi_\theta(\cdot|s_t)).
$$

Advantages are computed with generalized advantage estimation:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
\qquad
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}.
$$

Code anchors:

- `scripts_gail/ps_gail/training/rewards.py:44` computes returns and GAE advantages.
- `scripts_gail/ps_gail/training/ppo.py:382` performs PPO updates.
- `scripts_gail/ps_gail/training/policy.py` handles distributions, action masking, continuous actions, and centralized-critic inputs.

## Rollout and Testing System

The rollout code controls one or more NGSIM vehicles with the learned policy while retaining replay/background traffic from the environment. It records:

- policy observations and next observations;
- actions and old action log probabilities;
- critic values;
- trajectory ids and vehicle ids;
- normalized trajectory states;
- crash/offroad penalties;
- optional scene and sequence features.

The code path is:

- `scripts_gail/ps_gail/training/rollouts.py:366` collects rollouts.
- `scripts_gail/ps_gail/training/rollouts.py:701` builds generator features.
- `scripts_gail/ps_gail/training/rollouts.py:715` initializes rollout rewards before discriminator refresh.
- `scripts_gail/ps_gail/training/rollouts.py:780` merges rollout batches across workers.

The testing system evaluates trained policies against held-out prebuilt splits using matched trajectory metrics. The final test printout includes 20-second position RMSE, crash rate, offroad rate, and hard-brake rate. This matters because discriminator reward alone is not enough evidence of driving quality; matched trajectory metrics provide a simulator-grounded behavioral check against the original NGSIM trajectory.

Evaluation anchors:

- `scripts_gail/ps_gail/training/evaluation.py:1110` evaluates matched trajectories.
- `scripts_gail/ps_gail/validation.py:88` converts validation metrics into a model-selection score.
- `scripts_gail/train_simple_ps_gail.py:1745` runs final test evaluation.

Regression tests also cover the GAIL-specific choices:

- `tests/test_ps_gail_training_logic.py:301` checks unified action-conditioned expert loading.
- `tests/test_ps_gail_training_logic.py:355` checks the discriminator architecture.
- `tests/test_ps_gail_training_logic.py:745` checks discriminator feature normalization during reward refresh.
- `tests/test_ps_gail_training_logic.py:837` checks BCE reward clipping.
- `tests/test_ps_gail_training_logic.py:1358` checks WGAN-GP discriminator metrics.
- `tests/test_ps_gail_training_logic.py:1387` checks WGAN-GP sequence discriminator support.

## Traceability Matrix

| Methodological choice | Mathematical role | Code anchor |
| --- | --- | --- |
| Expert replay through simulator | Samples from `\rho_E` under the same observation/action interface as training | `highway_env/imitation/expert_dataset.py:391` |
| Occupancy feature vector `[o_t, x_t, y_t, v_t]` | Approximate state/trajectory occupancy matching for discrete control | `scripts_gail/ps_gail/data.py:71` |
| Action-conditioned continuous feature `[o_t, a_t]` | Classical state-action GAIL feature for continuous control | `scripts_gail/ps_gail/training/rewards.py:286` |
| WGAN-GP critic default | Stabilized occupancy divergence estimate | `scripts_gail/ps_gail/training/discriminator.py:43` |
| Adversarial reward from discriminator | Learned reward/cost replacing hand-designed expert reward | `scripts_gail/ps_gail/training/rewards.py:68` |
| PPO update | Policy-gradient optimizer for discriminator-derived rewards | `scripts_gail/ps_gail/training/ppo.py:382` |
| Collision/offroad penalties | Safety constraints retained during adversarial imitation | `scripts_gail/ps_gail/training/rollouts.py:575` |
| Matched validation/test metrics | External behavioral check against held-out trajectories | `scripts_gail/ps_gail/training/evaluation.py:1110` |

## Literature Links

- Ho and Ermon, [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476), NeurIPS 2016.
- Goodfellow et al., [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets), NeurIPS 2014.
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), 2015.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Gulrajani et al., [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028), NeurIPS 2017.
- Ziebart et al., [Maximum Entropy Inverse Reinforcement Learning](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html), AAAI 2008.
- Abbeel and Ng, [Apprenticeship Learning via Inverse Reinforcement Learning](https://ai.stanford.edu/~pabbeel/pubs/AbbeelNg_alvirl_ICML2004.pdf), ICML 2004.
- Ross, Gordon, and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.
- FHWA, [Next Generation Simulation (NGSIM) Open Data](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/).
- Leurent, [highway-env documentation and citation](https://lavinama.github.io/highway-env/).

## Suggested Methodology Paragraph

This project implements GAIL as an occupancy-matching testing system for NGSIM highway-driving scenarios. Expert demonstrations are generated by replaying processed NGSIM trajectories through `NGSimEnv`, so the expert data shares the same observation, action, and vehicle-dynamics interface as learner rollouts. During training, the current policy is rolled out in the simulator to produce generator occupancy samples. A discriminator/critic is then trained to distinguish expert features from learner features. The resulting discriminator score is converted into an adversarial reward, combined with explicit collision/offroad penalties, and used to update an actor-critic policy with PPO. The default discriminator objective is WGAN-GP rather than classical BCE GAIL, because the Wasserstein critic and gradient penalty provide a smoother and more stable reward signal when simulator rollouts and expert demonstrations are high-dimensional and multi-vehicle. Policy quality is evaluated not only by adversarial training metrics but also by matched held-out trajectory metrics, including position RMSE, crash rate, offroad rate, and hard-brake rate.
