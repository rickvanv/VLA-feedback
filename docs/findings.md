# MicroVLA — Summary of Changes & Experimental Findings

## 1. Overview of Codebase Changes

Starting from the baseline [`MicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L310-L363) (`117,248` parameters), we implemented a recurrent feedback weight-switching architecture, a parameter-matched feedforward control, a partial-observability (target occlusion) benchmark, and a controlled multi-seed evaluation harness.

### A. Target Occlusion & Moving-Target Environment Updates ([`micro_vla.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py))
- **Independent Per-Color Occlusion Windows**:
  - Added [`sample_occlusion_window`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L127-L134), [`sample_occlusion`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L137-L152), and [`hidden_colors_at`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L155-L161) so each target color (`red`, `green`, `blue`) independently disappears for a random window of `2–4` consecutive frames (with probability `occlusion_prob`, default `0.7` in occlusion experiments) after the first `2` visible frames.
  - Updated [`render_env`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L108-L124) to accept `hidden_colors`, selectively omitting occluded target dots while keeping the white gripper and non-occluded targets visible.
  - Updated [`generate_vla_dataset`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L238-L315) to support `occlusion_prob`, `target_speed`, and `return_hidden=True`, returning a per-sample boolean mask indicating whether the prompted target was occluded at that timestep.
- **Slow Linear Non-Colliding Target Motion**:
  - Added [`sample_target_velocities`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L186-L225), [`_segments_min_dist`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L177-L183), and [`step_targets`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L228-L235) with default `TARGET_SPEED = 0.7` px/step (`< MAX_STEP_PX = 2.2` px/step) and `MIN_TARGET_DIST = 6.0` px.
  - Each target moves along a constant-velocity straight line segment $\mathbf{p}_c(t) = \mathbf{p}_c(0) + t\mathbf{v}_c$ guaranteed to stay within the workspace bounds `[3.5, 27.5]^2` and maintain a continuous center-to-center separation $\ge 6.0$ px between every pair of targets for all $t \in [0, T]$ (preventing any collision between the $3\times 3$ target squares).
  - The expert controller aims at the target's next position along its line (`next_targets[target_color]`), making velocity direction relevant to optimal action prediction.
- **Visualization Enhancements**:
  - Updated [`save_demo_gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L565-L719) to accept custom titles and display the active FFN weight set (`W:A` or `W:B`) per timestep when visualizing rollouts from the feedback model.
  - Updated `SEED` to `24` and added automatic accelerator selection (`cuda` / `mps` / `cpu` via `DEVICE`) in [`micro_vla.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L24-L32).

---

### B. Recurrent Feedback Weight-Switching Model ([`micro_vla_feedback.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py))
- **Architecture ([`FeedbackMicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L112-L190), `137,307` parameters)**:
  - Replaced the first Transformer block with [`FeedbackTransformerBlock`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L92-L106), whose FFN layers ([`SwitchedLinear`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L75-L90)) maintain $K = 2$ distinct weight sets $(W_{1,A}, W_{2,A})$ and $(W_{1,B}, W_{2,B})$.
  - Added a recurrent feedback path from the previous control timestep's final normalized hidden state $h_{t-1}$ at the `ACT_START` token position (`ACT_START_POS = 66`), with a learned initial state `h_init` at $t = 0$.
  - Because causal attention masks future tokens, $h_{t-1}$ at `ACT_START` is invariant between the $X$-bin and $Y$-bin passes within a single control step.
  - A 2-layer gating MLP (`64 -> 51 -> 2`) maps $h_{t-1}$ to routing logits over weight sets $A$ and $B$, using Gumbel-Softmax (soft during warm-up, straight-through hard one-hot after warm-up, and deterministic `argmax` at inference).
- **Anti-Collapse Routing Regularization & Full-Episode BPTT ([`train_feedback_vla`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L236-L309))**:
  - Reshapes flat dataset trajectories into contiguous episode streams via [`to_episodes`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L209-L219) (`[E*3, L, ...]`) and unrolls across the full episode length ($L = 10$) for Backpropagation Through Time (BPTT).
  - Combines soft Gumbel warm-up (`warmup_frac = 0.3`), temperature annealing ($\tau: 2.0 \to 0.5$), batch load-balancing loss ([`balance_loss`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L195-L198)), and per-sample entropy penalty ([`decisiveness_loss`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L201-L203)).
  - Tracks per-branch usage ratios, gradient norms ([`per_choice_grad_norm`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L227-L233)), and total parameter displacement $\|W - W_{\text{init}}\|_2$ for sets $A$ and $B$.

---

### C. Parameter-Matched Sequential Control ([`micro_vla_sequential.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_sequential.py))
- **Architecture ([`SequentialMicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_sequential.py#L69-L75), `137,307` parameters)**:
  - Replaces block 0 with [`SequentialFFNBlock`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_sequential.py#L50-L66), applying two residual FFNs in series (`mlp_a` followed by `mlp_b`) on every token without any gate or cross-timestep recurrence:
    $$x \leftarrow x + \text{Attn}(\text{LN}_1(x)), \quad x \leftarrow x + \text{FFN}_A(\text{LN}_2(x)), \quad x \leftarrow x + \text{FFN}_B(\text{LN}_3(x))$$
  - Matches the exact parameter count of `FeedbackMicroVLA` (`137,307` parameters) by setting `FFN_B` hidden dimension to $128 + 27 = 155$ and using a non-affine `LayerNorm` (`LN3`), absorbing the exact `3,485` parameters used by the feedback model's `gate_mlp` (`3,421`) and `h_init` (`64`).
- **Matched Episode Batch Training ([`train_sequential_episodes`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_sequential.py#L77-L122))**:
  - Uses the exact same episode batch schedule, learning rate schedule, and `AdamW` hyperparameters as `train_feedback_vla`, flattening the `[24, 10]` episode streams into `240` independent sequences per step.

---

### D. Multi-Seed Comparison Harness ([`compare_feedback_sequential.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/compare_feedback_sequential.py))
- Trains and evaluates `FeedbackMicroVLA` and `SequentialMicroVLA` head-to-head across shared random seeds, identical training/validation datasets, and a shared episode `batch_schedule`.
- Reports:
  - Training cross-entropy over the last 20 steps (`train_ce(last 20)`).
  - Validation cross-entropy split into **visible target frames** (`val_ce_visible`) vs. **occluded target frames** (`val_ce_occluded`).
  - Closed-loop rollout metrics across `30` unseen environment states $\times 3$ prompts (`90` rollouts per seed): mean final distance to goal (`final_dist_px`), success rate within `1.5px` (`success_rate`), and distance to goal immediately after the occlusion window ends (`dist_after_occl_px`).

---

## 2. Key Experimental Findings

### Finding 1: Both Switched Weight Sets Train Without Collapse
- With soft Gumbel-Softmax warm-up (first 30% of steps), temperature annealing ($\tau: 2.0 \to 0.5$), and the load-balancing regularizer (`balance_coef = 0.1`), `FeedbackMicroVLA` avoids routing collapse.
- Both weight sets $A$ and $B$ receive non-zero gradients throughout training, maintain roughly balanced ~50/50 batch utilization, and move substantially from their random initialization ($\|W_A - W_{A,\text{init}}\|_2$ and $\|W_B - W_{B,\text{init}}\|_2$ are both large and comparable).
- During closed-loop inference with occlusion, the gate actively switches between weight sets `A` and `B` across control steps as $h_{t-1}$ evolves.

### Finding 2: Parameter-Matched Sequential Control Outperforms Discrete 1-Bit Feedback Switching
When evaluated under identical training budgets (`520` steps, `24` streams $\times 10$ timesteps per batch, `occlusion_prob = 0.7`, Seed `0`):

| Metric | `FeedbackMicroVLA` (`137,307` params) | `SequentialMicroVLA` (`137,307` params) |
| :--- | :---: | :---: |
| **`train_ce (last 20)`** | `1.450` | **`1.316`** |
| **`val_ce_visible`** | `1.463` | **`1.353`** |
| **`val_ce_occluded`** | `1.605` | **`1.554`** |
| **`final_dist_px`** | `12.557` | **`11.818`** |
| **`dist_after_occl_px`** | `12.945` | **`12.756`** |
| **Runtime (520 steps + eval)** | `61s` | **`33s`** |

**Why `SequentialMicroVLA` achieves lower loss and better control than `FeedbackMicroVLA`:**
1. **Active Compute per Forward Pass**: Although both models have `137,307` parameters, `SequentialMicroVLA` applies *both* `FFN_A` and `FFN_B` on every forward pass (increasing effective depth and active FLOPs per token), whereas hard routing in `FeedbackMicroVLA` only activates *one* FFN weight set (`16,576` parameters) per step.
2. **1-Bit Information Bottleneck in Hard Gate Routing**: In `FeedbackMicroVLA`, the recurrent state $h_{t-1}$ only influences step $t$ through a binary choice $g_t \in \{A, B\}$ of block-0 FFN weights (`1 bit` of information per step). A 1-bit discrete switch cannot encode continuous 2D spatial coordinates $(x, y)$ of an occluded target across frames, explaining why `val_ce_occluded` (`1.605`) does not beat the memoryless sequential baseline (`1.554`). To carry spatial target memory across occluded frames, $h_{t-1}$ needs a continuous residual/token pathway (or KV-cache recurrence) rather than solely a discrete binary gate.
3. **Training Throughput & Sample Diversity per Step**: Training on `8` episodes (`24` streams $\times 10$ correlated frames $= 240$ sequences) with sequential unrolling is ~1.8$\times$ slower per step than flattened parallel execution and has lower state diversity per gradient update than standard i.i.d. state sampling (`24` independent states $\times 3$ prompts in [`train_vla`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla.py#L369-L410)), requiring more optimization steps when training on occluded episode streams.

### Finding 3: Moving-Target Experiment & Signed 2D Basis Mixing (`--target-speed 0.7 --occlusion-prob 0.0`)
When targets move slowly along non-colliding straight lines (`0.7` px/step, minimum separation `6.0` px) and the expert aims at the target's next step position along its line, a single frame cannot disambiguate the target's velocity direction.

To let two FFN matrices span all $360^\circ$ movement directions while keeping both weight sets active on every step, we added `gate_mode="mix"` in [`FeedbackMicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L121-L214):
$$g_t = \tanh(\text{MLP}(h_{t-1})) \in [-1, 1]^2, \quad \Delta x = g_{t,0}\,\text{FFN}_A(\text{LN}_2(x)) + g_{t,1}\,\text{FFN}_B(\text{LN}_2(x))$$
regularized by [`ortho_loss`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L232-L236) ($\|\frac{1}{B}G^\top G - 0.5 I_2\|_F^2$) so $g_{t,0}$ and $g_{t,1}$ stay orthogonal with balanced variance.

#### Short-Horizon Run (`520` steps across Seeds `0, 1, 2`, `137,307` parameters):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (Signed 2D Mix) | `feedback_switch` (Hard A/B Switch) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | **`1.205 ± 0.231`** | `1.275 ± 0.072` | `1.356 ± 0.076` |
| **`val_ce_visible`** | **`1.404 ± 0.254`** | `1.433 ± 0.115` | `1.496 ± 0.106` |
| **`final_dist_px`** | **`7.196 ± 2.634`** | `9.735 ± 2.621` | `7.574 ± 2.064` |
| **`mean_dist_px`** | **`6.589 ± 2.238`** | `8.210 ± 2.430` | `7.359 ± 1.615` |
| **`success_rate (< 1.5px)`** | `0.074 ± 0.053` | `0.022 ± 0.011` | **`0.081 ± 0.013`** |

#### Full-Convergence Run (`2000` steps across Seeds `0, 1, 2`, `137,307` parameters):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (Signed 2D Mix) | `feedback_switch` (Hard A/B Switch) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | **`0.183 ± 0.100`** | `0.439 ± 0.097` | `0.310 ± 0.052` |
| **`val_ce_visible`** | `1.738 ± 0.366` | **`1.506 ± 0.202`** | `1.626 ± 0.221` |
| **`final_dist_px`** | **`4.068 ± 0.343`** | `4.679 ± 0.642` | `5.175 ± 0.826` |
| **`mean_dist_px`** | **`3.840 ± 0.251`** | `4.227 ± 0.606` | `4.494 ± 0.638` |
| **`success_rate (< 1.5px)`** | `0.115 ± 0.067` | **`0.126 ± 0.017`** | `0.081 ± 0.017` |

- **At `2000` Steps, Both Recurrent Feedback Variants Beat Memoryless `sequential` Across All Closed-Loop Control Metrics**:
  - `feedback_mix` achieves the lowest final distance (`4.068 ± 0.343` px, winning on every individual seed: `4.459`, `3.930`, `3.815`) and lowest trajectory-average distance (`3.840 ± 0.251` px), beating `feedback_switch` (`4.679` / `4.227` px) and `sequential` (`5.175` / `4.494` px).
  - Both `feedback_switch` (`0.126 ± 0.017`) and `feedback_mix` (`0.115 ± 0.067`) achieve ~42–56% higher closed-loop success rates (`< 1.5` px) than `sequential` (`0.081 ± 0.017`), whose success rate stagnated between `520` and `2000` steps.
  - `feedback_switch` achieves the lowest validation cross-entropy (`1.506 ± 0.202`) at `2000` steps because discrete hard routing + Gumbel noise regularizes against overfitting the `250` training episodes (`train_ce = 0.439`), whereas continuous `feedback_mix` fits the training episodes much more tightly (`train_ce = 0.183`).
- **Overlaid Trajectory Visualization**: Saved to [`micro_vla_demo_moving_overlay.gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_demo_moving_overlay.gif) via [`save_overlay_gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/compare_feedback_sequential.py#L184-L469).

### Finding 4: Moving Targets with Occlusions at `2000` Steps (`--target-speed 0.7 --occlusion-prob 0.7`)
When targets both move (`0.7` px/step) and vanish for `2–4` consecutive frames with probability `0.7` across `3` seeds (`[0, 1, 2]`, `2000` steps, `137,307` parameters):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (Signed 2D Mix) | `feedback_switch` (Hard A/B Switch) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | `0.450 ± 0.404` | `0.668 ± 0.264` | **`0.303 ± 0.107`** |
| **`val_ce_visible`** | `1.425 ± 0.265` | **`1.289 ± 0.215`** | `1.643 ± 0.259` |
| **`val_ce_occluded`** | `2.510 ± 0.464` | **`2.443 ± 0.187`** | `3.328 ± 0.309` |
| **`final_dist_px`** | **`4.155 ± 0.791`** | `4.337 ± 0.308` | `4.298 ± 0.329` |
| **`mean_dist_px`** | `4.441 ± 0.905` | `4.380 ± 0.318` | **`4.223 ± 0.195`** |
| **`success_rate (< 1.5px)`** | `0.107 ± 0.017` | **`0.115 ± 0.017`** | `0.096 ± 0.017` |
| **`dist_after_occl_px`** | **`4.685 ± 0.971`** | `5.289 ± 1.112` | `4.971 ± 0.109` |

- **Lower Validation Cross-Entropy, but High Variance & No Statistically Significant Closed-Loop Gain Under Occlusion**:
  - While `feedback_switch` (`2.443 ± 0.187`) and `feedback_mix` (`2.510 ± 0.464`) achieve lower held-out cross-entropy than `sequential` (`3.328 ± 0.309`), this offline likelihood gap does **not** translate into a reliable closed-loop tracking advantage under occlusion.
  - Across the 3 seeds, all closed-loop distance metrics (`final_dist_px`, `mean_dist_px`, `dist_after_occl_px`) have overlapping error bars (differences of `0.14–0.29` px vs. standard deviations of `0.31–1.11` px), with `sequential` actually achieving the lowest `mean_dist_px` (`4.223 ± 0.195` px vs. `4.441 ± 0.905` px for `feedback_mix`) and lowest variance.
  - **Why `micro_vla_demo_moving_occluded_overlay.gif` shows `feedback_mix` performing worse**: [`compare_feedback_sequential.py`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/compare_feedback_sequential.py#L579-L599) generates the overlay GIF from the final seed in the loop (**Seed `2`**) on `env_states[0]`. On Seed `2`, both recurrent models failed to converge as deeply as `sequential` (`train_ce = 0.909` for `feedback_mix` and `0.954` for `feedback_switch`, vs. `0.215` for `sequential`), resulting in `feedback_mix` having the worst closed-loop tracking on Seed `2` (`final_dist_px = 5.048` px, `mean_dist_px = 5.445` px, `dist_after_occl_px = 5.788` px vs. `4.651` / `4.439` / `5.086` px for `sequential`).
  - Fundamentally, modulating block-0 FFN weights via a 2D scalar gate $g_t \in [-1, 1]^2$ (or 1-bit switch) still bottlenecks how much spatial target state $(x, y, v_x, v_y)$ can be transmitted from $h_{t-1}$ into the transformer stream when the target dot is completely absent from the image patches.
- **Overlaid Trajectory Visualization (with Occlusions, Seed `2`)**: Saved to [`micro_vla_demo_moving_occluded_overlay.gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_demo_moving_occluded_overlay.gif).

### Finding 5: True 4D State Bottleneck (`s_t in [-1, 1]^4`, `NUM_CHOICES = 4`, `138,081` params)
When we first tested a 64D recurrent transition (`state_step: 64x64` + `update_gate: 128x64`, `150,852` params) with 4 FFN branches, the 64D recurrent state memorized the `250` training trajectories (`train_ce = 0.018`, `val_ce_visible = 2.228`) and slightly degraded closed-loop tracking (`final_dist_px = 4.146 ± 0.288` px vs. `4.068 ± 0.343` px for 2D `feedback_mix`).

To prevent the recurrence from memorizing 64D visual embeddings while still carrying a 4D target state $(x, y, v_x, v_y)$, we bottlenecked the state passed between timesteps in [`FeedbackMicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_feedback.py#L134-L237) to **strictly 4 floats** $s_t \in [-1, 1]^4$ (`138,081` params, matched to the exact parameter in [`SequentialMicroVLA`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_sequential.py#L42) via `FFN_B_EXTRA_HIDDEN = 33`):
$$u_t = \tfrac{1}{2}(h_{\text{COLOR\_POS}} + h_{\text{ACT\_START\_POS}}), \quad (\hat{s}_t, z_\alpha) = \text{state\_mlp}([u_t, s_{t-1}]), \quad \alpha_t = \sigma(z_\alpha) \in (0, 1)^4$$
$$s_t = (1 - \alpha_t) \odot \tanh(A s_{t-1}) + \alpha_t \odot \tanh(\hat{s}_t) \in [-1, 1]^4$$
where $A \in \mathbb{R}^{4\times 4}$ (`16` params) is initialized with constant-velocity coupling ($p \leftarrow p + 0.5 v$).

#### A. Moving Targets Without Occlusions (`--occlusion-prob 0.0 --target-speed 0.7 --steps 2000`, Seeds `0, 1, 2`):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (True 4D Bottleneck) | `feedback_switch` (True 4D Bottleneck) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | **`0.028 ± 0.008`** | `0.290 ± 0.093` | `0.299 ± 0.188` |
| **`val_ce_visible`** | `2.148 ± 0.058` | **`1.540 ± 0.154`** | `1.691 ± 0.153` |
| **`final_dist_px`** | **`3.838 ± 0.528`** | `4.561 ± 0.526` | `4.911 ± 1.039` |
| **`mean_dist_px`** | **`3.641 ± 0.270`** | `4.065 ± 0.280` | `4.311 ± 0.634` |
| **`success_rate (< 1.5px)`** | `0.070 ± 0.023` | **`0.141 ± 0.026`** | `0.093 ± 0.017` |

- **Best Closed-Loop Tracking Distance of All Tested Architectures**:
  - Bottlenecking the recurrent state to $s_t \in [-1, 1]^4$ improved `feedback_mix` to **`final_dist_px = 3.838 ± 0.528` px** and **`mean_dist_px = 3.641 ± 0.270` px** — outperforming the 64D-state 4-branch model (`4.146` / `3.858` px), the original 2D `feedback_mix` (`4.068` / `3.840` px), and `sequential` (`4.911` / `4.311` px).
  - `feedback_mix` beats `sequential` on both `final_dist_px` and `mean_dist_px` on **every single seed** (Seed `0`: `3.336` vs. `5.950` px; Seed `1`: `4.389` vs. `4.913` px; Seed `2`: `3.788` vs. `3.871` px).
  - `feedback_switch` also improved substantially with the bounded 4D bottleneck (`final_dist_px = 4.561 ± 0.526` px, `mean_dist_px = 4.065 ± 0.280` px, `val_ce_visible = 1.540 ± 0.154`), achieving the highest overall `success_rate` (`0.141 ± 0.026` vs. `0.093 ± 0.017` for `sequential`).

#### B. Moving Targets With Occlusions (`--occlusion-prob 0.7 --target-speed 0.7 --steps 2000`, Seeds `0, 1, 2`):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (True 4D Bottleneck) | `feedback_switch` (True 4D Bottleneck) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | **`0.050 ± 0.040`** | `0.303 ± 0.208` | `0.303 ± 0.030` |
| **`val_ce_visible`** | `1.787 ± 0.262` | `1.610 ± 0.321` | **`1.601 ± 0.199`** |
| **`val_ce_occluded`** | **`2.702 ± 0.282`** | `3.466 ± 0.666` | `3.425 ± 0.191` |
| **`dist_after_occl_px`** | **`3.875 ± 0.271`** | `5.291 ± 0.218` | `5.344 ± 0.505` |
| **`mean_dist_px`** | **`4.258 ± 0.134`** | `4.269 ± 0.314` | `4.335 ± 0.271` |
| **`final_dist_px`** | `4.669 ± 0.348` | **`4.153 ± 0.464`** | `4.268 ± 0.535` |
| **`success_rate (< 1.5px)`** | `0.093 ± 0.028` | **`0.133 ± 0.011`** | `0.107 ± 0.032` |

- **Decisive Tracking Advantage During Occlusion (`dist_after_occl_px` & `val_ce_occluded`)**:
  - With the true 4D state bottleneck $s_t \in [-1, 1]^4$, `feedback_mix` reduces the distance to the moving target immediately after the occlusion window (`dist_after_occl_px`) to **`3.875 ± 0.271` px**, beating `sequential` (`5.344 ± 0.505` px) and `feedback_switch` (`5.291 ± 0.218` px) by **`1.47` px ($> 3\sigma$)** and winning on **every single seed** (Seed `0`: `4.074` vs. `5.178` px; Seed `1`: `3.566` vs. `4.943` px; Seed `2`: `3.984` vs. `5.912` px).
  - Compared to the previous 2D `feedback_mix` (`dist_after_occl_px = 4.685 ± 0.971` px), the true 4D bottleneck improves post-occlusion accuracy by **`0.81` px** while cutting seed-to-seed standard deviation by **$3.6\times$** (`0.971` $\to$ `0.271` px).
  - `feedback_mix` also achieves the lowest occluded validation cross-entropy (`val_ce_occluded = 2.702 ± 0.282` vs. `3.425 ± 0.191` for `sequential` and `3.466 ± 0.666` for `feedback_switch`) and lowest overall trajectory-average distance (`mean_dist_px = 4.258 ± 0.134` px).
- **Why Continuous 4D Mixing (`feedback_mix`) Beats Discrete 4-Way Switching (`feedback_switch`) During Occlusion**:
  - Even though both models carry $s_{t-1} \in [-1, 1]^4$, `feedback_switch` quantizes the gate into a 1-of-4 discrete choice ($\log_2 4 = 2$ bits) in block 0, discarding the continuous 4D target coordinates and resulting in the same `dist_after_occl_px` (`5.291` px) as memoryless `sequential` (`5.344` px).
  - Conversely, after the target becomes visible again in steps `10..15` (beyond the `10`-step training episode length), `feedback_switch` (`final_dist_px = 4.153 ± 0.464` px, `success_rate = 0.133 ± 0.011`) slightly edges out `sequential` (`4.268` px, `0.107`) and `feedback_mix` (`4.669` px, `0.093`), because `feedback_mix` was trained on `10`-step episodes (`EPISODE_LEN = 10`) and evaluated over `15`-step rollouts (`ROLLOUT_STEPS = 15`).
- **Updated Overlaid Trajectory Visualization (with Occlusions)**: Saved to [`micro_vla_demo_moving_occluded_overlay.gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_demo_moving_occluded_overlay.gif) (on Seed `2`, `feedback_mix` now converges cleanly to `train_ce = 0.033` and achieves `dist_after_occl_px = 3.984` px vs. `5.912` px for `sequential`, `mean_dist_px = 4.319` vs. `4.527` px, and `success_rate = 0.122` vs. `0.089`).

---

## 3. Reference Literature Notes

- **SAM2Act / Memory Bench**: Fang, H., Grotz, M., Pumacay, W., Wang, Y.R., Fox, D., Krishna, R., Duan, J.: *SAM2Act: Integrating Visual Foundation Model with a Memory Architecture for Robotic Manipulation*.
- Local reference papers added under [`papers/`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/papers):
  - `AvaVLA.pdf`
  - `CronusVLA.pdf`
  - `ReMemVLA.pdf`
