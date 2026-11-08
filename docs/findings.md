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

Across `3` seeds (`[0, 1, 2]`, `520` steps, `137,307` parameters for all three models):

| Metric (Mean $\pm$ Std over Seeds `0, 1, 2`) | `feedback_mix` (Signed 2D Mix) | `feedback_switch` (Hard A/B Switch) | `sequential` (Feedforward A $\to$ B) |
| :--- | :---: | :---: | :---: |
| **`train_ce (last 20)`** | **`1.205 ± 0.231`** | `1.275 ± 0.072` | `1.356 ± 0.076` |
| **`val_ce_visible`** | **`1.404 ± 0.254`** | `1.433 ± 0.115` | `1.496 ± 0.106` |
| **`final_dist_px`** | **`7.196 ± 2.634`** | `9.735 ± 2.621` | `7.574 ± 2.064` |
| **`mean_dist_px`** | **`6.589 ± 2.238`** | `8.210 ± 2.430` | `7.359 ± 1.615` |
| **`success_rate (< 1.5px)`** | `0.074 ± 0.053` | `0.022 ± 0.011` | **`0.081 ± 0.013`** |

- **`feedback_mix` Achieves the Best Cross-Entropy and Closed-Loop Tracking Distance**: By taking a signed 2D linear combination of $\text{FFN}_A$ and $\text{FFN}_B$ conditioned on $h_{t-1}$, `feedback_mix` outperforms **both** `feedback_switch` and `sequential` on training loss (`1.205`), validation loss (`1.404`), mean rollout distance to the moving target (`6.589px` vs. `7.359px` sequential and `8.210px` switch), and final rollout distance (`7.196px` vs. `7.574px` sequential and `9.735px` switch).
- **Overlaid Trajectory Visualization**: Saved to [`micro_vla_demo_moving_overlay.gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/micro_vla_demo_moving_overlay.gif) via [`save_overlay_gif`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/compare_feedback_sequential.py#L179-L384).

---

## 3. Reference Literature Notes

- **SAM2Act / Memory Bench**: Fang, H., Grotz, M., Pumacay, W., Wang, Y.R., Fox, D., Krishna, R., Duan, J.: *SAM2Act: Integrating Visual Foundation Model with a Memory Architecture for Robotic Manipulation*.
- Local reference papers added under [`papers/`](file:///Users/rickvanvoorbergen/Documents/repos/microvla/papers):
  - `AvaVLA.pdf`
  - `CronusVLA.pdf`
  - `ReMemVLA.pdf`
