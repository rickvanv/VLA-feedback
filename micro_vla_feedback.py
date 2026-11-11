"""
micro_vla_feedback.py: MicroVLA with a recurrent feedback connection that switches weights.

Built on top of micro_vla.py (env, tokenizer, data and visualization are reused from there).

What is new
-----------
The FFN of the *first* Transformer block owns FOUR half-width basis weight sets
(A, B, C, D) to span the 4-dimensional moving-target state (x, y, v_x, v_y). How those 4 sets
are combined or selected at control step t is decided by a gating MLP that reads the model's
*recurrent state from the previous step t-1*:

  - gate_mode='mix':    4D signed basis combination g_t = tanh(MLP(h_{t-1})) in [-1, 1]^4
                        Delta_x = sum_{k=1..4} g_t[k] * FFN_k(LN2(x))
  - gate_mode='switch': discrete selection of one of the 4 weight sets via Gumbel-softmax

To prevent multi-frame occlusions (2-4 frames) from overwriting the remembered target state
while avoiding 64D recurrent overfitting, the model carries a strict 4D state bottleneck
s_t in [-1, 1]^4 across timesteps via a gated 4x4 state-space transition:
    u_t           = 0.5 * (h_final[COLOR_POS] + h_final[ACT_START_POS])
    s_hat, z_gate = state_mlp([u_t, s_{t-1}]).chunk(2, dim=-1)
    alpha_t       = sigmoid(z_gate)                          in (0, 1)^4
    s_t           = (1 - alpha_t) * tanh(A s_{t-1}) + alpha_t * tanh(s_hat)   in [-1, 1]^4
Because attention is causal, both COLOR_POS and ACT_START_POS are identical for the X-pass
and the Y-pass of a step, and reading from COLOR_POS decouples the 4D spatial memory from
the 1-step action classification head (lm_head) at ACT_START_POS.

Making sure BOTH weight sets are trained
----------------------------------------
Hard routing can collapse onto one weight set (the other then never receives gradients).
We prevent this with:
  1. Soft warm-up: for the first part of training the Gumbel-softmax is used *soft*, so both
     weight sets receive gradient on every sample (weighted by their routing probability).
  2. Temperature annealing tau: 2.0 -> 0.5, after which the gate becomes a hard
     straight-through one-hot (forward picks exactly one matrix, backward still trains the gate).
  3. Load-balancing loss: log(2) - H(mean_batch p) pushes the batch-average routing towards 50/50,
     so each weight set keeps being selected (and hence updated) for ~half of the samples.
  4. Decisiveness loss: small per-sample entropy penalty so routing becomes a real decision
     conditioned on h_{t-1}, not a coin flip driven only by Gumbel noise.
  5. Monitoring: per-set usage and per-set gradient norms are printed during training, and the
     distance each weight set moved from its initialization is reported at the end.

Training uses full-episode unrolls (BPTT through the feedback path) so that h_{t-1} during
training comes from exactly the same recurrence as during closed-loop inference.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from micro_vla import (
    COLORS,
    DEVICE,
    IMG_SIZE,
    MAX_STEP_PX,
    NUM_BINS,
    NUM_PATCHES,
    PATCH_SIZE,
    TARGET_SPEED,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    CausalSelfAttention,
    MicroPatchEmbed,
    TransformerBlock,
    bin_to_action,
    generate_vla_dataset,
    hidden_colors_at,
    itos,
    print_ascii_rollout,
    render_env,
    sample_env_state,
    sample_target_velocities,
    save_demo_gif,
    step_targets,
    stoi,
)

NUM_CHOICES = 4  # 4 FFN basis branches to span 4D target state (x, y, v_x, v_y)
COLOR_POS = NUM_PATCHES + 1  # index of <color> in [patches.., "reach", color, ACT_START, ..]
ACT_START_POS = NUM_PATCHES + 2  # index of ACT_START in [patches.., "reach", color, ACT_START, ..]


# ------------------------------------------------------------------------------
# Switched layers
# ------------------------------------------------------------------------------
class SwitchedLinear(nn.Module):
    """Linear layer with K alternative weight matrices; a per-sample gate g [B, K] picks/mixes them."""

    def __init__(self, d_in: int, d_out: int, num_choices: int = NUM_CHOICES):
        super().__init__()
        # Initialize each weight set independently (same scheme as nn.Linear) so they start different
        ref = [nn.Linear(d_in, d_out) for _ in range(num_choices)]
        self.weight = nn.Parameter(torch.stack([l.weight.data for l in ref]))  # [K, out, in]
        self.bias = nn.Parameter(torch.stack([l.bias.data for l in ref]))  # [K, out]

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in], g: [B, K] (one-hot when hard -> exactly one matrix is "taken")
        W = torch.einsum("bk,koi->boi", g, self.weight)  # [B, out, in]
        b = torch.einsum("bk,ko->bo", g, self.bias)  # [B, out]
        return torch.einsum("bti,boi->bto", x, W) + b.unsqueeze(1)


class FeedbackTransformerBlock(nn.Module):
    """Pre-norm block whose K=4 half-width FFN basis branches are selected or linearly mixed by a 4D feedback gate."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, ffn_hidden: int = 64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = SwitchedLinear(d_model, ffn_hidden, num_choices=NUM_CHOICES)
        self.fc2 = SwitchedLinear(ffn_hidden, d_model, num_choices=NUM_CHOICES)

    def forward(self, x: torch.Tensor, g: torch.Tensor, mix_branches: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x_ln = self.ln2(x)
        if mix_branches:
            # Linear combination of the K=4 FFN branch outputs with signed 4D basis weights g in [-1, 1]^4:
            # Delta_x = sum_{k=0..3} g[:, k] * FFN_k(x_ln)
            h = F.gelu(torch.einsum("bti,koi->btko", x_ln, self.fc1.weight) + self.fc1.bias)
            y = torch.einsum("btkh,koh->btko", h, self.fc2.weight) + self.fc2.bias
            x = x + torch.einsum("bk,btko->bto", g, y)
        else:
            x = x + self.fc2(F.gelu(self.fc1(x_ln, g)), g)
        return x


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class FeedbackMicroVLA(nn.Module):
    """MicroVLA where block 0's 4 FFN basis branches are controlled by a 4D recurrent state s_{t-1} in [-1, 1]^4:
    - gate_mode='mix':    signed 4D linear combination g = tanh(gate_proj(s_{t-1})) in [-1, 1]^4 of FFN_{A,B,C,D}
    - gate_mode='switch': discrete selection of weight set A, B, C, or D via Gumbel-softmax
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        state_hidden: int = 53,  # 53 makes the param count exactly match micro_vla_sequential.py (138,081)
        gate_mode: str = "mix",
    ):
        super().__init__()
        assert gate_mode in ("mix", "switch"), f"Unknown gate_mode: {gate_mode}"
        self.d_model = d_model
        self.gate_mode = gate_mode
        self.patch_embed = MicroPatchEmbed(IMG_SIZE, PATCH_SIZE, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [FeedbackTransformerBlock(d_model, n_heads, ffn_hidden=d_model)]
            + [TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # True 4D recurrent state bottleneck s_{t-1} in [-1, 1]^4: cannot memorize 64D episode embeddings
        self.h_init = nn.Parameter(torch.tensor([0.6, -0.6, 0.6, -0.6]))
        self.state_step = nn.Linear(NUM_CHOICES, NUM_CHOICES, bias=False)
        self.state_mlp = nn.Sequential(
            nn.Linear(d_model + NUM_CHOICES, state_hidden),
            nn.GELU(),
            nn.Linear(state_hidden, 2 * NUM_CHOICES),
        )
        self.gate_proj = nn.Linear(NUM_CHOICES, NUM_CHOICES)
        with torch.no_grad():
            # Initialize 4x4 state transition with constant-velocity coupling (pos += 0.5 * vel)
            nn.init.eye_(self.state_step.weight)
            self.state_step.weight[0, 2] = 0.5
            self.state_step.weight[1, 3] = 0.5
            nn.init.eye_(self.gate_proj.weight)
            nn.init.zeros_(self.gate_proj.bias)

    def initial_feedback(self, batch_size: int) -> torch.Tensor:
        return self.h_init.unsqueeze(0).expand(batch_size, -1)

    def gate(self, h_prev: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """Return (g [B,4] used for routing/mixing, p [B,4] probabilities or signed 4D weights)."""
        logits = self.gate_proj(h_prev)
        if self.gate_mode == "mix":
            g = torch.tanh(logits)  # signed 4D basis weights in [-1, 1]^4
            return g, g
        p = F.softmax(logits, dim=-1)
        if self.training:
            g = F.gumbel_softmax(logits, tau=tau, hard=hard)  # straight-through when hard
        else:
            g = F.one_hot(logits.argmax(-1), NUM_CHOICES).to(logits.dtype)  # deterministic at eval
        return g, p

    def forward(
        self,
        img: torch.Tensor,
        token_ids: torch.Tensor,
        h_prev: torch.Tensor,
        targets: torch.Tensor = None,
        tau: float = 1.0,
        hard: bool = True,
    ):
        B, seq_len = token_ids.shape
        g, p = self.gate(h_prev, tau=tau, hard=hard)

        v_emb = self.patch_embed(img)
        t_emb = self.token_embed(token_ids) + self.token_pos_emb[:, :seq_len, :]
        x = torch.cat([v_emb, t_emb], dim=1)

        x = self.blocks[0](x, g, mix_branches=(self.gate_mode == "mix"))
        for block in self.blocks[1:]:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h)

        # Read observation summary and update the bounded 4D state s_t in [-1, 1]^4
        u_t = 0.5 * (h[:, COLOR_POS, :] + h[:, ACT_START_POS, :])
        s_hat, z_alpha = self.state_mlp(torch.cat([u_t, h_prev], dim=-1)).chunk(2, dim=-1)
        alpha = torch.sigmoid(z_alpha)
        s_step = torch.tanh(self.state_step(h_prev))
        h_next = (1.0 - alpha) * s_step + alpha * torch.tanh(s_hat)

        loss = None
        if targets is not None:
            vision_ignore = torch.full(
                (B, NUM_PATCHES), -100, dtype=torch.long, device=targets.device
            )
            full_targets = torch.cat([vision_ignore, targets], dim=1)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), full_targets.view(-1), ignore_index=-100
            )

        return logits, loss, h_next, {"g": g, "p": p}


# ------------------------------------------------------------------------------
# Routing & mixing regularizers
# ------------------------------------------------------------------------------
def balance_loss(p: torch.Tensor) -> torch.Tensor:
    """log(K) - H(mean_b p_b) >= 0; zero iff the batch uses every weight set equally often."""
    mean_p = p.mean(0)
    return math.log(p.size(-1)) + (mean_p * torch.log(mean_p + 1e-9)).sum()


def decisiveness_loss(p: torch.Tensor) -> torch.Tensor:
    """Mean per-sample routing entropy; low when each sample commits to one weight set."""
    return -(p * torch.log(p + 1e-9)).sum(-1).mean()


def ortho_loss(g: torch.Tensor, target_var: float = 0.25) -> torch.Tensor:
    """Encourage signed 4D basis weights g [B, 4] to be orthogonal and have equal second moment (4 * 0.25 = 1)."""
    gram = (g.T @ g) / max(1, g.size(0))
    target = target_var * torch.eye(g.size(-1), device=g.device)
    return (gram - target).pow(2).sum()


# ------------------------------------------------------------------------------
# Training (full-episode unrolls with BPTT through the feedback connection)
# ------------------------------------------------------------------------------
def to_episodes(images, input_seqs, target_seqs, episode_len: int):
    """Reshape flat micro_vla data [E*L*3, ...] into per-stream episodes [E*3, L, ...]."""
    N = images.size(0)
    E = N // (episode_len * 3)

    def reshape(t):
        t = t.view(E, episode_len, 3, *t.shape[1:])  # [E, L, prompt, ...]
        t = t.transpose(1, 2).contiguous()  # [E, prompt, L, ...]
        return t.view(E * 3, episode_len, *t.shape[3:])  # one stream per (episode, prompt)

    return reshape(images), reshape(input_seqs), reshape(target_seqs)


def switched_params(model: FeedbackMicroVLA):
    blk = model.blocks[0]
    return [blk.fc1.weight, blk.fc1.bias, blk.fc2.weight, blk.fc2.bias]


def per_choice_grad_norm(model: FeedbackMicroVLA):
    """Gradient norm landing on each weight set k (proves both receive learning signal)."""
    sq = torch.zeros(NUM_CHOICES, device=model.h_init.device)
    for prm in switched_params(model):
        if prm.grad is not None:
            sq += prm.grad.flatten(1).pow(2).sum(1)
    return sq.sqrt().cpu().tolist()


def train_feedback_vla(
    model: FeedbackMicroVLA,
    ep_images,
    ep_inputs,
    ep_targets,
    steps: int = 520,
    streams_per_batch: int = 24,
    lr: float = 4e-3,
    warmup_frac: float = 0.3,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
    balance_coef: float = 0.1,
    decisive_coef: float = 0.01,
    ortho_coef: float = 0.05,
    detach_feedback: bool = False,
    batch_schedule=None,
    log_every: int = 65,
    device: torch.device = DEVICE,
):
    """batch_schedule: optional list of episode-index tensors (one per step) to share batches across models."""
    model.to(device)
    ep_images = ep_images.to(device)
    ep_inputs = ep_inputs.to(device)
    ep_targets = ep_targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_episodes = ep_images.size(0) // 3
    eps_per_batch = max(1, streams_per_batch // 3)
    L = ep_images.size(1)
    decay_start = int(steps * 0.71)
    warmup_steps = int(steps * warmup_frac)
    history = {"loss": [], "usage": [], "grad": []}
    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        if step > decay_start:
            frac = (step - decay_start) / max(1, steps - decay_start)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * max(0.1, 1.0 - 0.9 * frac)
        tau = tau_start * (tau_end / tau_start) ** ((step - 1) / max(1, steps - 1))
        hard = step > warmup_steps

        # Sample whole episodes, keeping all 3 counterfactual prompt streams together
        if batch_schedule is not None:
            ep_idx = batch_schedule[step - 1].to(device) * 3
        else:
            ep_idx = torch.randint(0, num_episodes, (eps_per_batch,), device=device) * 3
        idx = torch.cat([ep_idx, ep_idx + 1, ep_idx + 2])
        imgs, inps, tgts = ep_images[idx], ep_inputs[idx], ep_targets[idx]

        h = model.initial_feedback(idx.numel())
        ce_total, bal_total, dec_total, ort_total = 0.0, 0.0, 0.0, 0.0
        usage = torch.zeros(NUM_CHOICES, device=device)
        for t in range(L):
            _, ce, h, info = model(imgs[:, t], inps[:, t], h, tgts[:, t], tau=tau, hard=hard)
            if detach_feedback:
                h = h.detach()
            ce_total = ce_total + ce
            if model.gate_mode == "mix":
                if t >= 1:
                    ort_total = ort_total + ortho_loss(info["g"])
                usage += info["g"].detach().abs().mean(0)
            else:
                bal_total = bal_total + balance_loss(info["p"])
                dec_total = dec_total + decisiveness_loss(info["p"])
                usage += info["g"].detach().sum(0)
        ce_mean = ce_total / L
        if model.gate_mode == "mix":
            loss = ce_mean + ortho_coef * ort_total / max(1, L - 1)
            usage = (usage / L).cpu().tolist()
        else:
            loss = ce_mean + balance_coef * bal_total / L + decisive_coef * dec_total / L
            usage = (usage / usage.sum()).cpu().tolist()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = per_choice_grad_norm(model)
        optimizer.step()

        history["loss"].append(float(ce_mean.item()))
        history["usage"].append(usage)
        history["grad"].append(grads)
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            mode_str = "mix(|g|)" if model.gate_mode == "mix" else f"tau:{tau:.2f} {'hard' if hard else 'soft'}"
            u_str = "/".join(f"{u:.2f}" for u in usage)
            g_str = "/".join(f"{gr:.2f}" for gr in grads)
            print(
                f"step {step:4d}/{steps} | ce: {ce_mean.item():.4f} | {mode_str} "
                f"| A..D: {u_str} | grad A..D: {g_str} | {time.time() - t0:5.1f}s"
            )

    return history


# ------------------------------------------------------------------------------
# Closed-loop inference with feedback carried across control steps
# ------------------------------------------------------------------------------
@torch.no_grad()
def run_inference(
    model: FeedbackMicroVLA,
    prompt: str = "reach blue",
    steps: int = 15,
    env_state=None,
    hidden_steps=None,
    target_velocities=None,
    target_speed: float = 0.0,
):
    """Closed-loop rollout; targets are not rendered at timesteps in `hidden_steps` (occlusion)
    and move along non-colliding lines when `target_velocities` or `target_speed > 0` is given."""
    hidden_steps = hidden_steps or set()
    model.eval()
    words = prompt.strip().lower().split()
    assert len(words) == 2 and words[0] == "reach" and words[1] in COLORS, prompt
    target_color = words[1]
    gripper_pos, raw_targets = (
        sample_env_state() if env_state is None else (list(env_state[0]), env_state[1])
    )
    targets = {c: list(p) for c, p in raw_targets.items()}
    if target_velocities is None and target_speed > 0.0:
        target_velocities = sample_target_velocities(targets, speed=target_speed, max_steps=steps)
    goal_pos = targets[target_color]
    prefix = [stoi["reach"], stoi[target_color], stoi["ACT_START"]]
    frames, trajectory, records = [], [list(gripper_pos)], []
    dev = model.h_init.device
    h = model.initial_feedback(1)

    for t in range(steps):
        img = render_env(
            gripper_pos, targets, hidden_colors=hidden_colors_at(hidden_steps, t, target_color)
        )
        frames.append(img)
        img_b = img.unsqueeze(0).to(dev)

        logits_x, _, h_next, info = model(img_b, torch.tensor([prefix], device=dev), h)
        x_probs = F.softmax(logits_x[0, -1, X_OFFSET : X_OFFSET + NUM_BINS], dim=-1)
        x_bin = int(x_probs.argmax())
        act_x_tok = X_OFFSET + x_bin

        # Same h_{t-1} for the Y pass -> same weight set within one control step
        logits_y, _, _, _ = model(img_b, torch.tensor([prefix + [act_x_tok]], device=dev), h)
        y_probs = F.softmax(logits_y[0, -1, Y_OFFSET : Y_OFFSET + NUM_BINS], dim=-1)
        y_bin = int(y_probs.argmax())
        act_y_tok = Y_OFFSET + y_bin
        h = h_next

        dx, dy = bin_to_action(x_bin), bin_to_action(y_bin)
        gripper_pos = [
            max(2.0, min(29.0, gripper_pos[0] + dx * MAX_STEP_PX)),
            max(2.0, min(29.0, gripper_pos[1] + dy * MAX_STEP_PX)),
        ]
        targets = step_targets(targets, target_velocities)
        goal_pos = targets[target_color]
        dist = math.hypot(goal_pos[0] - gripper_pos[0], goal_pos[1] - gripper_pos[1])
        trajectory.append(list(gripper_pos))
        records.append(
            {
                "step": t + 1,
                "x_tok": itos[act_x_tok],
                "y_tok": itos[act_y_tok],
                "dx": dx,
                "dy": dy,
                "dist": dist,
                "x_probs": x_probs.cpu(),
                "y_probs": y_probs.cpu(),
                "choice": int(info["g"].argmax().item()),
                "p_choice": info["p"][0].cpu().tolist(),
            }
        )

    frames.append(render_env(gripper_pos, targets))
    return {
        "prompt": prompt,
        "target_color": target_color,
        "targets": targets,
        "frames": frames,
        "trajectory": trajectory,
        "records": records,
    }


if __name__ == "__main__":
    t_start = time.time()
    EPISODE_LEN = 10
    OCCLUSION_PROB = 0.7
    print(
        f"Generating synthetic expert demonstrations (250 episodes x 10 steps x 3 prompts, "
        f"occlusion_prob={OCCLUSION_PROB}, target_speed={TARGET_SPEED})..."
    )
    images, input_seqs, target_seqs = generate_vla_dataset(
        num_episodes=250,
        episode_len=EPISODE_LEN,
        occlusion_prob=OCCLUSION_PROB,
        target_speed=TARGET_SPEED,
    )
    ep_images, ep_inputs, ep_targets = to_episodes(images, input_seqs, target_seqs, EPISODE_LEN)

    model = FeedbackMicroVLA(vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4)
    init_switched = [p.detach().cpu().clone() for p in switched_params(model)]
    print(
        f"Initialized FeedbackMicroVLA | Parameters: {sum(p.numel() for p in model.parameters()):,} "
        f"| Device: {DEVICE}"
    )

    print(f"\n--- Training FeedbackMicroVLA on {DEVICE} (full-episode BPTT) ---")
    history = train_feedback_vla(model, ep_images, ep_inputs, ep_targets, steps=2080)

    # Verify both weight sets actually learned: distance moved from init, per set
    moved = torch.zeros(NUM_CHOICES)
    for p0, p in zip(init_switched, switched_params(model)):
        moved += (p.detach().cpu() - p0).flatten(1).pow(2).sum(1)
    moved = moved.sqrt().tolist()
    usage_total = torch.tensor(history["usage"]).mean(0).tolist()
    moved_str = "  ".join(f"{'ABCD'[k]}={moved[k]:.3f}" for k in range(NUM_CHOICES))
    usage_str = "/".join(f"{u:.2f}" for u in usage_total)
    print(
        f"\nWeight-set update norm (||W - W_init||): {moved_str} | "
        f"avg training usage A..D: {usage_str}"
    )
    if min(moved) < 0.1 * max(moved):
        print("WARNING: one weight set barely trained (routing collapse).")

    eval_env_state = sample_env_state()
    eval_vels = sample_target_velocities(eval_env_state[1], speed=TARGET_SPEED, max_steps=15)
    eval_hidden = {"blue": {3, 4, 5}, "red": {5, 6, 7}, "green": {2, 3, 4}}
    rollouts = [
        run_inference(
            model,
            prompt=f"reach {c}",
            steps=15,
            env_state=eval_env_state,
            hidden_steps=eval_hidden,
            target_velocities=eval_vels,
        )
        for c in ["blue", "red", "green"]
    ]
    for r in rollouts:
        print_ascii_rollout(r)
        print("  weight set per step: " + "".join("ABCD"[rec["choice"]] for rec in r["records"]))

    save_demo_gif(
        rollouts,
        history["loss"],
        save_path="micro_vla_demo_feedback.gif",
        title="micro_vla_feedback.py — VLA with h(t-1) feedback gating block-0 FFN weights (A/B via Gumbel-softmax)",
    )
    print(f"\nTotal runtime: {time.time() - t_start:.2f}s")

