"""
micro_vla_feedback.py: MicroVLA with a recurrent feedback connection that switches weights.

Built on top of micro_vla.py (env, tokenizer, data and visualization are reused from there).

What is new
-----------
The FFN of the *first* Transformer block owns TWO complete sets of weight matrices
(W1_a, W2_a) and (W1_b, W2_b). Which set is used at control step t is decided by a small
gating MLP that reads the model's *final hidden state from the previous step t-1*:

    h_{t-1}  --gate MLP-->  logits (2)  --Gumbel-softmax-->  g_t (one-hot, straight-through)
    W_t = g_t[0] * W_a + g_t[1] * W_b              (per sample, inside block 0's FFN)

"Final hidden state" = output of the last block (after ln_f) at the ACT_START position.
Because attention is causal, that vector is identical for the X-pass and the Y-pass of a
step, so the feedback is well defined regardless of how many action tokens were decoded.

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
    IMG_SIZE,
    MAX_STEP_PX,
    NUM_BINS,
    NUM_PATCHES,
    PATCH_SIZE,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    CausalSelfAttention,
    MicroPatchEmbed,
    TransformerBlock,
    bin_to_action,
    generate_vla_dataset,
    itos,
    print_ascii_rollout,
    render_env,
    sample_env_state,
    save_demo_gif,
    stoi,
)

NUM_CHOICES = 2  # number of alternative weight sets in the switched block
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
    """Pre-norm block whose FFN weights are selected per sample by a feedback gate."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = SwitchedLinear(d_model, 2 * d_model)
        self.fc2 = SwitchedLinear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x), g)), g)
        return x


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class FeedbackMicroVLA(nn.Module):
    """MicroVLA where block 0's FFN weight set is chosen from h_{t-1} via MLP + Gumbel-softmax."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        gate_hidden: int = 51,  # 51 makes the param count exactly match micro_vla_sequential.py
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_embed = MicroPatchEmbed(IMG_SIZE, PATCH_SIZE, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [FeedbackTransformerBlock(d_model, n_heads)]
            + [TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Feedback path: h_{t-1} -> gate logits. h_init is the learned "previous state" at t=0.
        self.h_init = nn.Parameter(torch.zeros(d_model))
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, NUM_CHOICES),
        )

    def initial_feedback(self, batch_size: int) -> torch.Tensor:
        return self.h_init.unsqueeze(0).expand(batch_size, -1)

    def gate(self, h_prev: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """Return (g [B,K] used for routing, p [B,K] noise-free routing probabilities)."""
        logits = self.gate_mlp(h_prev)
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

        x = self.blocks[0](x, g)
        for block in self.blocks[1:]:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h)
        h_next = h[:, ACT_START_POS, :]  # final hidden state fed back at step t+1

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
# Routing regularizers
# ------------------------------------------------------------------------------
def balance_loss(p: torch.Tensor) -> torch.Tensor:
    """log(K) - H(mean_b p_b) >= 0; zero iff the batch uses every weight set equally often."""
    mean_p = p.mean(0)
    return math.log(p.size(-1)) + (mean_p * torch.log(mean_p + 1e-9)).sum()


def decisiveness_loss(p: torch.Tensor) -> torch.Tensor:
    """Mean per-sample routing entropy; low when each sample commits to one weight set."""
    return -(p * torch.log(p + 1e-9)).sum(-1).mean()


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
    sq = torch.zeros(NUM_CHOICES)
    for prm in switched_params(model):
        if prm.grad is not None:
            sq += prm.grad.flatten(1).pow(2).sum(1)
    return sq.sqrt().tolist()


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
    detach_feedback: bool = False,
    batch_schedule=None,
    log_every: int = 65,
):
    """batch_schedule: optional list of episode-index tensors (one per step) to share batches across models."""
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
            ep_idx = batch_schedule[step - 1] * 3
        else:
            ep_idx = torch.randint(0, num_episodes, (eps_per_batch,)) * 3
        idx = torch.cat([ep_idx, ep_idx + 1, ep_idx + 2])
        imgs, inps, tgts = ep_images[idx], ep_inputs[idx], ep_targets[idx]

        h = model.initial_feedback(idx.numel())
        ce_total, bal_total, dec_total, usage = 0.0, 0.0, 0.0, torch.zeros(NUM_CHOICES)
        for t in range(L):
            _, ce, h, info = model(imgs[:, t], inps[:, t], h, tgts[:, t], tau=tau, hard=hard)
            if detach_feedback:
                h = h.detach()
            ce_total = ce_total + ce
            bal_total = bal_total + balance_loss(info["p"])
            dec_total = dec_total + decisiveness_loss(info["p"])
            usage += info["g"].detach().sum(0)
        ce_mean = ce_total / L
        loss = ce_mean + balance_coef * bal_total / L + decisive_coef * dec_total / L

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = per_choice_grad_norm(model)
        optimizer.step()

        usage = (usage / usage.sum()).tolist()
        history["loss"].append(float(ce_mean.item()))
        history["usage"].append(usage)
        history["grad"].append(grads)
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(
                f"step {step:4d}/{steps} | ce: {ce_mean.item():.4f} | tau: {tau:.2f} "
                f"{'hard' if hard else 'soft'} | usage A/B: {usage[0]:.2f}/{usage[1]:.2f} "
                f"| grad A/B: {grads[0]:.3f}/{grads[1]:.3f} | {time.time() - t0:5.1f}s"
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
):
    """Closed-loop rollout; targets are not rendered at timesteps in `hidden_steps` (occlusion)."""
    hidden_steps = hidden_steps or set()
    model.eval()
    words = prompt.strip().lower().split()
    assert len(words) == 2 and words[0] == "reach" and words[1] in COLORS, prompt
    target_color = words[1]
    gripper_pos, targets = (
        sample_env_state() if env_state is None else (list(env_state[0]), env_state[1])
    )
    goal_pos = targets[target_color]
    prefix = [stoi["reach"], stoi[target_color], stoi["ACT_START"]]
    frames, trajectory, records = [], [list(gripper_pos)], []
    h = model.initial_feedback(1)

    for t in range(steps):
        img = render_env(gripper_pos, targets, show_targets=t not in hidden_steps)
        frames.append(img)
        img_b = img.unsqueeze(0)

        logits_x, _, h_next, info = model(img_b, torch.tensor([prefix]), h)
        x_probs = F.softmax(logits_x[0, -1, X_OFFSET : X_OFFSET + NUM_BINS], dim=-1)
        x_bin = int(x_probs.argmax())
        act_x_tok = X_OFFSET + x_bin

        # Same h_{t-1} for the Y pass -> same weight set within one control step
        logits_y, _, _, _ = model(img_b, torch.tensor([prefix + [act_x_tok]]), h)
        y_probs = F.softmax(logits_y[0, -1, Y_OFFSET : Y_OFFSET + NUM_BINS], dim=-1)
        y_bin = int(y_probs.argmax())
        act_y_tok = Y_OFFSET + y_bin
        h = h_next

        dx, dy = bin_to_action(x_bin), bin_to_action(y_bin)
        gripper_pos = [
            max(2.0, min(29.0, gripper_pos[0] + dx * MAX_STEP_PX)),
            max(2.0, min(29.0, gripper_pos[1] + dy * MAX_STEP_PX)),
        ]
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
                "x_probs": x_probs,
                "y_probs": y_probs,
                "choice": int(info["g"].argmax()),
                "p_choice": info["p"][0].tolist(),
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
    print("Generating synthetic expert demonstrations (250 episodes x 10 steps x 3 prompts)...")
    images, input_seqs, target_seqs = generate_vla_dataset(num_episodes=250, episode_len=EPISODE_LEN)
    ep_images, ep_inputs, ep_targets = to_episodes(images, input_seqs, target_seqs, EPISODE_LEN)

    model = FeedbackMicroVLA(vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4)
    init_switched = [p.detach().clone() for p in switched_params(model)]
    print(f"Initialized FeedbackMicroVLA | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- Training FeedbackMicroVLA on CPU (full-episode BPTT) ---")
    history = train_feedback_vla(model, ep_images, ep_inputs, ep_targets, steps=520)

    # Verify both weight sets actually learned: distance moved from init, per set
    moved = torch.zeros(NUM_CHOICES)
    for p0, p in zip(init_switched, switched_params(model)):
        moved += (p.detach() - p0).flatten(1).pow(2).sum(1)
    moved = moved.sqrt().tolist()
    usage_total = torch.tensor(history["usage"]).mean(0).tolist()
    print(
        f"\nWeight-set update norm (||W - W_init||): A={moved[0]:.3f}  B={moved[1]:.3f} | "
        f"avg training usage A/B: {usage_total[0]:.2f}/{usage_total[1]:.2f}"
    )
    if min(moved) < 0.1 * max(moved):
        print("WARNING: one weight set barely trained (routing collapse).")

    eval_env_state = sample_env_state()
    rollouts = [
        run_inference(model, prompt=f"reach {c}", steps=15, env_state=eval_env_state)
        for c in ["blue", "red", "green"]
    ]
    for r in rollouts:
        print_ascii_rollout(r)
        print("  weight set per step: " + "".join("AB"[rec["choice"]] for rec in r["records"]))

    save_demo_gif(
        rollouts,
        history["loss"],
        save_path="micro_vla_demo_feedback.gif",
        title="micro_vla_feedback.py — VLA with h(t-1) feedback gating block-0 FFN weights (A/B via Gumbel-softmax)",
    )
    print(f"\nTotal runtime: {time.time() - t_start:.2f}s")

