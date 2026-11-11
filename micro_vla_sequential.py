"""
micro_vla_sequential.py: Parameter-matched control for micro_vla_feedback.py.

The first Transformer block owns two FFN weight sets (A and B) like the feedback model, but
instead of a gate selecting one of them per sample, they are applied one after another:

    x = x + Attn(LN1(x))
    x = x + FFN_A(LN2(x))
    x = x + FFN_B(LN3(x))

So every sample always uses both weight sets. There is no feedback connection and no gate.

Exact parameter match with micro_vla_feedback.py (both 138,081 params):
  feedback extras over baseline:   4x64 FFN sets (+16,704) + state_mlp 68->53->8 (4,089)
                                   + state_step 4x4 (16) + gate_proj 4x4+4 (20) + s_init (4) = 20,833
  sequential extras over baseline: FFN_B with hidden 161 instead of 128 (20,833) + LN3 w/o affine (0)

Training uses the same episode batches as the feedback model (see train_sequential_episodes):
every update sees exactly the same 24 streams x 10 timesteps, just without recurrence.
"""

import time
import torch
import torch.nn as nn

from micro_vla import (
    DEVICE,
    TARGET_SPEED,
    VOCAB_SIZE,
    NUM_PATCHES,
    CausalSelfAttention,
    MicroVLA,
    generate_vla_dataset,
    print_ascii_rollout,
    run_inference,
    sample_env_state,
    sample_target_velocities,
    save_demo_gif,
)
from micro_vla_feedback import to_episodes

FFN_B_EXTRA_HIDDEN = 33  # widens FFN_B so the param count equals micro_vla_feedback.py (138,081) exactly


def make_ffn(d_model: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, d_model),
    )


class SequentialFFNBlock(nn.Module):
    """Pre-norm block with attention followed by two FFNs (weight sets A then B) in series."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_a = make_ffn(d_model, 2 * d_model)
        self.ln3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp_b = make_ffn(d_model, 2 * d_model + FFN_B_EXTRA_HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp_a(self.ln2(x))
        x = x + self.mlp_b(self.ln3(x))
        return x


class SequentialMicroVLA(MicroVLA):
    """MicroVLA whose first block runs FFN weight sets A and B sequentially."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 64, n_layers: int = 3, n_heads: int = 4):
        super().__init__(vocab_size, d_model, n_layers, n_heads)
        self.blocks[0] = SequentialFFNBlock(d_model, n_heads)


def train_sequential_episodes(
    model: MicroVLA,
    ep_images,
    ep_inputs,
    ep_targets,
    steps: int = 520,
    streams_per_batch: int = 24,
    lr: float = 4e-3,
    batch_schedule=None,
    log_every: int = 130,
    device: torch.device = DEVICE,
):
    """Same optimizer, LR schedule and episode batches as train_feedback_vla, but no recurrence:
    the [streams, L] timesteps are flattened into one batch of independent samples."""
    model.to(device)
    ep_images = ep_images.to(device)
    ep_inputs = ep_inputs.to(device)
    ep_targets = ep_targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_episodes = ep_images.size(0) // 3
    eps_per_batch = max(1, streams_per_batch // 3)
    decay_start = int(steps * 0.71)
    loss_history = []
    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        if step > decay_start:
            frac = (step - decay_start) / max(1, steps - decay_start)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * max(0.1, 1.0 - 0.9 * frac)

        if batch_schedule is not None:
            ep_idx = batch_schedule[step - 1].to(device) * 3
        else:
            ep_idx = torch.randint(0, num_episodes, (eps_per_batch,), device=device) * 3
        idx = torch.cat([ep_idx, ep_idx + 1, ep_idx + 2])
        imgs = ep_images[idx].flatten(0, 1)
        inps = ep_inputs[idx].flatten(0, 1)
        tgts = ep_targets[idx].flatten(0, 1)
        _, loss = model(imgs, inps, tgts)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(f"step {step:4d}/{steps} | ce: {loss.item():.4f} | {time.time() - t0:5.1f}s")

    return loss_history


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

    model = SequentialMicroVLA(vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4)
    print(
        f"Initialized SequentialMicroVLA | Parameters: {sum(p.numel() for p in model.parameters()):,} "
        f"| Sequence Length: {NUM_PATCHES + 4} tokens | Device: {DEVICE}"
    )

    print(f"\n--- Training SequentialMicroVLA on {DEVICE} (same episode batches as feedback model) ---")
    loss_history = train_sequential_episodes(model, ep_images, ep_inputs, ep_targets, steps=2080)

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

    save_demo_gif(
        rollouts,
        loss_history,
        save_path="micro_vla_demo_sequential.gif",
        title="micro_vla_sequential.py — parameter-matched control: block-0 FFN weights A then B (no gate)",
    )
    print(f"Total runtime: {time.time() - t_start:.2f}s")
