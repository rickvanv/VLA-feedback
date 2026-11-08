"""
compare_feedback_sequential.py: Fair multi-seed comparison of the two parameter-matched variants.

For each seed, both models get:
  - the same training dataset and the same held-out validation dataset
  - exactly the same sequence of episode batches (shared batch schedule)
  - the same optimizer, LR schedule and number of updates
and are then evaluated on:
  - validation cross-entropy, split into frames where the queried target is VISIBLE vs OCCLUDED
    (feedback model unrolled over full episodes with deterministic gate)
  - closed-loop rollouts on the same random env states and occlusion windows for all 3 prompts

Usage: python compare_feedback_sequential.py --occlusion-prob 0.7 --seeds 0 1 2 --steps 2080
"""

import argparse
import math
import random
import statistics
import time
import torch
import torch.nn.functional as F

from micro_vla import (
    DEVICE,
    MAX_STEP_PX,
    NUM_BINS,
    NUM_PATCHES,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    bin_to_action,
    generate_vla_dataset,
    hidden_colors_at,
    render_env,
    sample_env_state,
    sample_occlusion,
    stoi,
)
from micro_vla_feedback import FeedbackMicroVLA, to_episodes, train_feedback_vla
from micro_vla_sequential import SequentialMicroVLA, train_sequential_episodes

EPISODE_LEN = 10
EPS_PER_BATCH = 8  # 8 episodes x 3 prompts = 24 streams x 10 timesteps = 240 sequences per update
NUM_EVAL_STATES = 30
ROLLOUT_STEPS = 15
SUCCESS_PX = 1.5


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def per_sample_ce(logits, targets):
    """Mean CE over the two action tokens (X at ACT_START, Y at ACT_X) for each sample -> [B]."""
    lx = F.cross_entropy(logits[:, NUM_PATCHES + 2], targets[:, 2], reduction="none")
    ly = F.cross_entropy(logits[:, NUM_PATCHES + 3], targets[:, 3], reduction="none")
    return 0.5 * (lx + ly)


@torch.no_grad()
def val_ce_feedback(model, ep_images, ep_inputs, ep_targets):
    model.eval()
    h = model.initial_feedback(ep_images.size(0))
    ces = []
    for t in range(ep_images.size(1)):
        logits, _, h, _ = model(ep_images[:, t], ep_inputs[:, t], h)
        ces.append(per_sample_ce(logits, ep_targets[:, t]))
    return torch.stack(ces, dim=1)  # [streams, L]


@torch.no_grad()
def val_ce_sequential(model, ep_images, ep_inputs, ep_targets):
    model.eval()
    S, L = ep_images.shape[:2]
    logits, _ = model(ep_images.flatten(0, 1), ep_inputs.flatten(0, 1))
    return per_sample_ce(logits, ep_targets.flatten(0, 1)).view(S, L)


@torch.no_grad()
def closed_loop(model, env_states, hidden_windows, is_feedback: bool):
    """Run all (NUM_EVAL_STATES x 3 prompts) closed-loop rollouts in a single GPU batch."""
    model.eval()
    dev = next(model.parameters()).device
    colors = ["red", "green", "blue"]
    M = len(env_states) * len(colors)
    grippers = [list(s[0]) for s in env_states for _ in colors]
    targets_list = [s[1] for s in env_states for _ in colors]
    hiddens = [h for h in hidden_windows for _ in colors]
    target_colors = [c for _ in env_states for c in colors]
    goals = [t[c] for t, c in zip(targets_list, target_colors)]
    prefixes = torch.tensor(
        [[stoi["reach"], stoi[c], stoi["ACT_START"]] for c in target_colors],
        dtype=torch.long,
        device=dev,
    )
    h = model.initial_feedback(M) if is_feedback else None
    dists_per_step = []

    for t in range(ROLLOUT_STEPS):
        imgs = torch.stack(
            [
                render_env(g, tgts, hidden_colors=hidden_colors_at(hid, t, tc))
                for g, tgts, hid, tc in zip(grippers, targets_list, hiddens, target_colors)
            ]
        ).to(dev)
        if is_feedback:
            logits_x, _, h_next, _ = model(imgs, prefixes, h)
        else:
            logits_x, _ = model(imgs, prefixes)
        x_bins = logits_x[:, -1, X_OFFSET : X_OFFSET + NUM_BINS].argmax(dim=-1)
        act_x_toks = (X_OFFSET + x_bins).unsqueeze(1)
        inp_y = torch.cat([prefixes, act_x_toks], dim=1)
        if is_feedback:
            logits_y, _, _, _ = model(imgs, inp_y, h)
            h = h_next
        else:
            logits_y, _ = model(imgs, inp_y)
        y_bins = logits_y[:, -1, Y_OFFSET : Y_OFFSET + NUM_BINS].argmax(dim=-1)

        step_dists = []
        for i, (xb, yb) in enumerate(zip(x_bins.cpu().tolist(), y_bins.cpu().tolist())):
            dx, dy = bin_to_action(xb), bin_to_action(yb)
            grippers[i] = [
                max(2.0, min(29.0, grippers[i][0] + dx * MAX_STEP_PX)),
                max(2.0, min(29.0, grippers[i][1] + dy * MAX_STEP_PX)),
            ]
            step_dists.append(math.hypot(goals[i][0] - grippers[i][0], goals[i][1] - grippers[i][1]))
        dists_per_step.append(step_dists)

    final = dists_per_step[-1]
    success = [float(d < SUCCESS_PX) for d in final]
    after_occ = [
        dists_per_step[max(hid[tc])][i]
        for i, (hid, tc) in enumerate(zip(hiddens, target_colors))
        if hid[tc]
    ]
    return {
        "final_dist_px": statistics.mean(final),
        "success_rate": statistics.mean(success),
        "dist_after_occl_px": statistics.mean(after_occ) if after_occ else float("nan"),
    }


def fmt(xs):
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--occlusion-prob", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=520)
    args = ap.parse_args()

    t_start = time.time()
    results = {"feedback": {}, "sequential": {}}
    print(
        f"Device: {DEVICE} | Occlusion prob: {args.occlusion_prob} | "
        f"steps: {args.steps} | seeds: {args.seeds}"
    )

    for seed in args.seeds:
        print(f"\n================ seed {seed} ================")
        seed_all(seed)
        tr = generate_vla_dataset(250, EPISODE_LEN, occlusion_prob=args.occlusion_prob)
        tr = tuple(x.to(DEVICE) for x in to_episodes(*tr, EPISODE_LEN))
        *va, va_hidden = generate_vla_dataset(
            50, EPISODE_LEN, occlusion_prob=args.occlusion_prob, return_hidden=True
        )
        va = tuple(x.to(DEVICE) for x in to_episodes(*va, EPISODE_LEN))
        va_hidden = (
            va_hidden.view(-1, EPISODE_LEN, 3).transpose(1, 2).reshape(-1, EPISODE_LEN).to(DEVICE)
        )
        env_states = [sample_env_state() for _ in range(NUM_EVAL_STATES)]
        hidden_windows = [
            sample_occlusion(EPISODE_LEN, args.occlusion_prob)
            for _ in range(NUM_EVAL_STATES)
        ]
        num_episodes = tr[0].size(0) // 3
        schedule = [
            torch.randint(0, num_episodes, (EPS_PER_BATCH,), device=DEVICE)
            for _ in range(args.steps)
        ]

        runs = {
            "feedback": (FeedbackMicroVLA, train_feedback_vla, val_ce_feedback, True),
            "sequential": (SequentialMicroVLA, train_sequential_episodes, val_ce_sequential, False),
        }
        for name, (Model, train_fn, val_fn, is_fb) in runs.items():
            seed_all(seed)  # same init RNG state for both models
            model = Model(vocab_size=VOCAB_SIZE).to(DEVICE)
            t0 = time.time()
            out = train_fn(model, *tr, steps=args.steps, batch_schedule=schedule, log_every=0)
            train_loss = out["loss"] if isinstance(out, dict) else out

            ce = val_fn(model, *va)
            metrics = {
                "train_ce(last 20)": statistics.mean(train_loss[-20:]),
                "val_ce_visible": float(ce[~va_hidden].mean().item()),
                "val_ce_occluded": float(ce[va_hidden].mean().item()) if va_hidden.any() else float("nan"),
            }
            metrics.update(closed_loop(model, env_states, hidden_windows, is_feedback=is_fb))
            for k, v in metrics.items():
                results[name].setdefault(k, []).append(v)
            print(
                f"{name:10s} | params {sum(p.numel() for p in model.parameters()):,} | "
                + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
                + f" | {time.time() - t0:.1f}s"
            )

    print(f"\n=============== summary over seeds {args.seeds} (mean ± std) ===============")
    print(f"{'metric':20s} | {'feedback':>16s} | {'sequential':>16s}")
    for k in results["feedback"]:
        print(f"{k:20s} | {fmt(results['feedback'][k]):>16s} | {fmt(results['sequential'][k]):>16s}")
    print(f"(success = final distance < {SUCCESS_PX}px over {NUM_EVAL_STATES} states x 3 prompts)")
    print(f"Total runtime: {time.time() - t_start:.1f}s")
