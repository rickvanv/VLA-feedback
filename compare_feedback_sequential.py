"""
compare_feedback_sequential.py: Fair multi-seed comparison of the two parameter-matched variants.

For each seed, both models get:
  - the same training dataset and the same held-out validation dataset
  - exactly the same sequence of episode batches (shared batch schedule)
  - the same optimizer, LR schedule and number of updates
and are then evaluated on:
  - validation cross-entropy, split into frames where targets are VISIBLE vs OCCLUDED
    (feedback model unrolled over full episodes with deterministic gate)
  - closed-loop rollouts on the same random env states and occlusion windows for all 3 prompts

Occlusion task (--occlusion-prob > 0): in a fraction of episodes, all targets disappear for 2-4
consecutive frames (never in the first 2 frames). The expert keeps acting on the true target
position, so predicting the right action during occlusion requires memory of earlier frames.

Usage: python compare_feedback_sequential.py --occlusion-prob 0.7 --seeds 0 1 2 --steps 520
"""

import argparse
import random
import statistics
import time
import torch
import torch.nn.functional as F

from micro_vla import (
    NUM_PATCHES,
    VOCAB_SIZE,
    generate_vla_dataset,
    run_inference as run_inference_seq,
    sample_env_state,
    sample_occlusion,
)
from micro_vla_feedback import FeedbackMicroVLA, run_inference as run_inference_fb, to_episodes, train_feedback_vla
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


def closed_loop(run_fn, model, env_states, hidden_windows):
    final, success, after_occ = [], [], []
    for state, hidden in zip(env_states, hidden_windows):
        for c in ["red", "green", "blue"]:
            r = run_fn(model, prompt=f"reach {c}", steps=ROLLOUT_STEPS, env_state=state, hidden_steps=hidden)
            d = r["records"][-1]["dist"]
            final.append(d)
            success.append(float(d < SUCCESS_PX))
            occ_c = hidden[c] if isinstance(hidden, dict) else hidden
            if occ_c:  # distance to goal right after the last occluded action for target c
                after_occ.append(r["records"][max(occ_c)]["dist"])
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
    print(f"Occlusion probability per episode: {args.occlusion_prob} | steps: {args.steps} | seeds: {args.seeds}")

    for seed in args.seeds:
        print(f"\n================ seed {seed} ================")
        seed_all(seed)
        tr = generate_vla_dataset(250, EPISODE_LEN, occlusion_prob=args.occlusion_prob)
        tr = to_episodes(*tr, EPISODE_LEN)
        *va, va_hidden = generate_vla_dataset(
            50, EPISODE_LEN, occlusion_prob=args.occlusion_prob, return_hidden=True
        )
        va = to_episodes(*va, EPISODE_LEN)
        va_hidden = va_hidden.view(-1, EPISODE_LEN, 3).transpose(1, 2).reshape(-1, EPISODE_LEN)
        env_states = [sample_env_state() for _ in range(NUM_EVAL_STATES)]
        hidden_windows = [
            sample_occlusion(EPISODE_LEN, args.occlusion_prob)
            for _ in range(NUM_EVAL_STATES)
        ]
        num_episodes = tr[0].size(0) // 3
        schedule = [torch.randint(0, num_episodes, (EPS_PER_BATCH,)) for _ in range(args.steps)]

        runs = {
            "feedback": (FeedbackMicroVLA, train_feedback_vla, val_ce_feedback, run_inference_fb),
            "sequential": (SequentialMicroVLA, train_sequential_episodes, val_ce_sequential, run_inference_seq),
        }
        for name, (Model, train_fn, val_fn, infer_fn) in runs.items():
            seed_all(seed)  # same init RNG state for both models
            model = Model(vocab_size=VOCAB_SIZE)
            t0 = time.time()
            out = train_fn(model, *tr, steps=args.steps, batch_schedule=schedule, log_every=0)
            train_loss = out["loss"] if isinstance(out, dict) else out

            ce = val_fn(model, *va)
            metrics = {
                "train_ce(last 20)": statistics.mean(train_loss[-20:]),
                "val_ce_visible": float(ce[~va_hidden].mean()),
                "val_ce_occluded": float(ce[va_hidden].mean()) if va_hidden.any() else float("nan"),
            }
            metrics.update(closed_loop(infer_fn, model, env_states, hidden_windows))
            for k, v in metrics.items():
                results[name].setdefault(k, []).append(v)
            print(
                f"{name:10s} | params {sum(p.numel() for p in model.parameters()):,} | "
                + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
                + f" | {time.time() - t0:.0f}s"
            )

    print(f"\n=============== summary over seeds {args.seeds} (mean ± std) ===============")
    print(f"{'metric':20s} | {'feedback':>16s} | {'sequential':>16s}")
    for k in results["feedback"]:
        print(f"{k:20s} | {fmt(results['feedback'][k]):>16s} | {fmt(results['sequential'][k]):>16s}")
    print(f"(success = final distance < {SUCCESS_PX}px over {NUM_EVAL_STATES} states x 3 prompts)")
    print(f"Total runtime: {time.time() - t_start:.0f}s")
