"""
compare_feedback_sequential.py: Fair multi-seed comparison of parameter-matched VLA variants:
  - feedback_mix:    signed 2D linear combination g = tanh(MLP(h_{t-1})) in [-1, 1]^2 of FFN_A and FFN_B
  - feedback_switch: discrete selection of FFN_A or FFN_B via Gumbel-softmax from h_{t-1}
  - sequential:      feedforward FFN_A followed by FFN_B (no recurrence, no gate)

All three models have the exact same parameter count (137,307 parameters) and receive:
  - the same training dataset and held-out validation dataset
  - the exact same sequence of episode batches (shared batch schedule)
  - the same optimizer, LR schedule and number of updates
and are evaluated on validation cross-entropy and closed-loop rollouts, plus an animated GIF
(`micro_vla_demo_moving_overlay.gif`) overlaying the trajectories of all variants on the same
moving-target episodes.

Usage: uv run python compare_feedback_sequential.py --occlusion-prob 0.0 --target-speed 0.7 --seeds 0 1 2 --steps 2000
"""

import argparse
import math
import random
import statistics
import time
import torch
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
    bin_to_action,
    generate_vla_dataset,
    hidden_colors_at,
    render_env,
     run_inference as run_inference_seq,
    sample_env_state,
    sample_occlusion,
    sample_target_velocities,
    step_targets,
    stoi,
)
from micro_vla_feedback import (
    FeedbackMicroVLA,
    run_inference as run_inference_fb,
    to_episodes,
    train_feedback_vla,
)
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
def closed_loop(model, env_states, hidden_windows, is_feedback: bool, target_vels_list=None):
    """Run all (NUM_EVAL_STATES x 3 prompts) closed-loop rollouts in a single GPU batch."""
    model.eval()
    dev = next(model.parameters()).device
    colors = ["red", "green", "blue"]
    M = len(env_states) * len(colors)
    grippers = [list(s[0]) for s in env_states for _ in colors]
    targets_list = [{c: list(p) for c, p in s[1].items()} for s in env_states for _ in colors]
    vels_list = (
        [v for v in target_vels_list for _ in colors]
        if target_vels_list is not None
        else [None] * M
    )
    hiddens = [h for h in hidden_windows for _ in colors]
    target_colors = [c for _ in env_states for c in colors]
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
            targets_list[i] = step_targets(targets_list[i], vels_list[i])
            goal = targets_list[i][target_colors[i]]
            step_dists.append(math.hypot(goal[0] - grippers[i][0], goal[1] - grippers[i][1]))
        dists_per_step.append(step_dists)

    final = dists_per_step[-1]
    mean_traj = [statistics.mean(col) for col in zip(*dists_per_step)]
    success = [float(d < SUCCESS_PX) for d in final]
    after_occ = [
        dists_per_step[max(hid[tc])][i]
        for i, (hid, tc) in enumerate(zip(hiddens, target_colors))
        if hid[tc]
    ]
    metrics = {
        "final_dist_px": statistics.mean(final),
        "mean_dist_px": statistics.mean(mean_traj),
        "success_rate": statistics.mean(success),
    }
    if after_occ:
        metrics["dist_after_occl_px"] = statistics.mean(after_occ)
    return metrics


def render_targets_only(targets, hidden_colors=None) -> torch.Tensor:
    """Render a 32x32x3 RGB float tensor showing only the 3x3 colored target dots (no white gripper),
    so multiple model trajectories can be cleanly overlaid on the same moving-target canvas."""
    hidden_colors = hidden_colors or ()
    img = torch.zeros(IMG_SIZE, IMG_SIZE, 3, dtype=torch.float32)
    for color_name, t_pos in targets.items():
        if color_name not in hidden_colors:
            cx, cy = int(round(float(t_pos[0]))), int(round(float(t_pos[1])))
            x0, x1 = max(0, cx - 1), min(IMG_SIZE, cx + 2)
            y0, y1 = max(0, cy - 1), min(IMG_SIZE, cy + 2)
            img[y0:y1, x0:x1] = COLORS[color_name]
    return img


def save_overlay_gif(
    models: dict,
    loss_histories: dict,
    env_state,
    target_vels,
    hidden_steps=None,
    steps: int = ROLLOUT_STEPS,
    save_path: str = "micro_vla_demo_moving_overlay.gif",
):
    """Save an animated GIF overlaying the closed-loop trajectories of all model variants
    on the exact same moving-target episodes ('reach blue', 'reach red', 'reach green')."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    prompts = ["reach blue", "reach red", "reach green"]
    prompt_colors = ["blue", "red", "green"]

    # Collect rollouts for each variant across the 3 prompts on the exact same env_state & target_vels
    rollouts_by_model = {}
    for name, (model, is_fb) in models.items():
        run_fn = run_inference_fb if is_fb else run_inference_seq
        rollouts_by_model[name] = [
            run_fn(
                model,
                prompt=p,
                steps=steps,
                env_state=env_state,
                hidden_steps=hidden_steps,
                target_velocities=target_vels,
            )
            for p in prompts
        ]

    # Precompute target-only background frames and target trajectories for the shared environment
    bg_frames_per_prompt = []
    target_paths = {c: [] for c in COLORS}
    cur_targets = {c: list(pos) for c, pos in env_state[1].items()}
    for t in range(steps + 1):
        for c in COLORS:
            target_paths[c].append(list(cur_targets[c]))
        bg_frames_per_prompt.append(
            [
                render_targets_only(
                    cur_targets,
                    hidden_colors=hidden_colors_at(hidden_steps, t, tc) if t < steps else set(),
                )
                for tc in prompt_colors
            ]
        )
        if t < steps:
            cur_targets = step_targets(cur_targets, target_vels)

    variant_styles = {
        "feedback_mix": {"color": "#3fb950", "ls": "-", "marker": "o", "label": "Feedback (Mix)"},
        "feedback_switch": {"color": "#ffa657", "ls": "--", "marker": "^", "label": "Feedback (Switch)"},
        "sequential": {"color": "#d2a8ff", "ls": "-.", "marker": "s", "label": "Sequential"},
    }
    title_colors = {"blue": "#58a6ff", "red": "#ff7b72", "green": "#3fb950"}

    fig = plt.figure(figsize=(13.2, 6.8), facecolor="#0d1117")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.38, 1.0], hspace=0.44, wspace=0.28)

    env_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_gate = fig.add_subplot(gs[1, 1])
    ax_loss = fig.add_subplot(gs[1, 2])

    for ax in env_axes + [ax_dist, ax_gate, ax_loss]:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

    # Bottom-Right: Static Training Cross-Entropy Loss Curves
    for name, hist in loss_histories.items():
        st = variant_styles[name]
        ax_loss.plot(
            range(1, len(hist) + 1),
            hist,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=1.4,
            label=st["label"],
            alpha=0.9,
        )
    ax_loss.set_title("Training Action Cross-Entropy", color="#e6edf3", fontsize=9.5, pad=6)
    ax_loss.set_xlabel("Training Step", color="#8b949e", fontsize=8)
    ax_loss.set_ylabel("CE Loss", color="#8b949e", fontsize=8)
    ax_loss.grid(True, color="#21262d", linestyle="--", linewidth=0.6)
    ax_loss.legend(loc="upper right", fontsize=7, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")

    # Bottom-Left: Animated Mean Distance to Moving Goal vs. Step
    ax_dist.set_xlim(0, steps)
    max_d = max(
        rec["dist"]
        for r_list in rollouts_by_model.values()
        for r in r_list
        for rec in r["records"]
    )
    init_dists = [
        math.hypot(
            env_state[1][tc][0] - env_state[0][0],
            env_state[1][tc][1] - env_state[0][1],
        )
        for tc in prompt_colors
    ]
    mean_init_dist = statistics.mean(init_dists)
    ax_dist.set_ylim(0.0, max(max_d, mean_init_dist) * 1.12)
    ax_dist.set_title("Mean Distance to Moving Target (3 Prompts)", color="#e6edf3", fontsize=9.5, pad=6)
    ax_dist.set_xlabel("Control Step t", color="#8b949e", fontsize=8)
    ax_dist.set_ylabel("Distance (px)", color="#8b949e", fontsize=8)
    ax_dist.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    dist_curves = {}
    dist_series = {}
    for name, r_list in rollouts_by_model.items():
        st = variant_styles[name]
        series = [mean_init_dist] + [
            statistics.mean(r_list[p_idx]["records"][t]["dist"] for p_idx in range(3))
            for t in range(steps)
        ]
        dist_series[name] = series
        (line,) = ax_dist.plot(
            [], [], color=st["color"], linestyle=st["ls"], marker=st["marker"], markersize=3.5, linewidth=1.6, label=st["label"]
        )
        dist_curves[name] = line
    ax_dist.legend(loc="upper right", fontsize=7, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")

    # Bottom-Middle: 2D Signed Gate Space (g_0, g_1) in [-1, 1]^2 for feedback_mix
    ax_gate.set_xlim(-1.1, 1.1)
    ax_gate.set_ylim(-1.1, 1.1)
    ax_gate.axhline(0.0, color="#30363d", linewidth=0.8, linestyle="--")
    ax_gate.axvline(0.0, color="#30363d", linewidth=0.8, linestyle="--")
    ax_gate.set_title("Feedback (Mix) 2D Gate Weights (g_A, g_B)", color="#e6edf3", fontsize=9.5, pad=6)
    ax_gate.set_xlabel("g_A (FFN_A weight)", color="#8b949e", fontsize=8)
    ax_gate.set_ylabel("g_B (FFN_B weight)", color="#8b949e", fontsize=8)
    ax_gate.grid(True, color="#21262d", linestyle=":", linewidth=0.5)

    gate_lines, gate_dots = [], []
    for p_idx, tc in enumerate(prompt_colors):
        (gl,) = ax_gate.plot([], [], color=title_colors[tc], linewidth=1.2, alpha=0.8, label=f"reach {tc}")
        (gd,) = ax_gate.plot([], [], marker="o", markersize=5, color=title_colors[tc], markeredgecolor="#ffffff")
        gate_lines.append(gl)
        gate_dots.append(gd)
    ax_gate.legend(loc="lower left", fontsize=7, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")

    # Top Row: 3 Prompts with Overlaid Variant Trajectories
    im_artists, step_texts, target_dot_outlines = [], [], []
    traj_lines = {name: [] for name in models}
    head_dots = {name: [] for name in models}

    for i, tc in enumerate(prompt_colors):
        ax = env_axes[i]
        for p in range(0, IMG_SIZE, PATCH_SIZE):
            ax.axhline(p - 0.5, color="#30363d", linewidth=0.4, alpha=0.45)
            ax.axvline(p - 0.5, color="#30363d", linewidth=0.4, alpha=0.45)

        im = ax.imshow(bg_frames_per_prompt[0][i].numpy(), origin="upper", vmin=0.0, vmax=1.0)
        im_artists.append(im)

        # Draw target linear paths so the viewer can see the straight line each target moves along
        for c_name, t_path in target_paths.items():
            txs = [p[0] for p in t_path]
            tys = [p[1] for p in t_path]
            ax.plot(
                txs,
                tys,
                color=title_colors[c_name],
                linestyle=":",
                linewidth=1.0,
                alpha=0.9 if c_name == tc else 0.35,
            )

        # Mark shared initial gripper start position
        ax.plot(
            [env_state[0][0]],
            [env_state[0][1]],
            marker="x",
            markersize=6,
            color="#ffffff",
            markeredgewidth=1.5,
        )

        for name in models:
            st = variant_styles[name]
            (line,) = ax.plot(
                [],
                [],
                color=st["color"],
                linestyle=st["ls"],
                linewidth=1.6,
                alpha=0.92,
                label=st["label"] if i == 0 else None,
            )
            (dot,) = ax.plot(
                [],
                [],
                marker=st["marker"],
                markersize=5.5,
                color=st["color"],
                markeredgecolor="#0d1117",
                markeredgewidth=0.8,
            )
            traj_lines[name].append(line)
            head_dots[name].append(dot)

        txt = ax.text(
            0.8,
            30.2,
            "",
            color="#e6edf3",
            fontsize=6.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117", alpha=0.82, edgecolor="none"),
        )
        step_texts.append(txt)
        ax.set_title(
            f"Prompt: \"reach {tc}\" (moving target)",
            color=title_colors[tc],
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        if i == 0:
            ax.legend(loc="upper right", fontsize=6.5, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")

    has_occl = bool(hidden_steps and any(len(v) > 0 for v in hidden_steps.values()))
    occl_label = "With Occlusion" if has_occl else "No Occlusion"
    fig.suptitle(
        f"Moving Targets ({occl_label}, {len( next(iter(loss_histories.values())) )} Steps) — "
        "Overlaid Trajectories: Feedback (Mix) vs Feedback (Switch) vs Sequential",
        color="#f0f6fc",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
    )

    num_frames = steps + 1

    def update(frame_idx):
        for i, tc in enumerate(prompt_colors):
            im_artists[i].set_data(bg_frames_per_prompt[frame_idx][i].numpy())
            rec_idx = min(max(0, frame_idx - 1), steps - 1)
            parts = [f"t={frame_idx:2d}"]
            for name in models:
                r = rollouts_by_model[name][i]
                traj_slice = r["trajectory"][: frame_idx + 1]
                xs = [p[0] for p in traj_slice]
                ys = [p[1] for p in traj_slice]
                traj_lines[name][i].set_data(xs, ys)
                head_dots[name][i].set_data([xs[-1]], [ys[-1]])
                d = (
                    init_dists[i]
                    if frame_idx == 0
                    else r["records"][rec_idx]["dist"]
                )
                short = {"feedback_mix": "mix", "feedback_switch": "sw", "sequential": "seq"}[name]
                parts.append(f"{short}:{d:4.1f}px")
            step_texts[i].set_text(" | ".join(parts[:2]) + "\n" + " | ".join(parts[2:]))

        # Update distance curves
        xs_t = list(range(frame_idx + 1))
        for name, line in dist_curves.items():
            line.set_data(xs_t, dist_series[name][: frame_idx + 1])

        # Update 2D gate plot for feedback_mix
        if "feedback_mix" in rollouts_by_model:
            n_g = min( max(1, frame_idx), steps)
            for p_idx in range(3):
                recs = rollouts_by_model["feedback_mix"][p_idx]["records"][:n_g]
                gxs = [rec["p_choice"][0] for rec in recs]
                gys = [rec["p_choice"][1] for rec in recs]
                gate_lines[p_idx].set_data(gxs, gys)
                gate_dots[p_idx].set_data([gxs[-1]], [gys[-1]])

        return im_artists + step_texts

    anim = FuncAnimation(fig, update, frames=num_frames, interval=240, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"\nSaved overlaid comparison GIF to: {save_path}")


def fmt(xs):
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--occlusion-prob", type=float, default=0.0)
    ap.add_argument(
        "--target-speed",
        type=float,
        default=TARGET_SPEED,
        help=f"Linear target speed in px/step (default: {TARGET_SPEED})",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--gif-path", type=str, default="micro_vla_demo_moving_overlay.gif")
    args = ap.parse_args()

    t_start = time.time()
    variant_names = ["feedback_mix", "feedback_switch", "sequential"]
    results = {k: {} for k in variant_names}
    print(
        f"Device: {DEVICE} | Occlusion prob: {args.occlusion_prob} | "
        f"Target speed: {args.target_speed} px/step | steps: {args.steps} | seeds: {args.seeds}"
    )

    last_models, last_losses, last_state, last_vels, last_hidden = None, None, None, None, None

    for seed in args.seeds:
        print(f"\n================ seed {seed} ================")
        seed_all(seed)
        tr = generate_vla_dataset(
            250, EPISODE_LEN, occlusion_prob=args.occlusion_prob, target_speed=args.target_speed
        )
        tr = tuple(x.to(DEVICE) for x in to_episodes(*tr, EPISODE_LEN))
        *va, va_hidden = generate_vla_dataset(
            50,
            EPISODE_LEN,
            occlusion_prob=args.occlusion_prob,
            target_speed=args.target_speed,
            return_hidden=True,
        )
        va = tuple(x.to(DEVICE) for x in to_episodes(*va, EPISODE_LEN))
        va_hidden = (
            va_hidden.view(-1, EPISODE_LEN, 3).transpose(1, 2).reshape(-1, EPISODE_LEN).to(DEVICE)
        )
        env_states = [sample_env_state() for _ in range(NUM_EVAL_STATES)]
        target_vels = [
            sample_target_velocities(s[1], speed=args.target_speed, max_steps=ROLLOUT_STEPS)
            for s in env_states
        ]
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
            "feedback_mix": (
                lambda: FeedbackMicroVLA(vocab_size=VOCAB_SIZE, gate_mode="mix"),
                train_feedback_vla,
                val_ce_feedback,
                True,
            ),
            "feedback_switch": (
                lambda: FeedbackMicroVLA(vocab_size=VOCAB_SIZE, gate_mode="switch"),
                train_feedback_vla,
                val_ce_feedback,
                True,
            ),
            "sequential": (
                lambda: SequentialMicroVLA(vocab_size=VOCAB_SIZE),
                train_sequential_episodes,
                val_ce_sequential,
                False,
            ),
        }
        seed_models, seed_losses = {}, {}
        for name, (make_model, train_fn, val_fn, is_fb) in runs.items():
            seed_all(seed)  # same init RNG state for all models
            model = make_model().to(DEVICE)
            t0 = time.time()
            out = train_fn(model, *tr, steps=args.steps, batch_schedule=schedule, log_every=0)
            train_loss = out["loss"] if isinstance(out, dict) else out
            seed_models[name] = (model, is_fb)
            seed_losses[name] = train_loss

            ce = val_fn(model, *va)
            metrics = {
                "train_ce(last 20)": statistics.mean(train_loss[-20:]),
                "val_ce_visible": float(ce[~va_hidden].mean().item()),
            }
            if va_hidden.any():
                metrics["val_ce_occluded"] = float(ce[va_hidden].mean().item())
            metrics.update(
                closed_loop(
                    model, env_states, hidden_windows, is_feedback=is_fb, target_vels_list=target_vels
                )
            )
            for k, v in metrics.items():
                results[name].setdefault(k, []).append(v)
            print(
                f"{name:15s} | params {sum(p.numel() for p in model.parameters()):,} | "
                + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items())
                + f" | {time.time() - t0:.1f}s"
            )
        last_models, last_losses = seed_models, seed_losses
        last_state, last_vels, last_hidden = env_states[0], target_vels[0], hidden_windows[0]

    print(f"\n=============== summary over seeds {args.seeds} (mean ± std) ===============")
    header = f"{'metric':20s} | " + " | ".join(f"{n:>16s}" for n in variant_names)
    print(header)
    for k in results["feedback_mix"]:
        row = f"{k:20s} | " + " | ".join(f"{fmt(results[n][k]):>16s}" for n in variant_names)
        print(row)
    print(f"(success = final distance < {SUCCESS_PX}px over {NUM_EVAL_STATES} states x 3 prompts)")

    if last_models is not None and args.gif_path:
        save_overlay_gif(
            last_models,
            last_losses,
            env_state=last_state,
            target_vels=last_vels,
            hidden_steps=last_hidden,
            steps=ROLLOUT_STEPS,
            save_path=args.gif_path,
        )
    print(f"Total runtime: {time.time() - t_start:.1f}s")
