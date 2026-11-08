"""
micro_vla.py: The most atomic way to train and run inference for a Vision-Language-Action (VLA)
model in PyTorch from first principles. Inspired by Andrej Karpathy's microgpt and Google RT-2.

Core insight of RT-2 style VLAs:
Vision patches, language instruction tokens, and discretized continuous robot actions all live
in a single unified sequence modeled autoregressively by a causal Transformer with cross-entropy.
"""

import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PART I: CORE VISION-LANGUAGE-ACTION (VLA) ALGORITHM (<= 400 LINES)
# ==============================================================================

# Let there be reproducibility and hardware acceleration (Apple Silicon MPS / CUDA / CPU)
SEED = 24
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
torch.set_num_threads(6)
random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------------------
# Phase 1: Bins and Action Tokenization Helper
# ------------------------------------------------------------------------------
# Continuous 2D velocity actions (delta_x, delta_y) in [-1.0, 1.0] are quantized
# into N=32 discrete bins per axis and mapped into a unified token vocabulary.
NUM_BINS = 32
TEXT_VOCAB = ["PAD", "ACT_START", "reach", "red", "green", "blue"]
X_BIN_VOCAB = [f"X_BIN_{i}" for i in range(NUM_BINS)]
Y_BIN_VOCAB = [f"Y_BIN_{i}" for i in range(NUM_BINS)]

VOCAB = TEXT_VOCAB + X_BIN_VOCAB + Y_BIN_VOCAB
VOCAB_SIZE = len(
    VOCAB
)  # 6 text/control tokens + 32 X bins + 32 Y bins = 70 total tokens
stoi = {tok: i for i, tok in enumerate(VOCAB)}
itos = {i: tok for i, tok in enumerate(VOCAB)}

X_OFFSET = len(TEXT_VOCAB)  # Token IDs 6 .. 37  -> X_BIN_0 .. X_BIN_31
Y_OFFSET = len(TEXT_VOCAB) + NUM_BINS  # Token IDs 38 .. 69 -> Y_BIN_0 .. Y_BIN_31


def action_to_bin(
    action_val: float,
    num_bins: int = NUM_BINS,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> int:
    """Quantize a continuous action float in [min_val, max_val] to a discrete bin index in [0, num_bins - 1]."""
    clamped = max(min_val, min(max_val, float(action_val)))
    norm = (clamped - min_val) / (max_val - min_val)
    return int(round(norm * (num_bins - 1)))


def bin_to_action(
    bin_idx: int, num_bins: int = NUM_BINS, min_val: float = -1.0, max_val: float = 1.0
) -> float:
    """Dequantize a discrete bin index in [0, num_bins - 1] back to a continuous float in [min_val, max_val]."""
    clamped_idx = max(0, min(num_bins - 1, int(bin_idx)))
    norm = clamped_idx / (num_bins - 1)
    return min_val + norm * (max_val - min_val)


# ------------------------------------------------------------------------------
# Phase 2: Synthetic 2D RGB Environment and Expert Data Engine
# ------------------------------------------------------------------------------
# A 32x32 RGB workspace with 3 colored target dots (Red, Green, Blue) and a white
# 2-DOF point-mass gripper. Positions are continuous coordinates (x, y) in [2, 29].
IMG_SIZE = 32
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 8 x 8 = 64 visual patches
MAX_STEP_PX = 2.2  # pixels moved per unit action (1.0)
TARGET_SPEED = 0.7  # slow target motion (px/step) along straight lines (< MAX_STEP_PX)
MIN_TARGET_DIST = 6.0  # minimum center-to-center distance (px) so 3x3 targets never collide

COLORS = {
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "green": torch.tensor([0.0, 1.0, 0.0]),
    "blue": torch.tensor([0.0, 0.0, 1.0]),
}
GRIPPER_COLOR = torch.tensor([1.0, 1.0, 1.0])

# Perimeter tabletop target slots aligned within 4x4 patch centers
TARGET_SLOTS = [
    (5.5, 5.5),
    (17.5, 5.5),
    (25.5, 5.5),
    (5.5, 17.5),
    (25.5, 17.5),
    (5.5, 25.5),
    (17.5, 25.5),
    (25.5, 25.5),
]


def render_env(gripper_pos, targets, hidden_colors=None) -> torch.Tensor:
    """Render a 32x32x3 RGB float tensor with 3x3 colored target dots and a 3x3 white gripper.
    Any target color listed in `hidden_colors` is occluded (not drawn) for this frame."""
    hidden_colors = hidden_colors or ()
    img = torch.zeros(IMG_SIZE, IMG_SIZE, 3, dtype=torch.float32)

    def draw_square(pos, color, radius=1):
        cx, cy = int(round(float(pos[0]))), int(round(float(pos[1])))
        x0, x1 = max(0, cx - radius), min(IMG_SIZE, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(IMG_SIZE, cy + radius + 1)
        img[y0:y1, x0:x1] = color

    for color_name, t_pos in targets.items():
        if color_name not in hidden_colors:
            draw_square(t_pos, COLORS[color_name], radius=1)
    draw_square(gripper_pos, GRIPPER_COLOR, radius=1)
    return img


def sample_occlusion_window(
    episode_len: int, min_len: int = 2, max_len: int = 4, first_visible: int = 2
) -> set:
    """Pick one window of consecutive timesteps during which a target is invisible.
    The first `first_visible` frames always show the target so it can be memorized."""
    length = random.randint(min_len, max_len)
    start = random.randint(first_visible, max(first_visible, episode_len - length))
    return set(range(start, min(episode_len, start + length)))


def sample_occlusion(
    episode_len: int,
    occlusion_prob: float = 0.7,
    min_len: int = 2,
    max_len: int = 4,
    first_visible: int = 2,
) -> dict:
    """Sample independent occlusion windows per target color: {color: set_of_timesteps}."""
    return {
        c: (
            sample_occlusion_window(episode_len, min_len, max_len, first_visible)
            if occlusion_prob > 0 and random.random() < occlusion_prob
            else set()
        )
        for c in COLORS
    }


def hidden_colors_at(hidden_steps, t: int, target_color: str) -> set:
    """Resolve which target colors are hidden at step `t` from a dict {color: set} or a set of steps."""
    if not hidden_steps:
        return set()
    if isinstance(hidden_steps, dict):
        return {c for c, ts in hidden_steps.items() if t in ts}
    return {target_color} if t in hidden_steps else set()


def sample_env_state():
    """Spawn the gripper in the central workspace and 3 colored target dots at distinct perimeter slots."""
    gripper_pos = [random.uniform(11.0, 20.0), random.uniform(11.0, 20.0)]
    chosen_slots = random.sample(TARGET_SLOTS, 3)
    targets = {}
    for color_name, (sx, sy) in zip(["red", "green", "blue"], chosen_slots):
        targets[color_name] = [
            sx + random.uniform(-0.4, 0.4),
            sy + random.uniform(-0.4, 0.4),
        ]
    return gripper_pos, targets


def _segments_min_dist(p1, v1, p2, v2, max_steps: int) -> float:
    """Exact minimum Euclidean distance between two linearly moving points on t in [0, max_steps]."""
    dpx, dpy = p1[0] - p2[0], p1[1] - p2[1]
    dvx, dvy = v1[0] - v2[0], v1[1] - v2[1]
    vv = dvx * dvx + dvy * dvy
    t_star = 0.0 if vv < 1e-12 else max(0.0, min(float(max_steps), -(dpx * dvx + dpy * dvy) / vv))
    return math.hypot(dpx + t_star * dvx, dpy + t_star * dvy)


def sample_target_velocities(
    targets: dict,
    speed: float = TARGET_SPEED,
    max_steps: int = 15,
    min_dist: float = MIN_TARGET_DIST,
    margin: float = 3.5,
) -> dict:
    """Sample constant 2D velocities so each target moves slowly along a straight line within bounds
    and no two targets ever collide (continuous distance >= min_dist for all t in [0, max_steps])."""
    if speed <= 0.0:
        return {c: [0.0, 0.0] for c in COLORS}
    lo, hi = margin, IMG_SIZE - 1 - margin
    color_names = list(COLORS.keys())

    def sample_in_bounds_vel(pos):
        for _ in range(200):
            angle = random.uniform(0.0, 2.0 * math.pi)
            vx, vy = speed * math.cos(angle), speed * math.sin(angle)
            ex, ey = pos[0] + max_steps * vx, pos[1] + max_steps * vy
            if lo <= ex <= hi and lo <= ey <= hi:
                return [vx, vy]
        # Fallback: move toward workspace center
        cx, cy = (IMG_SIZE - 1) / 2.0 - pos[0], (IMG_SIZE - 1) / 2.0 - pos[1]
        norm = max(1e-6, math.hypot(cx, cy))
        return [speed * cx / norm, speed * cy / norm]

    for _ in range(500):
        vels = {c: sample_in_bounds_vel(targets[c]) for c in color_names}
        collision = False
        for i in range(len(color_names)):
            for j in range(i + 1, len(color_names)):
                c1, c2 = color_names[i], color_names[j]
                if _segments_min_dist(targets[c1], vels[c1], targets[c2], vels[c2], max_steps) < min_dist:
                    collision = True
                    break
            if collision:
                break
        if not collision:
            return vels
    return {c: [0.0, 0.0] for c in color_names}


def step_targets(targets: dict, velocities: dict = None) -> dict:
    """Advance each target one step along its linear trajectory."""
    if not velocities:
        return {c: list(pos) for c, pos in targets.items()}
    return {
        c: [pos[0] + velocities[c][0], pos[1] + velocities[c][1]]
        for c, pos in targets.items()
    }


def generate_vla_dataset(
    num_episodes: int = 500,
    episode_len: int = 15,
    occlusion_prob: float = 0.0,
    target_speed: float = 0.0,
    return_hidden: bool = False,
):
    """
    Generate on-the-fly expert trajectories using a proportional controller.
    To prevent causal confusion (ignoring the prompt and moving toward whichever dot is closest),
    each rendered state along a trajectory is paired with all 3 counterfactual language prompts.
    With probability `occlusion_prob` an episode gets a window of 2-4 frames in which the targets
    are not rendered; the expert still acts on the true target positions (requires memory).
    When `target_speed > 0`, targets move slowly along non-colliding straight lines.
    Unified Sequence Structure per timestep:
      Input Sequence:  [PATCH_0 ... PATCH_63] | ["reach", <color>] | ["ACT_START"] | [ACT_X_BIN]
      Target Sequence: [   ... ignored ...  ] | [  ... ignored ... ] |  [ACT_X_BIN]  | [ACT_Y_BIN]
    """
    images, input_seqs, target_seqs, hidden_flags = [], [], [], []
    color_names = list(COLORS.keys())

    for _ in range(num_episodes):
        gripper_pos, targets = sample_env_state()
        velocities = sample_target_velocities(targets, speed=target_speed, max_steps=episode_len)
        nav_color = random.choice(color_names)
        hidden_by_color = sample_occlusion(episode_len, occlusion_prob)

        for t in range(episode_len):
            hidden_now = {c for c, ts in hidden_by_color.items() if t in ts}
            img = render_env(gripper_pos, targets, hidden_colors=hidden_now)
            next_targets = step_targets(targets, velocities)

            # Supervise all 3 language instructions ("reach red/green/blue") on the same visual state
            # Aim at the target's next position along its line so the expert tracks moving targets
            for target_color in color_names:
                goal_pos = next_targets[target_color]
                dx = max(-1.0, min(1.0, (goal_pos[0] - gripper_pos[0]) / MAX_STEP_PX))
                dy = max(-1.0, min(1.0, (goal_pos[1] - gripper_pos[1]) / MAX_STEP_PX))

                x_bin = action_to_bin(dx)
                y_bin = action_to_bin(dy)
                act_x_tok = X_OFFSET + x_bin
                act_y_tok = Y_OFFSET + y_bin

                seq_in = [
                    stoi["reach"],
                    stoi[target_color],
                    stoi["ACT_START"],
                    act_x_tok,
                ]
                seq_tgt = [-100, -100, act_x_tok, act_y_tok]

                images.append(img)
                input_seqs.append(seq_in)
                target_seqs.append(seq_tgt)
                hidden_flags.append(t in hidden_by_color[target_color])

            # Step the environment along the active trajectory using the quantized action
            nav_goal = next_targets[nav_color]
            nav_dx = max(-1.0, min(1.0, (nav_goal[0] - gripper_pos[0]) / MAX_STEP_PX))
            nav_dy = max(-1.0, min(1.0, (nav_goal[1] - gripper_pos[1]) / MAX_STEP_PX))
            xb, yb = action_to_bin(nav_dx), action_to_bin(nav_dy)
            gripper_pos = [
                max(2.0, min(29.0, gripper_pos[0] + bin_to_action(xb) * MAX_STEP_PX)),
                max(2.0, min(29.0, gripper_pos[1] + bin_to_action(yb) * MAX_STEP_PX)),
            ]
            targets = next_targets

    data = (
        torch.stack(images),  # [N, 32, 32, 3]
        torch.tensor(input_seqs, dtype=torch.long),  # [N, 4]
        torch.tensor(target_seqs, dtype=torch.long),  # [N, 4]
    )
    if return_hidden:
        return data + (torch.tensor(hidden_flags, dtype=torch.bool),)  # [N] targets occluded?
    return data


# ------------------------------------------------------------------------------
# Phase 3: Model Implementation (MicroPatchEmbed & MicroVLA)
# ------------------------------------------------------------------------------
class MicroPatchEmbed(nn.Module):
    """Divide 32x32x3 image into 4x4 non-overlapping patches (64 total) and project to d_model."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, d_model: int = 64):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.patch_dim = patch_size * patch_size * 3  # 4 * 4 * 3 = 48
        self.patch_size = patch_size
        self.proj = nn.Linear(self.patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 32, 32, 3] -> reshape into [B, 8, 4, 8, 4, 3] -> [B, 64, 48]
        B = img.size(0)
        p = self.patch_size
        g = self.grid_size
        patches = (
            img.view(B, g, p, g, p, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, self.num_patches, self.patch_dim)
        )
        return self.proj(patches) + self.pos_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention where each token only attends to current and past positions."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, T, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    """Pre-norm Causal Transformer Decoder block (Attention + 2-layer GELU MLP)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MicroVLA(nn.Module):
    """
    RT-2 style Vision-Language-Action Transformer.
    Concatenates 64 vision patch embeddings with text instruction and action tokens into a single
    causal stream, predicting discretized motor bin tokens via standard next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.patch_embed = MicroPatchEmbed(IMG_SIZE, PATCH_SIZE, d_model)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.token_pos_emb = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, img: torch.Tensor, token_ids: torch.Tensor, targets: torch.Tensor = None
    ):
        B, seq_len = token_ids.shape
        v_emb = self.patch_embed(img)  # [B, 64, d_model]
        t_emb = (
            self.token_embed(token_ids) + self.token_pos_emb[:, :seq_len, :]
        )  # [B, Seq_Len, d_model]
        x = torch.cat([v_emb, t_emb], dim=1)  # [B, 64 + Seq_Len, d_model]

        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # [B, 64 + Seq_Len, vocab_size]

        loss = None
        if targets is not None:
            # Ignore loss on the 64 vision patch positions; supervise only action token predictions
            vision_ignore = torch.full(
                (B, NUM_PATCHES), -100, dtype=torch.long, device=targets.device
            )
            full_targets = torch.cat(
                [vision_ignore, targets], dim=1
            )  # [B, 64 + Seq_Len]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                full_targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss


# ------------------------------------------------------------------------------
# Phase 4: Training and Autoregressive Evaluation Loop
# ------------------------------------------------------------------------------
def train_vla(
    model: MicroVLA,
    images,
    input_seqs,
    target_seqs,
    steps: int = 520,
    batch_size: int = 72,
    lr: float = 4e-3,
    device: torch.device = DEVICE,
):
    """Train MicroVLA with AdamW and cross-entropy loss over discretized action tokens."""
    model.to(device)
    images = images.to(device)
    input_seqs = input_seqs.to(device)
    target_seqs = target_seqs.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_states = images.size(0) // 3
    states_per_batch = batch_size // 3
    decay_start = int(steps * 0.71)
    loss_history = []
    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        if step > decay_start:
            frac = (step - decay_start) / max(1, steps - decay_start)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * max(0.1, 1.0 - 0.9 * frac)

        # Sample random visual states and include all 3 counterfactual prompts per state
        state_idx = torch.randint(0, num_states, (states_per_batch,), device=device) * 3
        idx = torch.cat([state_idx, state_idx + 1, state_idx + 2])
        _, loss = model(images[idx], input_seqs[idx], target_seqs[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)
        if step % 130 == 0 or step == 1 or step == steps:
            elapsed = time.time() - t0
            print(
                f"step {step:4d} / {steps:4d} | loss: {loss_val:.4f} | elapsed: {elapsed:5.2f}s"
            )

    return loss_history


@torch.no_grad()
def run_inference(
    model: MicroVLA,
    prompt: str = "reach blue",
    steps: int = 15,
    env_state=None,
    hidden_steps=None,
    target_velocities=None,
    target_speed: float = 0.0,
):
    """
    Closed-loop autoregressive rollout on an unseen environment state:
      1. Feed 32x32 RGB image + ["reach", <color>, "ACT_START"] -> predict ACT_X_BIN -> delta_x
      2. Append ACT_X_BIN -> predict ACT_Y_BIN -> delta_y
      3. Step synthetic environment and repeat for `steps` timesteps.
    Targets are not rendered at timesteps in `hidden_steps` (occlusion) and move along
    non-colliding straight lines when `target_velocities` or `target_speed > 0` is provided.
    """
    hidden_steps = hidden_steps or set()
    model.eval()
    words = prompt.strip().lower().split()
    assert (
        len(words) == 2 and words[0] == "reach" and words[1] in COLORS
    ), f"Unsupported prompt: {prompt}"
    target_color = words[1]

    gripper_pos, raw_targets = (
        sample_env_state() if env_state is None else (list(env_state[0]), env_state[1])
    )
    targets = {c: list(p) for c, p in raw_targets.items()}
    if target_velocities is None and target_speed > 0.0:
        target_velocities = sample_target_velocities(targets, speed=target_speed, max_steps=steps)
    goal_pos = targets[target_color]
    prompt_prefix = [stoi["reach"], stoi[target_color], stoi["ACT_START"]]

    frames, trajectory, step_records = [], [list(gripper_pos)], []
    dev = next(model.parameters()).device

    for t in range(steps):
        img = render_env(
            gripper_pos, targets, hidden_colors=hidden_colors_at(hidden_steps, t, target_color)
        )
        frames.append(img)
        img_batch = img.unsqueeze(0).to(dev)  # [1, 32, 32, 3]

        # Step 1: Autoregressively predict ACT_X_BIN from [PATCH_0..63, "reach", color, "ACT_START"]
        inp_x = torch.tensor([prompt_prefix], dtype=torch.long, device=dev)
        logits_x, _ = model(img_batch, inp_x)
        x_logits = logits_x[0, -1, X_OFFSET : X_OFFSET + NUM_BINS]
        x_probs = F.softmax(x_logits, dim=-1)
        x_bin = int(torch.argmax(x_probs).item())
        act_x_tok = X_OFFSET + x_bin
        dx = bin_to_action(x_bin)

        # Step 2: Append ACT_X_BIN and autoregressively predict ACT_Y_BIN
        inp_y = torch.tensor([prompt_prefix + [act_x_tok]], dtype=torch.long, device=dev)
        logits_y, _ = model(img_batch, inp_y)
        y_logits = logits_y[0, -1, Y_OFFSET : Y_OFFSET + NUM_BINS]
        y_probs = F.softmax(y_logits, dim=-1)
        y_bin = int(torch.argmax(y_probs).item())
        act_y_tok = Y_OFFSET + y_bin
        dy = bin_to_action(y_bin)

        # Step 3: Advance the continuous 2D environment (gripper + moving targets)
        gripper_pos = [
            max(2.0, min(29.0, gripper_pos[0] + dx * MAX_STEP_PX)),
            max(2.0, min(29.0, gripper_pos[1] + dy * MAX_STEP_PX)),
        ]
        targets = step_targets(targets, target_velocities)
        goal_pos = targets[target_color]
        dist = math.hypot(goal_pos[0] - gripper_pos[0], goal_pos[1] - gripper_pos[1])
        trajectory.append(list(gripper_pos))
        step_records.append(
            {
                "step": t + 1,
                "x_tok": itos[act_x_tok],
                "y_tok": itos[act_y_tok],
                "dx": dx,
                "dy": dy,
                "dist": dist,
                "x_probs": x_probs.cpu(),
                "y_probs": y_probs.cpu(),
            }
        )

    frames.append(render_env(gripper_pos, targets))
    return {
        "prompt": prompt,
        "target_color": target_color,
        "targets": targets,
        "frames": frames,
        "trajectory": trajectory,
        "records": step_records,
    }


# ==============================================================================
# PART II: VISUALIZATION & DEMO INFRASTRUCTURE (ASCII + ANIMATED GIF)
# ==============================================================================


def print_ascii_rollout(rollout: dict):
    """Print an ASCII map of the 2D workspace and step-by-step autoregressive token predictions."""
    grid_w, grid_h = 32, 16  # 2:1 aspect adjustment for terminal character cells
    grid = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    def to_cell(pos):
        cx = max(0, min(grid_w - 1, int(round(pos[0]))))
        cy = max(0, min(grid_h - 1, int(round(pos[1] * (grid_h - 1) / (IMG_SIZE - 1)))))
        return cx, cy

    # Plot Colored Target Dots (R, G, B) first, then Gripper trajectory (0..9, final as '@')
    for cname, tpos in rollout["targets"].items():
        cx, cy = to_cell(tpos)
        grid[cy][cx] = cname[0].upper()

    for idx, pos in enumerate(rollout["trajectory"]):
        cx, cy = to_cell(pos)
        grid[cy][cx] = str(min(9, idx)) if idx < len(rollout["trajectory"]) - 1 else "@"

    start_pos = rollout["trajectory"][0]
    goal_pos = rollout["targets"][rollout["target_color"]]
    init_dist = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
    final_dist = rollout["records"][-1]["dist"]
    min_dist = min(r["dist"] for r in rollout["records"])

    print(f"\n=== Autoregressive Rollout: '{rollout['prompt']}' ===")
    print(
        f"Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}) -> Goal ({rollout['target_color'].upper()}): "
        f"({goal_pos[0]:.1f}, {goal_pos[1]:.1f}) | Distance: {init_dist:.2f}px -> {final_dist:.2f}px (min: {min_dist:.2f}px)"
    )
    print("+" + "-" * grid_w + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_w + "+")
    print("Legend: [R/G/B] Targets | [0..9] Gripper Path | [@] Final Gripper Position")
    for rec in rollout["records"][:5] + rollout["records"][-2:]:
        print(
            f"  t={rec['step']:2d} | tokens=({rec['x_tok']:8s}, {rec['y_tok']:8s}) "
            f"-> action=({rec['dx']:+5.2f}, {rec['dy']:+5.2f}) | dist_to_goal={rec['dist']:5.2f}px"
        )


def save_demo_gif(
    rollouts: list,
    loss_history: list,
    save_path: str = "micro_vla_demo.gif",
    title: str = "micro_vla.py — Vision-Language-Action Transformer (64 Vision Patches + Unified Action Tokens)",
):
    """Save an animated GIF visualizing the RGB camera observations, patch grid, and action bin distributions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(12, 6.2), facecolor="#0d1117")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.28)

    # Top row: 3 unseen rollouts ("reach blue", "reach red", "reach green")
    env_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # Bottom row: X action bin probabilities, Y action bin probabilities, and Training Loss curve
    ax_xbin = fig.add_subplot(gs[1, 0])
    ax_ybin = fig.add_subplot(gs[1, 1])
    ax_loss = fig.add_subplot(gs[1, 2])

    for ax in env_axes + [ax_xbin, ax_ybin, ax_loss]:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

    # Static Training Loss Plot
    ax_loss.plot(
        range(1, len(loss_history) + 1), loss_history, color="#58a6ff", linewidth=1.5
    )
    ax_loss.set_title(
        "VLA Action Token Cross-Entropy", color="#e6edf3", fontsize=10, pad=6
    )
    ax_loss.set_xlabel("Training Step", color="#8b949e", fontsize=8)
    ax_loss.set_ylabel("Loss", color="#8b949e", fontsize=8)
    ax_loss.grid(True, color="#21262d", linestyle="--", linewidth=0.6)

    im_artists, path_lines, head_dots, step_texts = [], [], [], []
    title_colors = {"blue": "#58a6ff", "red": "#ff7b72", "green": "#3fb950"}

    for i, r in enumerate(rollouts):
        ax = env_axes[i]
        # Draw subtle 4x4 patch boundaries (8x8 grid)
        for p in range(0, IMG_SIZE, PATCH_SIZE):
            ax.axhline(p - 0.5, color="#30363d", linewidth=0.4, alpha=0.5)
            ax.axvline(p - 0.5, color="#30363d", linewidth=0.4, alpha=0.5)

        im = ax.imshow(r["frames"][0].numpy(), origin="upper", vmin=0.0, vmax=1.0)
        (line,) = ax.plot(
            [], [], color="#f0f6fc", linestyle="--", linewidth=1.3, alpha=0.85
        )
        (dot,) = ax.plot(
            [], [], marker="o", markersize=5, color="#ffffff", markeredgecolor="#000000"
        )
        txt = ax.text(
            1.0,
            29.8,
            "",
            color="#e6edf3",
            fontsize=7.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#0d1117",
                alpha=0.75,
                edgecolor="none",
            ),
        )

        cname = r["target_color"]
        ax.set_title(
            f"Prompt: \"{r['prompt']}\"",
            color=title_colors[cname],
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        im_artists.append(im)
        path_lines.append(line)
        head_dots.append(dot)
        step_texts.append(txt)

    bin_x_vals = [bin_to_action(b) for b in range(NUM_BINS)]
    bars_x = ax_xbin.bar(
        bin_x_vals, [0.0] * NUM_BINS, width=0.05, color="#58a6ff", alpha=0.9
    )
    ax_xbin.set_xlim(-1.08, 1.08)
    ax_xbin.set_ylim(0.0, 1.0)
    ax_xbin.set_title(
        "Autoregressive P(ACT_X_BIN) — 'reach blue'",
        color="#e6edf3",
        fontsize=9.5,
        pad=6,
    )
    ax_xbin.set_xlabel("Decoded Velocity delta_x", color="#8b949e", fontsize=8)
    ax_xbin.set_ylabel("Probability", color="#8b949e", fontsize=8)

    bars_y = ax_ybin.bar(
        bin_x_vals, [0.0] * NUM_BINS, width=0.05, color="#d2a8ff", alpha=0.9
    )
    ax_ybin.set_xlim(-1.08, 1.08)
    ax_ybin.set_ylim(0.0, 1.0)
    ax_ybin.set_title(
        "Autoregressive P(ACT_Y_BIN) — 'reach blue'",
        color="#e6edf3",
        fontsize=9.5,
        pad=6,
    )
    ax_ybin.set_xlabel("Decoded Velocity delta_y", color="#8b949e", fontsize=8)

    fig.suptitle(
        title,
        color="#f0f6fc",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    num_frames = len(rollouts[0]["frames"])

    def update(frame_idx):
        for i, r in enumerate(rollouts):
            im_artists[i].set_data(r["frames"][frame_idx].numpy())
            traj_slice = r["trajectory"][: frame_idx + 1]
            xs = [p[0] for p in traj_slice]
            ys = [p[1] for p in traj_slice]
            path_lines[i].set_data(xs, ys)
            head_dots[i].set_data([xs[-1]], [ys[-1]])

            rec_idx = min(frame_idx, len(r["records"]) - 1)
            rec = r["records"][rec_idx]
            label = f"t={frame_idx:2d} dist={rec['dist']:.1f}px\n({rec['dx']:+.2f},{rec['dy']:+.2f})"
            if "choice" in rec:  # feedback model: show which weight set was selected
                label += f" W:{'AB'[rec['choice']]}"
            step_texts[i].set_text(label)

        # Update action token probability histograms for the primary ("reach blue") rollout
        rec_blue = rollouts[0]["records"][
            min(frame_idx, len(rollouts[0]["records"]) - 1)
        ]
        for b, bar in enumerate(bars_x):
            bar.set_height(float(rec_blue["x_probs"][b]))
        for b, bar in enumerate(bars_y):
            bar.set_height(float(rec_blue["y_probs"][b]))
        return im_artists + path_lines + head_dots + step_texts

    anim = FuncAnimation(fig, update, frames=num_frames, interval=220, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"\nSaved animated rollout visualization to: {save_path}")


if __name__ == "__main__":
    t_start = time.time()
    print(
        f"Unified VLA Vocabulary Size: {VOCAB_SIZE} tokens "
        f"(6 text/control + {NUM_BINS} X-bins + {NUM_BINS} Y-bins)"
    )

    print(
        "Generating synthetic expert demonstrations (250 episodes x 10 steps x 3 prompts = 7,500 sequences)..."
    )
    images, input_seqs, target_seqs = generate_vla_dataset(
        num_episodes=250, episode_len=10
    )

    model = MicroVLA(vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4)
    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"Initialized MicroVLA | Parameters: {num_params:,} | Sequence Length: {NUM_PATCHES + 4} tokens"
    )

    print("\n--- Training MicroVLA on CPU ---")
    loss_history = train_vla(
        model, images, input_seqs, target_seqs, steps=520, batch_size=72, lr=4e-3
    )

    # Evaluate on an unseen environment configuration across all three language instructions
    eval_env_state = sample_env_state()
    rollouts = [
        run_inference(model, prompt="reach blue", steps=15, env_state=eval_env_state),
        run_inference(model, prompt="reach red", steps=15, env_state=eval_env_state),
        run_inference(model, prompt="reach green", steps=15, env_state=eval_env_state),
    ]

    for r in rollouts:
        print_ascii_rollout(r)

    save_demo_gif(rollouts, loss_history, save_path="micro_vla_demo.gif")
    print(
        f"Total runtime (data + train + inference + GIF): {time.time() - t_start:.2f}s"
    )
