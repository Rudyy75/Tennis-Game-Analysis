import csv
import os
import math

BASE_OUTPUT_DIR = "outputs"

runs = sorted(os.listdir(BASE_OUTPUT_DIR))
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, runs[-1])

print("Using run folder:", RUN_DIR)

INPUT  = os.path.join(RUN_DIR, "trajectories.csv")
OUTPUT = os.path.join(RUN_DIR, "ball_processed.csv")

# Maximum gap (in frame numbers) across which we allow ball interpolation.
# Gameplay frames may have large frame-number gaps between them (non-gameplay
# frames were dropped), so we need a sensible cap to avoid interpolating
# a ball position across a scene cut or between-point gap.
MAX_INTERP_GAP = 15   # frames — tune based on your FPS

rows = []

with open(INPUT, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# ── Build per-frame ball map ────────────────────────────────────────────────
# Use only unique frames (trajectories has one row per player per frame).

frame_ball = {}   # frame → (bx, by) or None

for r in rows:
    frame = int(r["frame"])
    bx    = r["ball_x"]
    by    = r["ball_y"]
    if frame not in frame_ball:                 # avoid duplicate frame entries
        frame_ball[frame] = (float(bx), float(by)) if bx not in ["", "None"] else None

frames         = sorted(frame_ball.keys())
ball_positions = [frame_ball[f] for f in frames]


# ── Gap-aware interpolation ─────────────────────────────────────────────────
# Only interpolate missing ball positions if both neighbouring detections
# exist AND the frame-number gap between them is small (≤ MAX_INTERP_GAP).
# This prevents bridging across rallies / set changes.

for i in range(len(ball_positions)):

    if ball_positions[i] is not None:
        continue

    # Search backward for last known position
    prev_idx, prev_pos = None, None
    for j in range(i - 1, -1, -1):
        if ball_positions[j] is not None:
            prev_idx, prev_pos = j, ball_positions[j]
            break

    # Search forward for next known position
    next_idx, next_pos = None, None
    for j in range(i + 1, len(ball_positions)):
        if ball_positions[j] is not None:
            next_idx, next_pos = j, ball_positions[j]
            break

    if prev_pos is not None and next_pos is not None:
        gap = frames[next_idx] - frames[prev_idx]
        if gap <= MAX_INTERP_GAP:
            # Linear interpolation between prev and next
            t = (i - prev_idx) / (next_idx - prev_idx)
            x = prev_pos[0] + t * (next_pos[0] - prev_pos[0])
            y = prev_pos[1] + t * (next_pos[1] - prev_pos[1])
            ball_positions[i] = (x, y)
        else:
            # Gap too large — use nearest neighbour only
            ball_positions[i] = prev_pos
    elif prev_pos is not None:
        ball_positions[i] = prev_pos
    elif next_pos is not None:
        ball_positions[i] = next_pos
    # If still None → no data at all, leave as None (shouldn't happen for
    # trajectories.csv which only has gameplay frames)


# ── Temporal smoothing ──────────────────────────────────────────────────────

smoothed = []
window   = 3

for i in range(len(ball_positions)):

    xs, ys = [], []

    for j in range(i - window, i + window + 1):
        if 0 <= j < len(ball_positions) and ball_positions[j] is not None:
            xs.append(ball_positions[j][0])
            ys.append(ball_positions[j][1])

    if xs:
        smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    else:
        smoothed.append(None)


# ── Velocity (pixels / frame) ───────────────────────────────────────────────

velocities = []

for i in range(len(smoothed)):

    if i == 0 or smoothed[i] is None or smoothed[i - 1] is None:
        velocities.append(0.0)
    else:
        x1, y1 = smoothed[i - 1]
        x2, y2 = smoothed[i]
        velocities.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


# ── Build lookup ────────────────────────────────────────────────────────────

frame_ball_processed = {}   # frame → (bx, by, vel)

for i, f in enumerate(frames):
    if smoothed[i] is not None:
        bx, by = smoothed[i]
        frame_ball_processed[f] = (bx, by, velocities[i])
    else:
        frame_ball_processed[f] = (None, None, 0.0)


# ── Write output ────────────────────────────────────────────────────────────

out    = open(OUTPUT, "w", newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "player_x", "player_y",
    "ball_x", "ball_y",
    "ball_velocity"
])

for r in rows:

    frame = int(r["frame"])
    pid   = r["player_id"]
    px    = r["cx"]
    py    = r["cy"]

    bx, by, vel = frame_ball_processed.get(frame, (None, None, 0.0))

    writer.writerow([frame, pid, px, py, bx, by, vel])

out.close()

print("Processed ball trajectory saved to:", OUTPUT)