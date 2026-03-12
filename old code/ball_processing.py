import csv
import os
import math

# ======================================
# AUTO DETECT LATEST RUN FOLDER
# ======================================

BASE_OUTPUT_DIR = "outputs"

runs = sorted(os.listdir(BASE_OUTPUT_DIR))
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, runs[-1])

print("Using run folder:", RUN_DIR)

# ======================================
# PATHS
# ======================================

INPUT = os.path.join(RUN_DIR, "trajectories.csv")
OUTPUT = os.path.join(RUN_DIR, "ball_processed.csv")

# ======================================
# LOAD DATA
# ======================================

rows = []

with open(INPUT, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# ======================================
# COLLECT BALL TRAJECTORY BY FRAME
# ======================================

frame_ball = {}

for r in rows:

    frame = int(r["frame"])
    bx = r["ball_x"]
    by = r["ball_y"]

    if bx not in ["", "None"]:
        frame_ball[frame] = (float(bx), float(by))
    else:
        frame_ball[frame] = None

frames = sorted(frame_ball.keys())

ball_positions = []

for f in frames:
    ball_positions.append(frame_ball[f])

# ======================================
# INTERPOLATE MISSING BALL
# ======================================

for i in range(len(ball_positions)):

    if ball_positions[i] is None:

        prev = None
        nextp = None

        for j in range(i-1, -1, -1):
            if ball_positions[j] is not None:
                prev = ball_positions[j]
                break

        for j in range(i+1, len(ball_positions)):
            if ball_positions[j] is not None:
                nextp = ball_positions[j]
                break

        if prev and nextp:

            x = (prev[0] + nextp[0]) / 2
            y = (prev[1] + nextp[1]) / 2

            ball_positions[i] = (x, y)

        elif prev:
            ball_positions[i] = prev

        elif nextp:
            ball_positions[i] = nextp

# ======================================
# SMOOTH TRAJECTORY
# ======================================

smoothed = []
window = 3

for i in range(len(ball_positions)):

    xs = []
    ys = []

    for j in range(i-window, i+window+1):

        if 0 <= j < len(ball_positions):

            x, y = ball_positions[j]

            xs.append(x)
            ys.append(y)

    sx = sum(xs) / len(xs)
    sy = sum(ys) / len(ys)

    smoothed.append((sx, sy))

# ======================================
# COMPUTE VELOCITY
# ======================================

velocities = []

for i in range(len(smoothed)):

    if i == 0:
        velocities.append(0)

    else:

        x1, y1 = smoothed[i-1]
        x2, y2 = smoothed[i]

        v = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        velocities.append(v)

# ======================================
# MAP FRAME → BALL DATA
# ======================================

frame_ball_processed = {}

for i, f in enumerate(frames):

    bx, by = smoothed[i]
    vel = velocities[i]

    frame_ball_processed[f] = (bx, by, vel)

# ======================================
# SAVE OUTPUT
# ======================================

out = open(OUTPUT, "w", newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "player_x",
    "player_y",
    "ball_x",
    "ball_y",
    "ball_velocity"
])

for r in rows:

    frame = int(r["frame"])
    pid = r["player_id"]

    px = r["cx"]
    py = r["cy"]

    bx, by, vel = frame_ball_processed[frame]

    writer.writerow([
        frame,
        pid,
        px,
        py,
        bx,
        by,
        vel
    ])

out.close()

print("Processed ball trajectory saved to:", OUTPUT)