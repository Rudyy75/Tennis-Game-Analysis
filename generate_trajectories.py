import csv
import os

BASE_OUTPUT_DIR = "outputs"

runs = sorted(os.listdir(BASE_OUTPUT_DIR))
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, runs[-1])

print("Using run folder:", RUN_DIR)

TRACKED_PLAYERS = os.path.join(RUN_DIR, "tracked_players.csv")
DETECTIONS      = os.path.join(RUN_DIR, "detections.csv")
OUTPUT          = os.path.join(RUN_DIR, "trajectories.csv")


def center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


# ── Build ball position map (gameplay frames only) ──────────────────────────
# Only gameplay frames have ball detections anyway (ball model skipped on
# non-gameplay frames in players_ball_detections.py), but we gate here too
# so this script stays correct even if run against older CSVs.

ball_positions = {}   # frame → (cx, cy) or None

with open(DETECTIONS, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        # Skip non-gameplay rows — don't want None ball entries for those frames
        # polluting the interpolation in ball_processing.py
        if r.get("is_gameplay", "1") != "1":
            continue

        frame = int(r["frame"])

        if r["ball_x1"] not in ["", "None"]:
            x1 = float(r["ball_x1"])
            y1 = float(r["ball_y1"])
            x2 = float(r["ball_x2"])
            y2 = float(r["ball_y2"])
            ball_positions[frame] = center(x1, y1, x2, y2)
        else:
            ball_positions[frame] = None


# ── Write combined trajectories ─────────────────────────────────────────────
# tracked_players.csv already contains only gameplay frames (from
# tracking_bytetrack.py), so no extra filter needed here.

out    = open(OUTPUT, "w", newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "cx", "cy",
    "ball_x", "ball_y"
])

with open(TRACKED_PLAYERS, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame     = int(r["frame"])
        player_id = int(r["player_id"])

        cx, cy = center(
            float(r["x1"]), float(r["y1"]),
            float(r["x2"]), float(r["y2"])
        )

        _ball          = ball_positions.get(frame, (None, None))
        ball_x, ball_y = _ball if _ball is not None else (None, None)

        writer.writerow([frame, player_id, cx, cy, ball_x, ball_y])

out.close()

print("Combined trajectories saved to:", OUTPUT)