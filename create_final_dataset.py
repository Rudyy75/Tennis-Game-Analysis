import csv
import os

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

TRACKED_PLAYERS = os.path.join(RUN_DIR, "tracked_players.csv")
BALL_PROCESSED  = os.path.join(RUN_DIR, "ball_processed.csv")
POSE_CSV        = os.path.join(RUN_DIR, "pose", "pose_outputs_csv", "pose_keypoints.csv")

OUTPUT = os.path.join(RUN_DIR, "final_dataset.csv")

# ======================================
# LOAD BALL DATA
# ======================================

ball_data = {}

with open(BALL_PROCESSED, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])
        pid   = int(r["player_id"])

        ball_data[(frame, pid)] = {
            "player_x":     r["player_x"],
            "player_y":     r["player_y"],
            "ball_x":       r["ball_x"],
            "ball_y":       r["ball_y"],
            "ball_velocity": r["ball_velocity"]
        }

# ======================================
# LOAD POSE DATA
# ======================================

pose_data = {}

with open(POSE_CSV, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])
        pid   = int(r["player_id"])

        keypoints = []

        for i in range(17):
            keypoints.append(r[f"kp{i}_x"])
            keypoints.append(r[f"kp{i}_y"])
            keypoints.append(r[f"kp{i}_conf"])

        pose_data[(frame, pid)] = keypoints

# ======================================
# CREATE HEADER
# ======================================

header = [
    "frame",
    "player_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "player_x",
    "player_y",
    "ball_x",
    "ball_y",
    "ball_velocity"
]

for i in range(17):
    header += [
        f"kp{i}_x",
        f"kp{i}_y",
        f"kp{i}_conf"
    ]

# ======================================
# WRITE FINAL DATASET
# ======================================

out = open(OUTPUT, "w", newline="")
writer = csv.writer(out)

writer.writerow(header)

with open(TRACKED_PLAYERS, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])
        pid   = int(r["player_id"])

        x1 = r["x1"]
        y1 = r["y1"]
        x2 = r["x2"]
        y2 = r["y2"]

        key = (frame, pid)

        if key not in ball_data:
            continue

        if key not in pose_data:
            continue

        player_x     = ball_data[key]["player_x"]
        player_y     = ball_data[key]["player_y"]
        ball_x       = ball_data[key]["ball_x"]
        ball_y       = ball_data[key]["ball_y"]
        ball_velocity = ball_data[key]["ball_velocity"]

        keypoints = pose_data[key]

        row = [
            frame,
            pid,
            x1, y1, x2, y2,
            player_x,
            player_y,
            ball_x,
            ball_y,
            ball_velocity
        ] + keypoints

        writer.writerow(row)

out.close()

print("Final dataset saved to:", OUTPUT)
