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

INPUT_CSV  = os.path.join(RUN_DIR, "detections.csv")
OUTPUT_CSV = os.path.join(RUN_DIR, "tracked_players.csv")

# ======================================
# TRACKING BY Y-COORDINATE
#
# No ByteTrack needed.
#
# Player identity is already correctly assigned in detections.csv
# by the net-based assignment in players_ball_detections.py:
#   Player1 = bottom player (near side, cy > net Y)
#   Player2 = top player    (far side,  cy < net Y)
#
# This script just reads those assignments and writes tracked_players.csv
# in the format expected by the rest of the pipeline.
#
# For frames where only one player is detected, that player is written
# with whatever ID was assigned during detection.
# ======================================

out = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "x1", "y1", "x2", "y2"
])

with open(INPUT_CSV, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])

        # Player 1 (bottom)
        if r["p1_x1"] not in ["", "None"]:
            writer.writerow([
                frame,
                1,
                float(r["p1_x1"]),
                float(r["p1_y1"]),
                float(r["p1_x2"]),
                float(r["p1_y2"])
            ])

        # Player 2 (top)
        if r["p2_x1"] not in ["", "None"]:
            writer.writerow([
                frame,
                2,
                float(r["p2_x1"]),
                float(r["p2_y1"]),
                float(r["p2_x2"]),
                float(r["p2_y2"])
            ])

out.close()

print("Tracked players saved to:", OUTPUT_CSV)
