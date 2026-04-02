import csv
import os

BASE_OUTPUT_DIR = "outputs"

runs = sorted(os.listdir(BASE_OUTPUT_DIR))
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, runs[-1])

print("Using run folder:", RUN_DIR)

INPUT_CSV  = os.path.join(RUN_DIR, "detections.csv")
OUTPUT_CSV = os.path.join(RUN_DIR, "tracked_players.csv")

out    = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "x1", "y1", "x2", "y2"
])

gameplay_count = 0

with open(INPUT_CSV, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        # ── Only process confirmed gameplay frames ──────────────────
        if r.get("is_gameplay", "0") != "1":
            continue

        frame = int(r["frame"])
        gameplay_count += 1

        if r["p1_x1"] not in ["", "None"]:
            writer.writerow([
                frame, 1,
                float(r["p1_x1"]), float(r["p1_y1"]),
                float(r["p1_x2"]), float(r["p1_y2"])
            ])

        if r["p2_x1"] not in ["", "None"]:
            writer.writerow([
                frame, 2,
                float(r["p2_x1"]), float(r["p2_y1"]),
                float(r["p2_x2"]), float(r["p2_y2"])
            ])

out.close()

print(f"Gameplay frames written: {gameplay_count}")
print("Tracked players saved to:", OUTPUT_CSV)