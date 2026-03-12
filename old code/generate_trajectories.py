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
DETECTIONS = os.path.join(RUN_DIR, "detections.csv")

OUTPUT = os.path.join(RUN_DIR, "trajectories.csv")


def center(x1,y1,x2,y2):
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    return cx,cy


# ======================================
# LOAD BALL DATA
# ======================================

ball_positions = {}

with open(DETECTIONS,"r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])

        if r["ball_x1"] not in ["","None"]:

            x1 = float(r["ball_x1"])
            y1 = float(r["ball_y1"])
            x2 = float(r["ball_x2"])
            y2 = float(r["ball_y2"])

            cx,cy = center(x1,y1,x2,y2)

            ball_positions[frame] = (cx,cy)

        else:
            ball_positions[frame] = (None,None)


# ======================================
# GENERATE TRAJECTORIES
# ======================================

out = open(OUTPUT,"w",newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "cx",
    "cy",
    "ball_x",
    "ball_y"
])


with open(TRACKED_PLAYERS,"r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])
        player_id = int(r["player_id"])

        x1 = float(r["x1"])
        y1 = float(r["y1"])
        x2 = float(r["x2"])
        y2 = float(r["y2"])

        cx,cy = center(x1,y1,x2,y2)

        ball_x,ball_y = ball_positions.get(frame,(None,None))

        writer.writerow([
            frame,
            player_id,
            cx,
            cy,
            ball_x,
            ball_y
        ])

out.close()

print("Combined trajectories saved to:", OUTPUT)