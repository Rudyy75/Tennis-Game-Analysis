import csv
import numpy as np
import supervision as sv

INPUT_CSV = "outputs1/detections.csv"
OUTPUT_CSV = "outputs1/tracked_players.csv"

tracker = sv.ByteTrack()

frames = {}

# =============================
# LOAD DETECTIONS
# =============================

with open(INPUT_CSV, "r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])

        boxes = []

        if r["p1_x1"] not in ["","None"]:
            boxes.append([
                float(r["p1_x1"]),
                float(r["p1_y1"]),
                float(r["p1_x2"]),
                float(r["p1_y2"])
            ])

        if r["p2_x1"] not in ["","None"]:
            boxes.append([
                float(r["p2_x1"]),
                float(r["p2_y1"]),
                float(r["p2_x2"]),
                float(r["p2_y2"])
            ])

        frames[frame] = boxes

# =============================
# TRACKING
# =============================

out = open(OUTPUT_CSV,"w",newline="")
writer = csv.writer(out)

writer.writerow([
    "frame",
    "player_id",
    "x1","y1","x2","y2"
])

last_players = {}

for frame_id in sorted(frames.keys()):

    boxes = frames[frame_id]

    if len(boxes) > 0:

        xyxy = np.array(boxes)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=np.ones(len(xyxy)),
            class_id=np.zeros(len(xyxy))
        )

        tracked = tracker.update_with_detections(detections)

        players = []

        for box in tracked.xyxy:

            x1,y1,x2,y2 = box
            cx = (x1+x2)/2

            players.append((cx,x1,y1,x2,y2))

        players.sort(key=lambda x:x[0])

        current = {}

        for i,p in enumerate(players[:2]):

            _,x1,y1,x2,y2 = p

            current[i+1] = (x1,y1,x2,y2)

        last_players = current

    # write using last known boxes
    for pid in [1,2]:

        if pid in last_players:

            x1,y1,x2,y2 = last_players[pid]

            writer.writerow([
                frame_id,
                pid,
                x1,y1,x2,y2
            ])

out.close()

print("Tracked players saved to:",OUTPUT_CSV)