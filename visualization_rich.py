import cv2
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import os

# =======================
# CONFIGURATION
# =======================

# FIXED PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "input", "V010.mp4")

PREDICTIONS_CSV  = os.path.join(BASE_DIR, "outputs", "run_20260310_201633", "dataset_with_predictions.csv")
DETECTIONS_CSV   = os.path.join(BASE_DIR, "outputs", "run_20260310_201633", "detections.csv")
TRAJECTORIES_CSV = os.path.join(BASE_DIR, "outputs", "run_20260310_201633", "trajectories.csv")

OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "run_20260310_201633", "visualization_rich.mp4")

TRAIL_LENGTH = 30

# =======================
# SHOT CLASSES & COLORS
# =======================

SHOT_COLORS = {
    "none": (100, 100, 100),
    "serve": (0, 180, 255),
    "forehand": (0, 200, 0),
    "backhand": (255, 50, 50),
}

SHOT_CLASSES = ["none", "serve", "forehand", "backhand"]

PLAYER_COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0)
}

TIMELINE_H = 16
LABEL_BAR_H = 40

# =======================
# LOAD ALL DATA
# =======================

print("Loading data...")

# --- Predictions ---
pred_df = pd.read_csv(PREDICTIONS_CSV)

frame_shots = {}
player_shots = {}

for frame_id, group in pred_df.groupby("frame_id"):

    shots = group.set_index("player_id")["predicted_shot"].to_dict()
    player_shots.update({(frame_id, pid): shot for pid, shot in shots.items()})

    active = "none"

    for pid in [1, 2]:
        if pid in shots and shots[pid] != "none":
            active = shots[pid]
            break

    frame_shots[frame_id] = active

# --- Detections ---
det_df = pd.read_csv(DETECTIONS_CSV)

detections = {}

for _, r in det_df.iterrows():

    frame = int(r["frame"])

    p1 = None
    if pd.notna(r["p1_x1"]):
        p1 = (int(r["p1_x1"]), int(r["p1_y1"]), int(r["p1_x2"]), int(r["p1_y2"]))

    p2 = None
    if pd.notna(r["p2_x1"]):
        p2 = (int(r["p2_x1"]), int(r["p2_y1"]), int(r["p2_x2"]), int(r["p2_y2"]))

    ball = None
    if pd.notna(r["ball_x1"]):
        bx = int((r["ball_x1"] + r["ball_x2"]) / 2)
        by = int((r["ball_y1"] + r["ball_y2"]) / 2)
        ball = (bx, by)

    detections[frame] = {"p1": p1, "p2": p2, "ball": ball}

# --- Trajectories ---
traj_df = pd.read_csv(TRAJECTORIES_CSV)

trajectories = defaultdict(dict)

for _, r in traj_df.iterrows():

    frame = int(r["frame"])
    pid = int(r["player_id"])

    if pd.notna(r["cx"]) and pd.notna(r["cy"]):
        trajectories[frame][pid] = (int(r["cx"]), int(r["cy"]))

# =======================
# VIDEO
# =======================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

OUT_H = height + TIMELINE_H + LABEL_BAR_H

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, OUT_H))

# =======================
# TIMELINE BAR
# =======================

print("Pre-rendering timeline bar...")

timeline_bar = np.zeros((TIMELINE_H, width, 3), dtype=np.uint8)

for i in range(width):

    frame_idx = int(i / width * total_frames)
    shot = frame_shots.get(frame_idx, "none")
    color = SHOT_COLORS.get(shot, (100, 100, 100))

    timeline_bar[:, i] = color

# =======================
# LABEL BAR
# =======================

def make_label_bar(active_shot, width):

    bar = np.zeros((LABEL_BAR_H, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    n = len(SHOT_CLASSES)
    slot_w = width // n

    for i, shot in enumerate(SHOT_CLASSES):

        x1 = i * slot_w
        x2 = x1 + slot_w - 4

        color = SHOT_COLORS[shot]

        bar[:, x1:x2] = tuple(c // 3 for c in color)

        if shot == active_shot:
            cv2.rectangle(bar, (x1, 1), (x2, LABEL_BAR_H - 2), (0, 255, 0), 2)

        text = shot.upper()

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55

        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)

        tx = x1 + (slot_w - tw) // 2
        ty = LABEL_BAR_H // 2 + th // 2

        cv2.putText(bar, text, (tx, ty), font, scale, (255, 255, 255), 1)

    return bar

# =======================
# TRAIL BUFFERS
# =======================

trails = {
    1: deque(maxlen=TRAIL_LENGTH),
    2: deque(maxlen=TRAIL_LENGTH)
}

# =======================
# DRAW HELPERS
# =======================

def draw_trail(frame, trail, color):

    pts = list(trail)

    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(ch * alpha) for ch in color)

        cv2.line(frame, pts[i-1], pts[i], c, 2)


def draw_player_box(frame, box, pid, shot):

    x1, y1, x2, y2 = box
    color = PLAYER_COLORS[pid]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"P{pid}"

    if shot and shot != "none":
        label += f" | {shot.upper()}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def draw_ball(frame, pos):

    cv2.circle(frame, pos, 6, (0, 0, 255), -1)
    cv2.circle(frame, pos, 6, (255, 255, 255), 1)

# =======================
# MAIN LOOP
# =======================

print("Rendering output video...")

frame_id = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    active_shot = frame_shots.get(frame_id, "none")

    det = detections.get(frame_id, {})

    for pid in [1, 2]:

        if pid in trajectories.get(frame_id, {}):
            trails[pid].append(trajectories[frame_id][pid])

    for pid in [1, 2]:

        if trails[pid]:
            draw_trail(frame, trails[pid], PLAYER_COLORS[pid])

    if det.get("ball"):
        draw_ball(frame, det["ball"])

    for pid, key in [(1, "p1"), (2, "p2")]:

        if det.get(key):

            shot = player_shots.get((frame_id, pid), "none")

            draw_player_box(frame, det[key], pid, shot)

    tl = timeline_bar.copy()

    playhead_x = int(frame_id / total_frames * width)

    cv2.line(tl, (playhead_x, 0), (playhead_x, TIMELINE_H), (255, 255, 255), 2)

    lb = make_label_bar(active_shot, width)

    combined = np.vstack([frame, tl, lb])

    out.write(combined)

    if frame_id % 500 == 0:
        print(f"Processed {frame_id}/{total_frames}")

cap.release()
out.release()

print("Done! Saved to:", OUTPUT_PATH)
