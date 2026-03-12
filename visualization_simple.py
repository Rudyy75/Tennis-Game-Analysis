import cv2
import pandas as pd
import numpy as np

# =======================
# CONFIGURATION
# =======================

VIDEO_PATH      = "input/videoplayback1.mp4"
PREDICTIONS_CSV = "outputs/run_XXXXXXXX_XXXXXX/dataset_with_predictions.csv"  # teammate's CSV
OUTPUT_PATH     = "outputs/run_XXXXXXXX_XXXXXX/visualization_simple.mp4"

# =======================
# SHOT CLASSES & COLORS (BGR)
# =======================

SHOT_COLORS = {
    "none":      (100, 100, 100),   # gray
    "serve":     (  0, 180, 255),   # orange
    "forehand":  (  0, 200,   0),   # green
    "backhand":  (255,  50,  50),   # blue
}

SHOT_CLASSES = ["none", "serve", "forehand", "backhand"]

TIMELINE_H  = 16   # height of scrubber bar
LABEL_BAR_H = 40   # height of label buttons row
PADDING     = 8    # padding inside label buttons

# =======================
# LOAD PREDICTIONS
# =======================

print("Loading predictions...")

df = pd.read_csv(PREDICTIONS_CSV)

# Per frame → pick the active shot
# Priority: any non-none shot wins; if both none → none; if conflict → player1 wins
frame_shots = {}

for frame_id, group in df.groupby("frame_id"):

    shots = group.set_index("player_id")["predicted_shot"].to_dict()

    # Pick active shot
    active = "none"

    for pid in [1, 2]:
        if pid in shots and shots[pid] != "none":
            active = shots[pid]
            break

    frame_shots[frame_id] = active

# =======================
# VIDEO
# =======================

cap = cv2.VideoCapture(VIDEO_PATH)

fps      = cap.get(cv2.CAP_PROP_FPS)
width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

# Output height = video height + timeline strip
OUT_H = height + TIMELINE_H + LABEL_BAR_H

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, OUT_H))

# =======================
# PRE-RENDER TIMELINE BAR
# =======================

print("Pre-rendering timeline bar...")

timeline_bar = np.zeros((TIMELINE_H, width, 3), dtype=np.uint8)

for i in range(width):

    frame_idx = int(i / width * total_frames)
    shot = frame_shots.get(frame_idx, "none")
    color = SHOT_COLORS.get(shot, (100, 100, 100))
    timeline_bar[:, i] = color

# =======================
# LABEL BAR TEMPLATE
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

        # Fill background with dim color
        bar[:, x1:x2] = tuple(c // 3 for c in color)

        # Highlight active with bright border
        if shot == active_shot:
            cv2.rectangle(bar, (x1, 1), (x2, LABEL_BAR_H - 2), (0, 255, 0), 2)

        # Label text
        text = shot.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx = x1 + (slot_w - tw) // 2
        ty = LABEL_BAR_H // 2 + th // 2

        cv2.putText(bar, text, (tx, ty), font, scale, (255, 255, 255), thickness)

    return bar


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

    # Build timeline with playhead
    tl = timeline_bar.copy()
    playhead_x = int(frame_id / total_frames * width)
    cv2.line(tl, (playhead_x, 0), (playhead_x, TIMELINE_H), (255, 255, 255), 2)

    # Build label bar
    lb = make_label_bar(active_shot, width)

    # Stack: video + timeline + labels
    combined = np.vstack([frame, tl, lb])

    out.write(combined)

    if frame_id % 500 == 0:
        print(f"  Processed {frame_id}/{total_frames} frames")

cap.release()
out.release()

print("Done! Saved to:", OUTPUT_PATH)
