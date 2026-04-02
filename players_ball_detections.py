import cv2
import os
import csv
from ultralytics import YOLO
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

VIDEO_PATH        = "input/V007.mp4"
PLAYER_MODEL_PATH = "runs/detect/train9/weights/best.pt"
BALL_MODEL_PATH   = "runs/detect/train5/weights/best.pt"

# Minimum player bounding-box height as a fraction of frame height.
# Rejects ball boys, distant crowd figures, umpire, etc.
MIN_PLAYER_HEIGHT_RATIO = 0.07   # 7% of frame height

# Consecutive-frame smoothing for net Y reference.
# Averages the last N detected net positions to avoid jitter.
NET_SMOOTH_WINDOW = 10

# ─────────────────────────────────────────────
#  OUTPUT DIRECTORY SETUP
# ─────────────────────────────────────────────

BASE_OUTPUT_DIR = "outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR  = os.path.join(BASE_OUTPUT_DIR, run_name)

# Only save frames that are confirmed gameplay frames → saves disk space
FRAMES_DIR = os.path.join(RUN_DIR, "frames")

os.makedirs(RUN_DIR,    exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(RUN_DIR, "predict.mp4")
CSV_PATH          = os.path.join(RUN_DIR, "detections.csv")

print("Saving results to:", RUN_DIR)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def filter_players_by_court(player_boxes, frame_w, frame_h):
    """Remove detections outside the court boundary margins."""
    LEFT_MARGIN   = 0.05 * frame_w
    RIGHT_MARGIN  = 0.95 * frame_w
    TOP_MARGIN    = 0.10 * frame_h
    BOTTOM_MARGIN = 0.95 * frame_h

    filtered = []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if LEFT_MARGIN <= cx <= RIGHT_MARGIN and TOP_MARGIN <= cy <= BOTTOM_MARGIN:
            filtered.append(box)
    return filtered


def keep_two_largest(boxes):
    """Keep only the two largest bounding boxes by area."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:2]


def assign_player_ids(player_boxes, ref_y):
    """
    Assign P1 (below net) and P2 (above net) using net Y as divider.
    ref_y  — smoothed net Y coordinate (or frame_h/2 as fallback).
    Returns (p1_box, p2_box) — either can be None.
    """
    below, above = [], []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        cy = (y1 + y2) / 2
        (below if cy > ref_y else above).append(box)

    def largest(lst):
        if not lst:
            return None
        return max(lst, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    return largest(below), largest(above)


def is_gameplay_frame(net_box, p1, p2, ref_y, frame_h):
    """
    A frame is a valid gameplay frame if ALL of these hold:
      1. Net was detected (or we have a stable smoothed net Y)
      2. Both P1 and P2 are detected
      3. P1 is strictly below net, P2 is strictly above net
         → guarantees full-court view with one player each side
      4. Both players are large enough (rejects ball boys / distant figures)

    NOTE: We check ref_y rather than net_box so that a brief
    net-detection miss doesn't drop an otherwise valid frame.
    """
    if p1 is None or p2 is None:
        return False

    # Players must be on opposite sides of net
    p1_cy = (p1[1] + p1[3]) / 2
    p2_cy = (p2[1] + p2[3]) / 2
    if not (p1_cy > ref_y and p2_cy < ref_y):
        return False

    # Both players must be a reasonable size
    min_h = MIN_PLAYER_HEIGHT_RATIO * frame_h
    if (p1[3] - p1[1]) < min_h or (p2[3] - p2[1]) < min_h:
        return False

    return True


# ─────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────

player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)


# ─────────────────────────────────────────────
#  VIDEO I/O
# ─────────────────────────────────────────────

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))


# ─────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────

csv_file = open(CSV_PATH, "w", newline="")
writer   = csv.writer(csv_file)
writer.writerow([
    "frame",
    "is_gameplay",
    "p1_x1", "p1_y1", "p1_x2", "p1_y2",
    "p2_x1", "p2_y1", "p2_x2", "p2_y2",
    "ball_x1", "ball_y1", "ball_x2", "ball_y2",
    "net_y"        # expose smoothed net Y for downstream scripts
])


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

frame_id         = 0
gameplay_count   = 0
net_y_history    = []           # rolling buffer for net Y smoothing
smoothed_net_y   = height / 2  # fallback until net is first seen

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    fh, fw = frame.shape[:2]

    # ── Player / net detection ──────────────────
    player_results = player_model(frame, conf=0.4, classes=[0, 1, 2], verbose=False)

    player_boxes = []
    net_box      = None

    if player_results[0].boxes is not None:
        for box in player_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls == 0:                         # class 0 = net
                net_box = [x1, y1, x2, y2]
            else:                                # class 1/2 = player
                player_boxes.append([x1, y1, x2, y2])

    # ── Update smoothed net Y ───────────────────
    if net_box is not None:
        raw_net_y = (net_box[1] + net_box[3]) / 2
        net_y_history.append(raw_net_y)
        if len(net_y_history) > NET_SMOOTH_WINDOW:
            net_y_history.pop(0)
        smoothed_net_y = sum(net_y_history) / len(net_y_history)

    # ── Filter & assign players ─────────────────
    player_boxes = filter_players_by_court(player_boxes, fw, fh)
    player_boxes = keep_two_largest(player_boxes)
    p1, p2       = assign_player_ids(player_boxes, smoothed_net_y)

    # ── Gameplay gate ───────────────────────────
    gameplay = is_gameplay_frame(net_box, p1, p2, smoothed_net_y, fh)

    # ── Ball detection (only on gameplay frames) ─
    ball = [None, None, None, None]

    if gameplay:
        ball_results = ball_model(frame, conf=0.3, verbose=False)
        if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            b            = ball_results[0].boxes[0]
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            ball         = [bx1, by1, bx2, by2]

    # ── Visualise ───────────────────────────────
    if net_box is not None:
        nx1, ny1, nx2, ny2 = net_box
        cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 255), 1)
        cv2.putText(frame, "Net", (nx1, ny1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw smoothed net line for debugging
    cv2.line(frame, (0, int(smoothed_net_y)), (fw, int(smoothed_net_y)),
             (0, 255, 255), 1, cv2.LINE_AA)

    for label, box, color in [("Player1", p1, (0, 255, 0)),
                               ("Player2", p2, (255, 0, 0))]:
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if ball[0] is not None:
        bx1, by1, bx2, by2 = ball
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(frame, "Ball", (bx1, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Gameplay indicator banner
    banner_color = (0, 180, 0) if gameplay else (0, 0, 180)
    banner_text  = "GAMEPLAY" if gameplay else "NON-GAMEPLAY"
    cv2.rectangle(frame, (0, 0), (200, 28), banner_color, -1)
    cv2.putText(frame, banner_text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Save frame image only if gameplay ──────
    if gameplay:
        gameplay_count += 1
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(frame_path, frame)

    # ── Write CSV row (ALL frames, gameplay flag tells downstream) ─
    p1_row   = p1   if p1   is not None else [None, None, None, None]
    p2_row   = p2   if p2   is not None else [None, None, None, None]
    ball_row = ball if ball[0] is not None else [None, None, None, None]

    writer.writerow([
        frame_id,
        int(gameplay),
        *p1_row,
        *p2_row,
        *ball_row,
        round(smoothed_net_y, 2)
    ])

    out.write(frame)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_id % 500 == 0:
        print(f"Frame {frame_id} | Gameplay so far: {gameplay_count}")

# ─────────────────────────────────────────────
#  CLEANUP
# ─────────────────────────────────────────────

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print(f"\nDone! Total frames: {frame_id} | Gameplay frames: {gameplay_count}")
print("Output saved to:", RUN_DIR)