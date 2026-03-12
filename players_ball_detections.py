import cv2
import os
import csv
from ultralytics import YOLO
from datetime import datetime

# =======================
# PATHS
# =======================

VIDEO_PATH = "input/V007.mp4"


PLAYER_MODEL_PATH = "runs/detect/train9/weights/best.pt"
BALL_MODEL_PATH = "runs/detect/train5/weights/best.pt"

BASE_OUTPUT_DIR = "outputs"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# =======================
# CREATE NEW RUN FOLDER
# =======================

run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, run_name)

FRAMES_DIR = os.path.join(RUN_DIR, "frames")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(RUN_DIR, "predict.mp4")
CSV_PATH = os.path.join(RUN_DIR, "detections.csv")

print("Saving results to:", RUN_DIR)

# =======================
# COURT FILTERING
# =======================

def filter_players_by_court(player_boxes, frame_w, frame_h):
    """
    Filter out detections clearly outside the court.
    Margins kept deliberately loose to avoid losing real player detections.
    TOP_MARGIN at 0.15 cuts net judge but still catches far baseline player.
    """

    LEFT_MARGIN   = 0.05 * frame_w
    RIGHT_MARGIN  = 0.95 * frame_w
    TOP_MARGIN    = 0.10 * frame_h
    BOTTOM_MARGIN = 0.95 * frame_h

    filtered = []

    for box in player_boxes:

        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if cx < LEFT_MARGIN:
            continue
        if cx > RIGHT_MARGIN:
            continue
        if cy < TOP_MARGIN:
            continue
        if cy > BOTTOM_MARGIN:
            continue

        filtered.append(box)

    return filtered


def keep_two_largest(boxes):
    """Keep only the two largest bounding boxes by area."""

    boxes = sorted(
        boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        reverse=True
    )

    return boxes[:2]


def assign_player_ids(player_boxes, net_box, frame_h):
    """
    Assign Player1 (bottom) and Player2 (top) using net Y as reference.

    If net is detected:
        - Players with cy > net_cy  → Player1 (near side / bottom)
        - Players with cy < net_cy  → Player2 (far side / top)

    If net is NOT detected:
        - Fall back to frame midpoint as reference

    Returns p1, p2 as boxes or None.
    """

    if len(player_boxes) == 0:
        return None, None

    # Determine reference Y
    if net_box is not None:
        net_x1, net_y1, net_x2, net_y2 = net_box
        ref_y = (net_y1 + net_y2) / 2
    else:
        # fallback to frame midpoint
        ref_y = frame_h / 2

    below = []  # Player1 candidates (bottom / near side)
    above = []  # Player2 candidates (top / far side)

    for box in player_boxes:
        x1, y1, x2, y2 = box
        cy = (y1 + y2) / 2

        if cy > ref_y:
            below.append(box)
        else:
            above.append(box)

    # Pick largest box from each side
    def largest(boxes):
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    p1 = largest(below)   # bottom player
    p2 = largest(above)   # top player

    return p1, p2


# =======================
# LOAD MODELS
# =======================

player_model = YOLO(PLAYER_MODEL_PATH)
ball_model   = YOLO(BALL_MODEL_PATH)

# =======================
# VIDEO READ
# =======================

cap = cv2.VideoCapture(VIDEO_PATH)

fps    = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =======================
# VIDEO WRITER
# =======================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (width, height)
)

# =======================
# CSV FILE
# =======================

csv_file = open(CSV_PATH, "w", newline="")
writer   = csv.writer(csv_file)

writer.writerow([
    "frame",
    "p1_x1", "p1_y1", "p1_x2", "p1_y2",
    "p2_x1", "p2_y1", "p2_x2", "p2_y2",
    "ball_x1", "ball_y1", "ball_x2", "ball_y2"
])

frame_id = 0

# =======================
# MAIN LOOP
# =======================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    fh, fw = frame.shape[:2]

    # save frame
    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_id:05d}.jpg")
    cv2.imwrite(frame_path, frame)

    # ---------------- PLAYER + NET DETECTION ----------------

    player_results = player_model(
        frame,
        conf=0.4,
        classes=[0, 1, 2],   # detect net(0), player1(1), player2(2)
        verbose=False
    )

    player_boxes = []
    net_box = None

    if player_results[0].boxes is not None:

        for box in player_results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            if cls == 0:
                # net — store it as reference, don't add to player boxes
                net_box = [x1, y1, x2, y2]
            else:
                # player1 or player2 — treat as generic player
                player_boxes.append([x1, y1, x2, y2])

    # Filter court and keep top 2
    player_boxes = filter_players_by_court(player_boxes, fw, fh)
    player_boxes = keep_two_largest(player_boxes)

    # Assign Player1/Player2 using net as reference (fallback: frame midpoint)
    p1, p2 = assign_player_ids(player_boxes, net_box, fh)

    # Draw net box if detected (useful for debugging)
    if net_box is not None:
        nx1, ny1, nx2, ny2 = net_box
        cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 255), 1)
        cv2.putText(
            frame, "Net",
            (nx1, ny1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 255), 1
        )

    p1_row = p1 if p1 is not None else [None, None, None, None]
    p2_row = p2 if p2 is not None else [None, None, None, None]

    # Draw players
    for label, box, color in [
        ("Player1", p1, (0, 255, 0)),
        ("Player2", p2, (255, 0, 0))
    ]:
        if box is None:
            continue

        x1, y1, x2, y2 = box

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ---------------- BALL DETECTION ----------------

    ball = [None, None, None, None]

    ball_results = ball_model(
        frame,
        conf=0.3,
        verbose=False
    )

    if ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:

        b = ball_results[0].boxes[0]

        x1, y1, x2, y2 = map(int, b.xyxy[0])

        ball = [x1, y1, x2, y2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            frame,
            "Ball",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Save to CSV
    writer.writerow([
        frame_id,
        *p1_row,
        *p2_row,
        *ball
    ])

    out.write(frame)

    cv2.imshow("Players + Ball Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =======================
# CLEANUP
# =======================

cap.release()
out.release()
csv_file.close()

cv2.destroyAllWindows()

print("Output saved to:", RUN_DIR)