import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

BASE_OUTPUT_DIR = "outputs"

runs = sorted(os.listdir(BASE_OUTPUT_DIR))
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, runs[-1])

print("Using run folder:", RUN_DIR)

FRAMES_DIR = os.path.join(RUN_DIR, "frames")
TRACKED_PLAYERS = os.path.join(RUN_DIR, "tracked_players.csv")

POSE_MODEL_PATH = "yolov8n-pose.pt"

BOX_EXPANSION_RATIO = 0.15
POSE_INPUT_SIZE = 640
KEYPOINT_CONF_THRESH = 0.4

POSE_DIR = os.path.join(RUN_DIR, "pose")

CROPS_DIR = os.path.join(POSE_DIR, "crops")
POSE_INPUT_DIR = os.path.join(POSE_DIR, "pose_inputs")
SKELETON_DIR = os.path.join(POSE_DIR, "skeletal_images")
CSV_DIR = os.path.join(POSE_DIR, "pose_outputs_csv")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(POSE_INPUT_DIR, exist_ok=True)
os.makedirs(SKELETON_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

CSV_PATH = os.path.join(CSV_DIR, "pose_keypoints.csv")

def expand_bbox(bbox, frame_w, frame_h, ratio):

    x1,y1,x2,y2 = bbox

    w = x2-x1
    h = y2-y1

    dx = ratio*w
    dy = ratio*h

    return (
        max(0,int(x1-dx)),
        max(0,int(y1-dy)),
        min(frame_w,int(x2+dx)),
        min(frame_h,int(y2+dy))
    )

def resize_with_padding(img,target=640):

    h,w = img.shape[:2]

    scale = min(target/w,target/h)

    nw = int(w*scale)
    nh = int(h*scale)

    resized = cv2.resize(img,(nw,nh))

    padded = np.zeros((target,target,3),dtype=np.uint8)

    pad_x = (target-nw)//2
    pad_y = (target-nh)//2

    padded[pad_y:pad_y+nh,pad_x:pad_x+nw] = resized

    return padded,scale,pad_x,pad_y

frame_boxes = {}

with open(TRACKED_PLAYERS,"r") as f:

    reader = csv.DictReader(f)

    for r in reader:

        frame = int(r["frame"])
        pid = int(r["player_id"])

        box = [
            float(r["x1"]),
            float(r["y1"]),
            float(r["x2"]),
            float(r["y2"])
        ]

        if frame not in frame_boxes:
            frame_boxes[frame] = {}

        frame_boxes[frame][pid] = box

processed_frames = set()

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as existing_f:
        existing_reader = csv.DictReader(existing_f)
        for row in existing_reader:
            processed_frames.add(int(row["frame"]))

print(f"Skipping {len(processed_frames)} already processed frames...")

pose_model = YOLO(POSE_MODEL_PATH)

file_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0

csv_file = open(CSV_PATH, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:

    header = ["frame","player_id"]

    for i in range(17):

        header += [
            f"kp{i}_x",
            f"kp{i}_y",
            f"kp{i}_conf"
        ]

    writer.writerow(header)

frame_files = sorted(os.listdir(FRAMES_DIR))

for frame_file in frame_files:

    frame_id = int(frame_file.split("_")[1].split(".")[0])

    if frame_id in processed_frames:
        continue

    frame_path = os.path.join(FRAMES_DIR,frame_file)

    frame = cv2.imread(frame_path)

    fh,fw = frame.shape[:2]

    if frame_id not in frame_boxes:
        continue

    for pid,box in frame_boxes[frame_id].items():

        x1,y1,x2,y2 = expand_bbox(box,fw,fh,BOX_EXPANSION_RATIO)

        crop = frame[int(y1):int(y2),int(x1):int(x2)]

        if crop.size == 0:
            continue

        crop_path = os.path.join(
            CROPS_DIR,
            f"frame_{frame_id:05d}_player_{pid}.jpg"
        )

        cv2.imwrite(crop_path,crop)

        pose_input,scale,pad_x,pad_y = resize_with_padding(crop)

        pose_input_path = os.path.join(
            POSE_INPUT_DIR,
            f"frame_{frame_id:05d}_player_{pid}.jpg"
        )

        cv2.imwrite(pose_input_path,pose_input)

        result = pose_model(pose_input,verbose=False)[0]

        if (
            result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0
        ):
            continue

        kps = result.keypoints.xy[0]
        confs = result.keypoints.conf[0]

        skeleton_img = result.plot()

        skeleton_path = os.path.join(
            SKELETON_DIR,
            f"frame_{frame_id:05d}_player_{pid}.jpg"
        )

        cv2.imwrite(skeleton_path,skeleton_img)

        row = [frame_id,pid]

        for kp,conf in zip(kps,confs):

            conf = float(conf)

            if conf < KEYPOINT_CONF_THRESH:

                row += [-1,-1,float(conf)]
                continue

            x_unpad = kp[0]-pad_x
            y_unpad = kp[1]-pad_y

            x_crop = x_unpad/scale
            y_crop = y_unpad/scale

            x_frame = x_crop + x1
            y_frame = y_crop + y1

            row += [float(x_frame),float(y_frame),float(conf)]

        writer.writerow(row)

    print("Processed frame",frame_id)

csv_file.close()

print("Pose pipeline finished")

print("Crops →",CROPS_DIR)
print("Pose inputs →",POSE_INPUT_DIR)
print("Skeleton images →",SKELETON_DIR)
print("Pose CSV →",CSV_PATH)