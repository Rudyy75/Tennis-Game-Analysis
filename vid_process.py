import cv2
import numpy as np
import tensorflow as tf
import os
import argparse
import csv
import time

# CONFIGURATION 
MODEL_PATH = 'models/tennis_score.keras'
CLASS_NAMES_PATH = 'models/class_names.txt'

# Hardcoded ROI coordinates (tuned with roi_tuner.py)
# 1. When scoreboard is SHRUNK (normal gameplay)
ROIS_SHRUNK = {
    'p1_points': (388, 38, 30, 27),
    'p2_points': (388, 70, 30, 27),
    'p1_games': (351, 38, 30, 27),
    'p2_games': (351, 70, 30, 27),
    'p1_sets': (322, 38, 30, 27),
    'p2_sets': (322, 70, 30, 27),
}

# 2. When scoreboard is EXPANDED (showing full player names, usually first few minutes)
# The user noted that expanded is simply shrunk + 163 on the X axis
ROIS_EXPANDED = {
    k: (x + 132, y, w, h) for k, (x, y, w, h) in ROIS_SHRUNK.items()
}

# HSV range for text detection (for change detection)
LOWER_BLUE = np.array([90, 50, 20])
UPPER_BLUE = np.array([130, 255, 150])

# Change detection threshold
CHANGE_THRESHOLD = 5.0

# Minimum confidence to accept a prediction
CONFIDENCE_THRESHOLD = 0.7

# Visibility Checks
SCOREBOARD_CHECK_ROI = (340, 34, 40, 64)  # Checks if scoreboard exists AT ALL (filters crowd shots)
EXPANDED_CHECK_ROI = (450, 34, 20, 20)    # Checks if scoreboard extends far to the right (Expanded)
SCOREBOARD_BLUE_RATIO = 0.3

# TEMPORAL VALIDATION 
# Valid tennis point progressions (forward only, no skips)
VALID_POINT_TRANSITIONS = {
    '0':  {'15'},
    '15': {'30'},
    '30': {'40'},
    '40': {'AD'},
    'AD': set(),  # AD only resets on game win (handled separately)
}

STABILITY_FRAMES = 1

POINTS_CONF_THRESHOLD = 0.70
GAMES_SETS_CONF_THRESHOLD = 0.70


def is_gameplay_frame(frame, grass_threshold=0.48, min_lines=7):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([30, 89, 101])
    upper_bound = np.array([75, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    grass_ratio = np.sum(mask > 0) / mask.size
    
    if grass_ratio < grass_threshold:
        return False
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(white_mask, 80, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=86, minLineLength=102, maxLineGap=15)
    
    if lines is None:
        return False
    
    horizontal_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 20 or angle > 160:
            horizontal_count += 1
    
    return horizontal_count >= min_lines


def get_scoreboard_state(frame: np.ndarray) -> str:
    # 1. Does the scoreboard exist at all?
    x, y, w, h = SCOREBOARD_CHECK_ROI
    if y+h > frame.shape[0] or x+w > frame.shape[1]:
        return 'NONE'
    
    check_region = frame[y:y+h, x:x+w]
    hsv = cv2.cvtColor(check_region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    
    if np.count_nonzero(mask) / mask.size < SCOREBOARD_BLUE_RATIO:
        return 'NONE'
    
    # 2. Is it expanded? (Check if the blue background extends far right)
    ex, ey, ew, eh = EXPANDED_CHECK_ROI
    if ey+eh > frame.shape[0] or ex+ew > frame.shape[1]:
        return 'SHRUNK'
        
    exp_region = frame[ey:ey+eh, ex:ex+ew]
    exp_hsv = cv2.cvtColor(exp_region, cv2.COLOR_BGR2HSV)
    exp_mask = cv2.inRange(exp_hsv, LOWER_BLUE, UPPER_BLUE)
    
    if np.count_nonzero(exp_mask) / exp_mask.size > 0.5:
        return 'EXPANDED'
        
    return 'SHRUNK'

import easyocr

from collections import deque

def process_video(video_path, output_csv='score_output.csv', output_video=None, preview=False, start_frame=1):
    
    # 1. Load Model
    print("Loading EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True) 
    print("EasyOCR loaded.")
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Seek to start frame if requested
    if start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        frame_count = start_frame - 1
        print(f"Starting from frame {start_frame}/{total_frames} ({((start_frame-1)/fps):.1f}s into video)")
    else:
        frame_count = 0
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # 3. Setup Output
    out_writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    csv_data = []
    
    # 4. Processing State
    prev_crops = {}
    
    current_scores = {
        'p1_points': '0', 'p2_points': '0',
        'p1_games': '0', 'p2_games': '0',
        'p1_sets': '0', 'p2_sets': '0'
    }
    current_confidences = {
        'p1_points': 1.0, 'p2_points': 1.0,
        'p1_games': 1.0, 'p2_games': 1.0,
        'p1_sets': 1.0, 'p2_sets': 1.0
    }
    pending_scores = {}
    pending_frames = {}
    state_history = deque(maxlen=15)
    last_stable_state = 'SHRUNK'
    
    frame_count = 0
    score_changes = 0
    
    print("\nProcessing ")
    if preview:
        print("PREVIEW MODE: Press 'q' to quit")
    
    start_time = time.time()
    inference_count = 0
    inference_time_total = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp_ms = int((frame_count / fps) * 1000)
        
        old_scores = current_scores.copy()
        scores_this_frame = current_scores.copy()
        game_winner = ''
        
        # SCOREBOARD STATE CHECK 
        sb_raw = get_scoreboard_state(frame)
        state_history.append(sb_raw)

        none_count = state_history.count('NONE')
        expanded_count = state_history.count('EXPANDED')
        shrunk_count = state_history.count('SHRUNK')

        # Require 12+ frames of consistent state before switching (prevents random flickers)
        if none_count >= 12:
            sb_state = 'NONE'
        elif expanded_count >= 12:
            sb_state = 'EXPANDED'
        elif shrunk_count >= 12:
            sb_state = 'SHRUNK'
        else:
            sb_state = last_stable_state

        if sb_state == 'NONE':
            last_stable_state = 'NONE'
            # Cutscene/replay/crowd shot - skip inference, keep previous scores
            csv_data.append({
                'frame_id': frame_count,
                'timestamp_ms': timestamp_ms,
                'p1_points': scores_this_frame['p1_points'],
                'p1_points_conf': round(current_confidences['p1_points'], 3),
                'p2_points': scores_this_frame['p2_points'],
                'p2_points_conf': round(current_confidences['p2_points'], 3),
                'p1_games': scores_this_frame['p1_games'],
                'p1_games_conf': round(current_confidences['p1_games'], 3),
                'p2_games': scores_this_frame['p2_games'],
                'p2_games_conf': round(current_confidences['p2_games'], 3),
                'p1_sets': scores_this_frame['p1_sets'],
                'p1_sets_conf': round(current_confidences['p1_sets'], 3),
                'p2_sets': scores_this_frame['p2_sets'],
                'p2_sets_conf': round(current_confidences['p2_sets'], 3),
            })
            
            if output_video or preview:
                cv2.putText(frame, "[NO SCOREBOARD]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"P1: {current_scores['p1_points']}  P2: {current_scores['p2_points']}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            if out_writer:
                out_writer.write(frame)
            if preview:
                cv2.imshow('Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 500 == 0:
                print(f"Processed {frame_count}/{total_frames} frames... ({score_changes} score changes) [NO SB]")
            continue
            
        # Select active ROIs based on state
        active_rois = ROIS_EXPANDED if sb_state == 'EXPANDED' else ROIS_SHRUNK
        if sb_state != 'NONE':
            last_stable_state = sb_state
        
        # PER-ROI: Change detection + CNN prediction 
        proposed_scores = {}  # Collect proposals before validating
        
        for label, roi in active_rois.items():
            x, y, w, h = roi
            
            if y+h > frame.shape[0] or x+w > frame.shape[1] or w == 0 or h == 0:
                continue
            
            crop = frame[y:y+h, x:x+w]
            
            # CHANGE DETECTION 
            should_predict = False
            if label not in prev_crops:
                should_predict = True
            else:
                diff = cv2.absdiff(crop, prev_crops[label])
                if np.mean(diff) > CHANGE_THRESHOLD:
                    should_predict = True
            
            # OCR INFERENCE 
            if should_predict:
                t0 = time.time()
                
                # Games/Sets are single digits (0-7) on a small crop.
                if 'games' in label or 'sets' in label:
                    img_ocr = cv2.resize(crop, (crop.shape[1]*5, crop.shape[0]*5),
                                         interpolation=cv2.INTER_CUBIC)
                    allowlist = '01234567'
                else:
                    # Points: standard 3x color resize
                    img_ocr = cv2.resize(crop, (crop.shape[1]*3, crop.shape[0]*3),
                                         interpolation=cv2.INTER_CUBIC)
                    allowlist = '01234567ADoO'
                
                result = reader.readtext(img_ocr, allowlist=allowlist)
                inference_time_total += time.time() - t0
                inference_count += 1
                
                if result:
                    text = result[0][1].upper()
                    confidence = result[0][2]
                    
                    # Clean up common OCR mistakes
                    if text in ['O', 'Q', '00']:
                        text = '0'
                    elif text in ['A', 'D']:
                        text = 'AD'
                        
                    raw_score = text
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if 'points' in label and raw_score in ['0', '15', '30', '40', 'AD']:
                            proposed_scores[label] = (raw_score, confidence)
                        elif 'games' in label and raw_score in ['0', '1', '2', '3', '4', '5', '6', '7']:
                            proposed_scores[label] = (raw_score, confidence)
                        elif 'sets' in label and raw_score in ['0', '1', '2', '3']:
                            proposed_scores[label] = (raw_score, confidence)
                    else:
                        proposed_scores[label] = ("?", 0.0)
                else:
                    proposed_scores[label] = ("?", 0.0)
            
            prev_crops[label] = crop.copy()
        
        # UPDATE ALL SCORES
        games_committed = []
        for label in active_rois.keys():
            if label not in proposed_scores:
                scores_this_frame[label] = current_scores[label]
                if label in pending_frames:
                    pending_frames[label] += 1
                    if pending_frames.get(label, 0) >= STABILITY_FRAMES and pending_scores.get(label) and pending_scores[label] != current_scores[label]:
                        print(f"Frame {frame_count}: {label} {current_scores[label]} -> {pending_scores[label]} (stable after {pending_frames[label]} frames)")
                        score_changes += 1
                        if 'games' in label:
                            try:
                                if int(pending_scores[label]) > int(current_scores[label]):
                                    games_committed.append(label)
                            except ValueError:
                                pass
                        current_scores[label] = pending_scores[label]
                        current_confidences[label] = pending_frames.get('_conf', 1.0)
                        pending_scores.pop(label, None)
                        pending_frames.pop(label, None)
                continue

            raw_score, confidence = proposed_scores[label]
            old_score = current_scores[label]

            if raw_score == '?' or raw_score == old_score:
                pending_scores.pop(label, None)
                pending_frames.pop(label, None)
                scores_this_frame[label] = current_scores[label]
                continue

            conf_threshold = POINTS_CONF_THRESHOLD if 'points' in label else GAMES_SETS_CONF_THRESHOLD
            if confidence < conf_threshold:
                pending_scores.pop(label, None)
                pending_frames.pop(label, None)
                scores_this_frame[label] = current_scores[label]
                continue

            if 'games' in label:
                try:
                    old_g = int(old_score)
                    new_g = int(raw_score)
                    if new_g == old_g + 1 or new_g == old_g - 1 or new_g == 0 or new_g != old_g:
                        pass
                    else:
                        pending_scores.pop(label, None)
                        pending_frames.pop(label, None)
                        scores_this_frame[label] = current_scores[label]
                        continue
                except ValueError:
                    pending_scores.pop(label, None)
                    pending_frames.pop(label, None)
                    scores_this_frame[label] = current_scores[label]
                    continue

            if 'sets' in label:
                try:
                    old_s = int(old_score)
                    new_s = int(raw_score)
                    if new_s != old_s + 1 and not (new_s == 0 and old_s > 0):
                        pending_scores.pop(label, None)
                        pending_frames.pop(label, None)
                        scores_this_frame[label] = current_scores[label]
                        continue
                except ValueError:
                    pending_scores.pop(label, None)
                    pending_frames.pop(label, None)
                    scores_this_frame[label] = current_scores[label]
                    continue

            if STABILITY_FRAMES <= 1:
                print(f"Frame {frame_count}: {label} {old_score} -> {raw_score} (conf: {confidence:.2f})")
                score_changes += 1
                if 'games' in label:
                    try:
                        if int(raw_score) > int(old_score):
                            games_committed.append(label)
                    except ValueError:
                        pass
                current_scores[label] = raw_score
                current_confidences[label] = confidence
            elif pending_scores.get(label) == raw_score:
                pending_frames[label] += 1
                if pending_frames[label] >= STABILITY_FRAMES:
                    print(f"Frame {frame_count}: {label} {old_score} -> {raw_score} (conf: {confidence:.2f}, stable)")
                    score_changes += 1
                    if 'games' in label:
                        try:
                            if int(raw_score) > int(old_score):
                                games_committed.append(label)
                        except ValueError:
                            pass
                    current_scores[label] = raw_score
                    current_confidences[label] = confidence
                    pending_scores.pop(label)
                    pending_frames.pop(label)
            else:
                pending_scores[label] = raw_score
                pending_frames[label] = 1
                pending_frames['_conf'] = confidence

            scores_this_frame[label] = current_scores[label]

        if games_committed:
            for p in ['p1_points', 'p2_points']:
                if current_scores[p] != '0':
                    print(f"Frame {frame_count}: {p} {current_scores[p]} -> 0 (game ended)")
                    current_scores[p] = '0'
                    current_confidences[p] = 1.0
                    pending_scores.pop(p, None)
                    pending_frames.pop(p, None)
        else:
            p1_pts = current_scores.get('p1_points', '0')
            p2_pts = current_scores.get('p2_points', '0')
            if p1_pts == '0' and p2_pts == '0':
                p1_was = old_scores.get('p1_points', '0')
                p2_was = old_scores.get('p2_points', '0')
                if p1_was in ('40', 'AD') and p2_was not in ('40', 'AD'):
                    g = int(current_scores['p1_games'])
                    current_scores['p1_games'] = str(g + 1)
                    print(f"Frame {frame_count}: p1_games {g} -> {g+1} (inferred from game point)")
                    score_changes += 1
                elif p2_was in ('40', 'AD') and p1_was not in ('40', 'AD'):
                    g = int(current_scores['p2_games'])
                    current_scores['p2_games'] = str(g + 1)
                    print(f"Frame {frame_count}: p2_games {g} -> {g+1} (inferred from game point)")
                    score_changes += 1

        # PREVIEW (draw ROI boxes and text for debugging) 
        if output_video or preview:
            for label, roi in active_rois.items():
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show raw prediction if available this frame
                if label in proposed_scores:
                    raw_p, conf_p = proposed_scores[label]
                    text = f"{raw_p} ({conf_p:.2f})"
                    color = (0, 255, 0) if conf_p >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.putText(frame, text, (x + w + 5, y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show validated score on the top left
            cv2.putText(frame, f"S{current_scores['p1_sets']} G{current_scores['p1_games']} P1: {current_scores['p1_points']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"S{current_scores['p2_sets']} G{current_scores['p2_games']} P2: {current_scores['p2_points']}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Save to CSV
        csv_data.append({
            'frame_id': frame_count,
            'timestamp_ms': timestamp_ms,
            'p1_points': scores_this_frame['p1_points'],
            'p1_points_conf': round(current_confidences['p1_points'], 3),
            'p2_points': scores_this_frame['p2_points'],
            'p2_points_conf': round(current_confidences['p2_points'], 3),
            'p1_games': scores_this_frame['p1_games'],
            'p1_games_conf': round(current_confidences['p1_games'], 3),
            'p2_games': scores_this_frame['p2_games'],
            'p2_games_conf': round(current_confidences['p2_games'], 3),
            'p1_sets': scores_this_frame['p1_sets'],
            'p1_sets_conf': round(current_confidences['p1_sets'], 3),
            'p2_sets': scores_this_frame['p2_sets'],
            'p2_sets_conf': round(current_confidences['p2_sets'], 3),
        })
        
        if out_writer:
            out_writer.write(frame)
        
        if preview:
            cv2.imshow('Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 500 == 0:
            print(f"Processed {frame_count}/{total_frames} frames... ({score_changes} score changes)")
    
    # 5. Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    if preview:
        cv2.destroyAllWindows()
    
    # 6. Write CSV
    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            'frame_id', 'timestamp_ms', 
            'p1_points', 'p1_points_conf', 'p2_points', 'p2_points_conf',
            'p1_games', 'p1_games_conf', 'p2_games', 'p2_games_conf',
            'p1_sets', 'p1_sets_conf', 'p2_sets', 'p2_sets_conf'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nComplete")
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference_ms = (inference_time_total / inference_count * 1000) if inference_count > 0 else 0
    
    print(f"Processed {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f"CNN called {inference_count} times (avg {avg_inference_ms:.1f}ms per inference)")
    print(f"Detected {score_changes} scoreboard updates")
    print(f"Saved scores to '{output_csv}'")
    if output_video:
        print(f"Saved masked video to '{output_video}'")


if __name__ == "__main__":
    default_video = 'input_clip.mp4'
    if not os.path.exists(default_video) and os.path.exists('../input_clip.mp4'):
        default_video = '../input_clip.mp4'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=default_video, help='Input video path')
    parser.add_argument('--output-csv', default='score_output.csv', help='Output CSV path')
    parser.add_argument('--output-video', default=None, help='Output masked video path')
    parser.add_argument('--preview', action='store_true', help='Show live preview')
    parser.add_argument('--start-frame', type=int, default=1, help='Frame number to start processing from')
    args = parser.parse_args()
    
    process_video(args.video, args.output_csv, args.output_video, args.preview, args.start_frame)
