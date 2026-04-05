import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Interactive ROI Selector')
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    args = parser.parse_args()

    if args.video:
        video_path = args.video
    else:
        video_path = 'input_clip.mp4'
        if not os.path.exists(video_path) and os.path.exists('../input_clip.mp4'):
            video_path = '../input_clip.mp4'

    if not os.path.exists(video_path):
        print(f"Error: Cannot find video file: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
        
    h_frame, w_frame = frame.shape[:2]
    print(f"Video: {video_path} ({w_frame}x{h_frame})")

    # Create window and trackbars
    cv2.namedWindow('ROI Tuner', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('X', 'ROI Tuner', 0, w_frame - 1, lambda x: None)
    cv2.createTrackbar('Y', 'ROI Tuner', 0, h_frame - 1, lambda x: None)
    cv2.createTrackbar('W', 'ROI Tuner', 50, w_frame, lambda x: None)
    cv2.createTrackbar('H', 'ROI Tuner', 50, h_frame, lambda x: None)
    
    print("Use the trackbars to draw a box around the digits.")
    print("Press the corresponding number key to LOCK the box in:")
    print("  [1] - P1 Points (15, 30, 40)")
    print("  [2] - P2 Points (15, 30, 40)")
    print("  [3] - P1 Games  (0-7)")
    print("  [4] - P2 Games  (0-7)")
    print("  [5] - P1 Sets   (0-3)")
    print("  [6] - P2 Sets   (0-3)")
    print("")
    print("Other Controls:")
    print("  SPACE : Play / Pause")
    print("  'n'   : Next frame")
    print("  'p'   : Previous frame")
    print("  's'   : Print Python Dictionary code")
    print("  'q'   : Quit")

    is_playing = True
    delay = 30  # ms delay between frames when playing
    active_rois = {}

    while True:
        if is_playing:
            ret, next_frame = cap.read()
            if not ret:
                print("End of video reached. Pausing.")
                is_playing = False
                # Step back one frame so we have a valid image
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                ret, frame = cap.read()
            else:
                frame = next_frame

        clone = frame.copy()
        x = cv2.getTrackbarPos('X', 'ROI Tuner')
        y = cv2.getTrackbarPos('Y', 'ROI Tuner')
        w = cv2.getTrackbarPos('W', 'ROI Tuner')
        h = cv2.getTrackbarPos('H', 'ROI Tuner')

        # Draw Active Sliders Box (Blue)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw Locked Boxes (Green)
        for label, (lx, ly, lw, lh) in active_rois.items():
            cv2.rectangle(clone, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
            cv2.putText(clone, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add status text
        status = "PLAYING" if is_playing else "PAUSED (Use sliders to tune ROI)"
        cv2.putText(clone, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 0) if is_playing else (0, 0, 255), 2)

        # Show zoomed crop
        if w > 0 and h > 0 and x + w <= w_frame and y + h <= h_frame:
            crop = frame[y:y+h, x:x+w]
            crop_big = cv2.resize(crop, (max(w*4, 100), max(h*4, 100)), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Zoomed Crop', crop_big)

        cv2.imshow('ROI Tuner', clone)
        
        # Wait for key
        wait_time = delay if is_playing else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord(' '):
            is_playing = not is_playing
        elif key == ord('n') and not is_playing:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                ret, frame = cap.read()
        elif key == ord('p') and not is_playing:
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 2))
            ret, frame = cap.read()
            
        # Lock ROIs
        elif key == ord('1'): active_rois['p1_points'] = (x, y, w, h); print(f"Locked P1 Points: {(x, y, w, h)}")
        elif key == ord('2'): active_rois['p2_points'] = (x, y, w, h); print(f"Locked P2 Points: {(x, y, w, h)}")
        elif key == ord('3'): active_rois['p1_games'] = (x, y, w, h); print(f"Locked P1 Games: {(x, y, w, h)}")
        elif key == ord('4'): active_rois['p2_games'] = (x, y, w, h); print(f"Locked P2 Games: {(x, y, w, h)}")
        elif key == ord('5'): active_rois['p1_sets'] = (x, y, w, h); print(f"Locked P1 Sets: {(x, y, w, h)}")
        elif key == ord('6'): active_rois['p2_sets'] = (x, y, w, h); print(f"Locked P2 Sets: {(x, y, w, h)}")

        elif key == ord('s'):
            print("\n" + "="*50)
            print("COPY AND PASTE THIS INTO src/03_process.py:")
            print("="*50)
            print("ROIS_SHRUNK = {")
            for k, (lx, ly, lw, lh) in active_rois.items():
                print(f"    '{k}': ({lx}, {ly}, {lw}, {lh}),")
            print("}")
            print("="*50 + "\n")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()