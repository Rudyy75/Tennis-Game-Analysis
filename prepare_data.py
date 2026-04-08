"""
prepare_data.py  —  Tennis Dashboard Data Preparation
NITK IE Project 2025-26

Usage:
    python prepare_data.py \
        --video      outputs/run_.../visualization_rich_h264.mp4 \
        --shots      outputs/run_.../final_dataset_v008.csv \
        --traj       outputs/run_.../trajectories.csv \
        --anirudh    outputs/run_.../dashboard_v008.csv \
        [--detections  outputs/run_.../detections.csv] \
        [--out       dashboard_data.json]

Anirudh's CSV must have these columns:
    rally_id, start_frame, end_frame, server, actual_winner,
    predicted_winner, correct, score, p1_win_probability, p2_win_probability
"""

import argparse, json, os, sys, re
import cv2
import pandas as pd
from pathlib import Path

SHOT_LABELS   = ["none", "serve", "forehand", "backhand", "smash", "volley"]
INVALID_SHOTS = {"none", "nan", "", "na", "null", "nat"}


def get_video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {path}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  {Path(path).name}  {w}x{h}  {fps:.2f}fps  {total} frames")
    return {"fps": round(fps, 4), "total_frames": total,
            "width": w, "height": h, "name": Path(path).stem}


def load_shots(path):
    df       = pd.read_csv(path)
    shot_col = next((c for c in ["predicted_shot", "shot_type"] if c in df.columns), None)
    if not shot_col:
        print("  [WARN] No shot column found")
        return []
    df[shot_col] = df[shot_col].astype(str).str.strip().str.lower()
    out = []
    for _, r in df.iterrows():
        s = r[shot_col]
        if s in INVALID_SHOTS or s not in SHOT_LABELS:
            s = "none"
        out.append({"f": int(r["frame_id"]) - 1, "p": int(r["player_id"]),
                    "s": SHOT_LABELS.index(s)})
    out.sort(key=lambda x: x["f"])
    print(f"  Shots: {len(out)} ({sum(1 for x in out if x['s'] != 0)} non-none)")
    return out


def load_trajectories(path):
    df  = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        if pd.notna(r["cx"]) and pd.notna(r["cy"]):
            out.append([int(r["frame"]) - 1, int(r["player_id"]),
                        round(float(r["cx"]), 1), round(float(r["cy"]), 1)])
    out.sort(key=lambda x: x[0])
    print(f"  Positions: {len(out)} records")
    return out


def load_rally_segments(path):
    """Gameplay segments from detections.csv — used for heatmap/shot identity."""
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if "is_gameplay" not in df.columns:
        return []
    rallies, in_rally, start, rid = [], False, None, 0
    for _, r in df.iterrows():
        frame = int(r["frame"]) - 1
        gp    = int(r["is_gameplay"]) == 1
        if gp and not in_rally:
            in_rally, start = True, frame
        elif not gp and in_rally:
            in_rally = False
            rid += 1
            rallies.append({"id": rid, "start": start, "end": frame})
    if in_rally:
        rid += 1
        rallies.append({"id": rid, "start": start, "end": int(df["frame"].max()) - 1})
    print(f"  Gameplay segments: {len(rallies)}")
    return rallies


def load_anirudh_csv(path):
    """
    Load Anirudh's merged dashboard CSV.

    Returns:
        probabilities     — list of dicts for score card + momentum chart
        rally_predictions — list of dicts for rally log table
        anirudh_segments  — list of {id, start, end} for precise frame sync
        identity_table    — {rally_id: a_is_p1 bool} for P1/P2 → A/B resolution
    """
    if not path or not os.path.exists(path):
        print("  Anirudh CSV: not provided")
        return None, None, [], {}

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    required = {"rally_id", "start_frame", "end_frame", "server",
                "actual_winner", "predicted_winner", "correct",
                "score", "p1_win_probability", "p2_win_probability"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Anirudh CSV missing columns: {missing}")

    print(f"  Anirudh CSV: {len(df)} rallies")

    probabilities     = []
    rally_predictions = []
    anirudh_segments  = []
    identity_table    = {}
    current_set  = 0
    prev_set_sum = 0

    for _, r in df.iterrows():
        rid = int(r["rally_id"])

        # ── Identity resolution (set boundary detection) ───────────
        m = re.search(r'Sets:\s*(\d+)-(\d+)', str(r.get("score", "")))
        if m:
            ss = int(m.group(1)) + int(m.group(2))
            if ss > prev_set_sum:
                current_set  += 1
                prev_set_sum  = ss
        identity_table[rid] = (current_set % 2 == 0)

        # ── Probabilities (score card + momentum chart) ────────────
        probabilities.append({
            "Rally_ID":           rid,
            "Score":              str(r["score"]),
            "P1_Win_Probability": float(r["p1_win_probability"]),
            "P2_Win_Probability": float(r["p2_win_probability"]),
        })

        # ── Rally predictions (rally log table) ────────────────────
        rally_predictions.append({
            "Rally_ID":         rid,
            "Server":           str(r["server"]).strip().upper(),
            "Actual_Winner":    str(r["actual_winner"]).strip().upper(),
            "Predicted_Winner": str(r["predicted_winner"]).strip().upper(),
            "Correct":          int(r["correct"]),
        })

        # ── Frame segments (precise video sync) ────────────────────
        # Convert to 0-based frame indices
        anirudh_segments.append({
            "id":    rid,
            "start": int(r["start_frame"]) - 1,
            "end":   int(r["end_frame"])   - 1,
        })

    sets_detected = current_set + 1
    print(f"  Identity table: {len(identity_table)} rallies, {sets_detected} sets detected")
    print(f"  Frame range   : {anirudh_segments[0]['start']} → {anirudh_segments[-1]['end']}")

    return probabilities, rally_predictions, anirudh_segments, identity_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True, help="Annotated output video (.mp4)")
    ap.add_argument("--shots",      required=True, help="final_dataset CSV from pipeline")
    ap.add_argument("--traj",       required=True, help="trajectories.csv from pipeline")
    ap.add_argument("--anirudh",    default=None,  help="Anirudh's merged dashboard CSV")
    ap.add_argument("--detections", default=None,  help="detections.csv (optional, for gameplay segments)")
    ap.add_argument("--out",        default="dashboard_data.json")
    args = ap.parse_args()

    print("\n── Video ─────────────────────────────────")
    meta = get_video_meta(args.video)

    print("\n── Pipeline data ─────────────────────────")
    shots     = load_shots(args.shots)
    positions = load_trajectories(args.traj)
    segments  = load_rally_segments(args.detections)

    print("\n── Anirudh data ──────────────────────────")
    probs, rallies, anirudh_segs, identity_table = load_anirudh_csv(args.anirudh)

    payload = {
        "meta":              meta,
        "shot_labels":       SHOT_LABELS,
        "shots":             shots,
        "positions":         positions,
        "rally_segments":    segments,        # gameplay segments from our detection pipeline
        "anirudh_segments":  anirudh_segs,   # precise rally frame ranges from Anirudh
        "identity_table":    {str(k): v for k, v in identity_table.items()},
        "probabilities":     probs,
        "rally_predictions": rallies,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n── Output: {args.out}  ({size_mb:.1f} MB) ──────────────")
    print(f"""
Steps:
  1. Open dashboard.html in Chrome / Firefox
  2. Select Video  → your _h264.mp4
  3. Select Data   → {args.out}
  4. Play — all panels sync live to exact rally frame boundaries
""")


if __name__ == "__main__":
    main()