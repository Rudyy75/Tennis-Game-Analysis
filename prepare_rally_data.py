import pandas as pd
import numpy as np
import os
import argparse
from typing import Tuple, Dict, List

import sys

MATCH_ID = sys.argv[1] if len(sys.argv) > 1 else 'v008'

# EXPECTED INPUT FILES (from teammates) 
SCORE_CSV = f'scores_{MATCH_ID}.csv'
TRACKING_CSV = f'pose_{MATCH_ID}.csv'         
SHOTS_CSV = f'shot_{MATCH_ID}.csv'                 

# OUTPUT FILES  
FEATURES_CSV = f'rally_data/rally_features_{MATCH_ID}.csv'
SEQUENCES_NPZ = f'rally_data/rally_sequences_{MATCH_ID}.npz'

# CONSTANTS 
COURT_WIDTH = 1280   
COURT_HEIGHT = 720   
FPS = 25
MAX_RALLY_LENGTH = 150  
NET_Y_RATIO = 0.50   


def load_real_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
   
    print("Loading real data...")
    
    # --- Load tracking (bounding boxes) ---
    tracking = pd.read_csv(TRACKING_CSV)
    tracking.columns = tracking.columns.str.strip()
    
    # Rename 'frame' → 'frame_id' for merge compatibility
    if 'frame' in tracking.columns and 'frame_id' not in tracking.columns:
        tracking = tracking.rename(columns={'frame': 'frame_id'})
    
    # Convert to numeric
    for col in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'player_x', 'player_y', 'ball_x', 'ball_y']:
        if col in tracking.columns:
            tracking[col] = pd.to_numeric(tracking[col], errors='coerce')
    
    # Compute center points if not already present
    if 'player_x' not in tracking.columns:
        tracking['player_x'] = (tracking['bbox_x1'] + tracking['bbox_x2']) / 2
        tracking['player_y'] = (tracking['bbox_y1'] + tracking['bbox_y2']) / 2
    
    print(f"  Tracking: {len(tracking)} rows, "
          f"frames: {tracking['frame_id'].nunique()}, "
          f"players: {sorted(tracking['player_id'].unique())}")
    
    # Load scores & segment rallies FIRST 
    scores = pd.read_csv(SCORE_CSV)
    scores.columns = scores.columns.str.strip()
    print(f"  Scores: {len(scores)} frames")
    
    scores = scores.sort_values('frame_id').reset_index(drop=True)
    scores['score_changed'] = (
        (scores['p1_points'] != scores['p1_points'].shift(1)) |
        (scores['p2_points'] != scores['p2_points'].shift(1))
    )
    scores.loc[scores.index[0], 'score_changed'] = False
    scores['rally_id'] = scores['score_changed'].cumsum()
    
    # Pivot tracking: one row per frame with both players
    # Only select columns needed for the pipeline to avoid overlap
    pivot_cols = ['player_x', 'player_y', 'ball_x', 'ball_y']
    tracking_pivot = tracking[['frame_id', 'player_id'] + pivot_cols].copy()
    
    p0 = tracking_pivot[tracking_pivot['player_id'] == 1].set_index('frame_id').drop(columns=['player_id']).rename(columns={
        'player_x': 'p0_player_x', 'player_y': 'p0_player_y',
        'ball_x': 'p0_ball_x', 'ball_y': 'p0_ball_y',
    })
    p1 = tracking_pivot[tracking_pivot['player_id'] == 2].set_index('frame_id').drop(columns=['player_id']).rename(columns={
        'player_x': 'p1_player_x', 'player_y': 'p1_player_y',
        'ball_x': 'p1_ball_x', 'ball_y': 'p1_ball_y',
    })
    
    merged = p0.join(p1, how='outer')
    merged = merged.sort_index().ffill().bfill().reset_index()
    
    # Forward-fill ball positions (use whichever player has ball detection)
    for col in ['p0_ball_x', 'p0_ball_y']:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    
    print(f"  Merged (both players): {len(merged)} frames")
    
    # Load shot classifications 
    SHOT_NAMES = ['Serve', 'Forehand', 'Backhand', 'Smash', 'Volley', 'Lob']
    
    if os.path.exists(SHOTS_CSV):
        shots = pd.read_csv(SHOTS_CSV)
        shots.columns = shots.columns.str.strip()
        if 'frame' in shots.columns and 'frame_id' not in shots.columns:
            shots = shots.rename(columns={'frame': 'frame_id'})
        
        # shot_type already a string in v010 format
        if 'shot_type' in shots.columns:
            shots = shots.sort_values('frame_id').drop_duplicates('frame_id', keep='first')
            
            # Make sure we grab player_id if it exists to know WHO hit the shot!
            cols_to_keep = ['frame_id', 'shot_type']
            if 'player_id' in shots.columns:
                cols_to_keep.append('player_id')
                
            merged = pd.merge(merged, shots[cols_to_keep], on='frame_id', how='left')
            if 'player_id' in merged.columns:
                merged = merged.rename(columns={'player_id': 'shot_player_id'})
                merged['shot_player_id'] = merged['shot_player_id'].fillna(0)
                
            merged['shot_type'] = merged['shot_type'].fillna('Unknown')
            print(f"  Shots loaded: {(merged['shot_type'] != 'Unknown').sum()} classified frames")
        else:
            merged['shot_type'] = 'Unknown'
    else:
        print(f"  WARNING: {SHOTS_CSV} not found — shot types set to Unknown")
        merged['shot_type'] = 'Unknown'
    
    print("  Merging scores onto tracking...")
    cols_to_merge = ['frame_id', 'p1_points', 'p2_points', 'rally_id']
    if 'p1_games' in scores.columns:
        cols_to_merge.extend(['p1_games', 'p2_games'])
    if 'p1_sets' in scores.columns:
        cols_to_merge.extend(['p1_sets', 'p2_sets'])
    
    scores_sorted = scores[cols_to_merge].sort_values('frame_id').reset_index(drop=True)
    score_frame_ids = scores_sorted['frame_id'].values
    
    merged_sorted = merged.sort_values('frame_id').reset_index(drop=True)
    merged_frame_ids = merged_sorted['frame_id'].values
    
    indices = np.searchsorted(score_frame_ids, merged_frame_ids, side='right') - 1
    indices = np.clip(indices, 0, len(scores_sorted) - 1)
    
    for col in cols_to_merge:
        if col != 'frame_id':
            merged_sorted[col] = scores_sorted[col].values[indices]
    
    merged = merged_sorted
    print("  Merge complete.")
    
    # Determine rally winners 
    rally_winners: Dict[int, int] = {}
    score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}
    
    rally_ids = sorted(merged['rally_id'].unique())
    
    # Pre-filter out OCR noise / flickers before calculating winners
    valid_rally_ids = []
    for rid in rally_ids:
        if len(merged[merged['rally_id'] == rid]) >= 5:
            valid_rally_ids.append(rid)
            
    ambiguous_count = 0
    
    for i, rid in enumerate(valid_rally_ids):
        rally_data = merged[merged['rally_id'] == rid]
        if len(rally_data) < 1:
            continue
        
        # Last rally: infer winner from final score state
        if i + 1 >= len(valid_rally_ids):
            p1_final = str(rally_data.iloc[-1]['p1_points']).strip()
            p2_final = str(rally_data.iloc[-1]['p2_points']).strip()
            p1_f = score_map.get(p1_final, 0)
            p2_f = score_map.get(p2_final, 0)
            # If one player is at AD or ahead, they likely won the last point
            if p1_f > p2_f:
                rally_winners[rid] = 1
            elif p2_f > p1_f:
                rally_winners[rid] = 0
            continue
        
        next_rid = valid_rally_ids[i + 1]
        next_rally = merged[merged['rally_id'] == next_rid]
        if len(next_rally) == 0:
            continue
        
        p1_before = str(rally_data.iloc[-1]['p1_points']).strip()
        p1_after = str(next_rally.iloc[0]['p1_points']).strip()
        p2_before = str(rally_data.iloc[-1]['p2_points']).strip()
        p2_after = str(next_rally.iloc[0]['p2_points']).strip()
        
        p1_b = score_map.get(p1_before, 0)
        p1_a = score_map.get(p1_after, 0)
        p2_b = score_map.get(p2_before, 0)
        p2_a = score_map.get(p2_after, 0)
        
        winner = None
        
        # Case 1: Normal point increment (e.g., 15->30, 30->40)
        if p1_a > p1_b and p2_a == p2_b:
            winner = 1  # P1 scored
        elif p2_a > p2_b and p1_a == p1_b:
            winner = 0  # P2 scored
        elif p1_a > p1_b:
            winner = 1
        elif p2_a > p2_b:
            winner = 0
        
        # Case 2: Score reset to 0-0 (game ended)
        # The player who had the winning score (40 or AD) won the game
        elif p1_a == 0 and p2_a == 0:
            # First: check game counter changes
            g1_a = int(float(next_rally.iloc[0].get('p1_games', 0)) if pd.notna(next_rally.iloc[0].get('p1_games')) else 0)
            g1_b = int(float(rally_data.iloc[-1].get('p1_games', 0)) if pd.notna(rally_data.iloc[-1].get('p1_games')) else 0)
            g2_a = int(float(next_rally.iloc[0].get('p2_games', 0)) if pd.notna(next_rally.iloc[0].get('p2_games')) else 0)
            g2_b = int(float(rally_data.iloc[-1].get('p2_games', 0)) if pd.notna(rally_data.iloc[-1].get('p2_games')) else 0)
            if g1_a > g1_b:
                winner = 1
            elif g2_a > g2_b:
                winner = 0
            else:
                # Game counter didn't change in OCR — infer from point scores
                # The player at 40/AD before the reset won the game
                if p1_b >= 3 and p1_b > p2_b:
                    winner = 1  # P1 was at 40+ and ahead
                elif p2_b >= 3 and p2_b > p1_b:
                    winner = 0  # P2 was at 40+ and ahead
                elif p1_b == 4:  # P1 had AD
                    winner = 1
                elif p2_b == 4:  # P2 had AD
                    winner = 0
                elif p1_b == 3 and p2_b == 3:
                    # Both at 40 (deuce), someone won 2 points to win game
                    # Check set counter as last resort
                    s1_a = int(float(next_rally.iloc[0].get('p1_sets', 0)) if pd.notna(next_rally.iloc[0].get('p1_sets')) else 0)
                    s1_b = int(float(rally_data.iloc[-1].get('p1_sets', 0)) if pd.notna(rally_data.iloc[-1].get('p1_sets')) else 0)
                    s2_a = int(float(next_rally.iloc[0].get('p2_sets', 0)) if pd.notna(next_rally.iloc[0].get('p2_sets')) else 0)
                    s2_b = int(float(rally_data.iloc[-1].get('p2_sets', 0)) if pd.notna(rally_data.iloc[-1].get('p2_sets')) else 0)
                    if s1_a > s1_b:
                        winner = 1
                    elif s2_a > s2_b:
                        winner = 0
        
        # Case 3: AD -> Deuce (40-40): the player NOT at AD won the point
        elif p1_b == 4 and p1_a == 3 and p2_a == 3:
            winner = 0  # P1 had AD, lost it -> P2 won
        elif p2_b == 4 and p2_a == 3 and p1_a == 3:
            winner = 1  # P2 had AD, lost it -> P1 won
        
        # Case 4: Score decreased but not to 0-0 (e.g., AD->40 in deuce)
        elif p1_a < p1_b and p2_a == p2_b:
            winner = 0  # P1's score went down -> P2 won
        elif p2_a < p2_b and p1_a == p1_b:
            winner = 1  # P2's score went down -> P1 won
        
        if winner is not None:
            rally_winners[rid] = winner
        else:
            ambiguous_count += 1
    
    print(f"  Rallies with known winners: {len(rally_winners)} "
          f"(P1: {sum(v == 1 for v in rally_winners.values())}, "
          f"P2: {sum(v == 0 for v in rally_winners.values())})")
    if ambiguous_count > 0:
        print(f"  Ambiguous (dropped): {ambiguous_count}")
    
    # Extract features and sequences 
    features_list: List[Dict] = []
    sequences_list: List[np.ndarray] = []
    labels: List[int] = []
    
    n_features = 22
    NET_Y = COURT_HEIGHT * NET_Y_RATIO
    
    for rid, winner in rally_winners.items():
        rally = merged[merged['rally_id'] == rid].copy()
        # No need to check len < 5 again, already pre-filtered
        
        # Court positions
        p1_x = rally['p0_player_x'].values.astype(float)
        p1_y = rally['p0_player_y'].values.astype(float)
        p2_x = rally['p1_player_x'].values.astype(float)
        p2_y = rally['p1_player_y'].values.astype(float)
        ball_x = rally['p0_ball_x'].values.astype(float)
        ball_y = rally['p0_ball_y'].values.astype(float)
        
        # Velocities
        p1_vx = np.diff(p1_x, prepend=p1_x[0])
        p1_vy = np.diff(p1_y, prepend=p1_y[0])
        p2_vx = np.diff(p2_x, prepend=p2_x[0])
        p2_vy = np.diff(p2_y, prepend=p2_y[0])
        ball_vx = np.diff(ball_x, prepend=ball_x[0])
        ball_vy = np.diff(ball_y, prepend=ball_y[0])
        
        # Accelerations
        p1_accel = np.sqrt(np.diff(p1_vx, prepend=0)**2 + np.diff(p1_vy, prepend=0)**2)
        p2_accel = np.sqrt(np.diff(p2_vx, prepend=0)**2 + np.diff(p2_vy, prepend=0)**2)
        
        # Speeds
        p1_speed = np.sqrt(p1_vx**2 + p1_vy**2)
        p2_speed = np.sqrt(p2_vx**2 + p2_vy**2)
        ball_speed = np.sqrt(ball_vx**2 + ball_vy**2)
        
        # Shot type one-hot
        shot_onehot = np.zeros((len(rally), 6))
        if 'shot_type' in rally.columns:
            for idx, st in enumerate(rally['shot_type'].values):
                st_str = str(st).strip()
                if st_str in SHOT_NAMES:
                    shot_onehot[idx, SHOT_NAMES.index(st_str)] = 1.0
        
        n_shots = int(np.sum(shot_onehot.sum(axis=1) > 0))
        if n_shots == 0:
            n_shots = max(1, len(rally) // 15)
        
        # Score encoding
        p1_score_arr = np.array([score_map.get(str(s).strip(), 0) for s in rally['p1_points'].values])
        p2_score_arr = np.array([score_map.get(str(s).strip(), 0) for s in rally['p2_points'].values])
        
        # Build per-frame sequence (22 features)
        sequence = np.column_stack([
            p1_x / COURT_WIDTH, p1_y / COURT_HEIGHT,
            p2_x / COURT_WIDTH, p2_y / COURT_HEIGHT,
            ball_x / COURT_WIDTH, ball_y / COURT_HEIGHT,
            p1_vx / 50, p1_vy / 50,
            p2_vx / 50, p2_vy / 50,
            ball_vx / 50, ball_vy / 50,
            p1_accel / 50,
            p2_accel / 50,
            shot_onehot,
            p1_score_arr / 3,
            p2_score_arr / 3,
        ])
        
        padded = np.zeros((MAX_RALLY_LENGTH, n_features))
        trim_len = min(len(rally), MAX_RALLY_LENGTH)
        padded[:trim_len, :] = sequence[:trim_len, :]
        
        sequences_list.append(padded)
        labels.append(winner)
        
        # Per-rally summary features
        p1_net_time = np.mean(p1_y < NET_Y + 80)
        p2_net_time = np.mean(p2_y > NET_Y - 80)
        
        late_start = int(len(rally) * 0.7)
        p1_late_spread = np.std(p1_x[late_start:]) + np.std(p1_y[late_start:])
        p2_late_spread = np.std(p2_x[late_start:]) + np.std(p2_y[late_start:])
        
        midpoint = len(rally) // 2
        early_ball_speed = np.mean(ball_speed[:midpoint]) if midpoint > 0 else 0
        late_ball_speed = np.mean(ball_speed[midpoint:])
        speed_momentum = late_ball_speed - early_ball_speed
        
        p1_dist_total = np.sum(p1_speed)
        p2_dist_total = np.sum(p2_speed)
        
        # Server identification: Check shot_type=='Serve' first
        server_identified = 0
        serve_frames = rally[(rally.get('shot_type', '') == 'Serve') & (rally.get('shot_player_id', 0) > 0)]
        if not serve_frames.empty:
            server_identified = int(serve_frames.iloc[0]['shot_player_id'])
        else:
            # Fallback to closest to ball at rally start
            valid_ball = (ball_x != 0) & (ball_y != 0)
            if np.any(valid_ball):
                first_idx = np.where(valid_ball)[0][0]
                d1 = (p1_x[first_idx] - ball_x[first_idx])**2 + (p1_y[first_idx] - ball_y[first_idx])**2
                d2 = (p2_x[first_idx] - ball_x[first_idx])**2 + (p2_y[first_idx] - ball_y[first_idx])**2
                server_identified = 1 if d1 < d2 else 2
        
        features_list.append({
            'rally_id': rid,
            'rally_length': len(rally),
            'rally_type': 'real',
            'server': server_identified,
            'n_shots': n_shots,
            'p1_distance': p1_dist_total,
            'p2_distance': p2_dist_total,
            'dist_ratio': p1_dist_total / (p2_dist_total + 1e-5),
            'p1_avg_speed': np.mean(p1_speed) if np.any(p1_speed) else 0,
            'p2_avg_speed': np.mean(p2_speed) if np.any(p2_speed) else 0,
            'p1_max_speed': np.max(p1_speed) if np.any(p1_speed) else 0,
            'p2_max_speed': np.max(p2_speed) if np.any(p2_speed) else 0,
            'late_speed_diff': np.mean(p1_speed[late_start:]) - np.mean(p2_speed[late_start:]) if late_start < len(p1_speed) else 0,
            'p1_avg_accel': np.mean(p1_accel) if np.any(p1_accel) else 0,
            'p2_avg_accel': np.mean(p2_accel) if np.any(p2_accel) else 0,
            'accel_diff': np.mean(p1_accel) - np.mean(p2_accel) if np.any(p1_accel) and np.any(p2_accel) else 0,
            'ball_avg_speed': np.mean(ball_speed) if np.any(ball_speed) else 0,
            'p1_net_proximity': p1_net_time,
            'p2_net_proximity': p2_net_time,
            'net_diff': p1_net_time - p2_net_time,
            'p1_court_coverage': np.std(p1_x) * np.std(p1_y) if len(p1_x) > 1 else 0,
            'p2_court_coverage': np.std(p2_x) * np.std(p2_y) if len(p2_x) > 1 else 0,
            'p1_late_spread': p1_late_spread if not np.isnan(p1_late_spread) else 0,
            'p2_late_spread': p2_late_spread if not np.isnan(p2_late_spread) else 0,
            'speed_momentum': speed_momentum if not np.isnan(speed_momentum) else 0,
            'p1_score': int(p1_score_arr[0]),
            'p2_score': int(p2_score_arr[0]),
            'p1_games': int(rally['p1_games'].values[0]) if 'p1_games' in rally.columns and not pd.isna(rally['p1_games'].values[0]) else -1,
            'p2_games': int(rally['p2_games'].values[0]) if 'p2_games' in rally.columns and not pd.isna(rally['p2_games'].values[0]) else -1,
            'p1_sets': int(rally['p1_sets'].values[0]) if 'p1_sets' in rally.columns and not pd.isna(rally['p1_sets'].values[0]) else -1,
            'p2_sets': int(rally['p2_sets'].values[0]) if 'p2_sets' in rally.columns and not pd.isna(rally['p2_sets'].values[0]) else -1,
            'winner': winner,
        })
    
    if len(labels) == 0:
        print("ERROR: No rallies with known winners found!")
        return pd.DataFrame(), np.zeros((0, MAX_RALLY_LENGTH, n_features)), np.array([])
    
    features_df = pd.DataFrame(features_list)
    sequences = np.array(sequences_list)
    labels = np.array(labels)
    
    return features_df, sequences, labels


def main():
    os.makedirs('rally_data', exist_ok=True)
    
    if not os.path.exists(TRACKING_CSV):
        print(f"ERROR: {TRACKING_CSV} not found!")
        return
        
    features_df, sequences, labels = load_real_data()
    
    if len(labels) == 0:
        print("No data to save.")
        return
    
    # Save outputs 
    features_df.to_csv(FEATURES_CSV, index=False)
    np.savez(SEQUENCES_NPZ, sequences=sequences, labels=labels)
    
    # Print summary 
    print(f"Total rallies: {len(labels)}")
    print(f"P1 wins: {np.sum(labels == 1)} ({np.mean(labels == 1):.1%})")
    print(f"P2 wins: {np.sum(labels == 0)} ({np.mean(labels == 0):.1%})")
    print(f"Sequence shape: {sequences.shape} (rallies, timesteps, features)")
    print(f"Features shape: {features_df.shape} (rallies, features)")
    print(f"\nSaved to:")
    print(f"  {FEATURES_CSV}")
    print(f"  {SEQUENCES_NPZ}")
    print(f"\nFeature columns: {list(features_df.columns)}")


if __name__ == '__main__':
    main()

