import pandas as pd
import numpy as np
import sys
import os

def export_dashboard(match_id='v008'):
    scores_file = f'scores_{match_id}.csv'
    sim_file = f'models/match_simulation_lstm_{match_id}.csv'
    feats_file = f'rally_data/rally_features_{match_id}.csv'
    out_file = f'dashboard_{match_id}.csv'
    
    if not os.path.exists(sim_file) or not os.path.exists(feats_file) or not os.path.exists(scores_file):
        print(f"Missing files for {match_id}")
        return
        
    scores = pd.read_csv(scores_file)
    scores.columns = scores.columns.str.strip()
    scores = scores.sort_values('frame_id').reset_index(drop=True)
    
    scores['score_changed'] = (
        (scores['p1_points'] != scores['p1_points'].shift(1)) |
        (scores['p2_points'] != scores['p2_points'].shift(1))
    )
    scores.loc[scores.index[0], 'score_changed'] = False
    scores['rally_id'] = scores['score_changed'].cumsum()
    
    # Get frame bounds for each rally
    frames_df = scores.groupby('rally_id').agg(
        start_frame=('frame_id', 'min'),
        end_frame=('frame_id', 'max')
    ).reset_index()
    
    # Load features for server info
    feats = pd.read_csv(feats_file)[['rally_id', 'server']]
    
    # Load simulation output for everything else
    sim = pd.read_csv(sim_file)
    
    # Rename for merge if needed
    if 'Rally_ID' in sim.columns:
        sim = sim.rename(columns={'Rally_ID': 'rally_id'})
        
    merged = pd.merge(sim, frames_df, on='rally_id', how='left')
    merged = pd.merge(merged, feats, on='rally_id', how='left')
    
    merged['start_frame'] = merged['start_frame'].astype(int)
    merged['end_frame'] = merged['end_frame'].astype(int)
    merged['p1_win_probability'] = merged['P1_Win_Probability'].round(4)
    merged['p2_win_probability'] = (1.0 - merged['p1_win_probability']).round(4)
    merged['score'] = merged['Score']
    merged['actual_winner'] = merged['Actual_Winner']
    merged['predicted_winner'] = merged['Predicted_Winner']
    merged['correct'] = merged['Correct'].astype(int)
    merged['server'] = merged['server'].fillna(0).map({1: 'P1', 2: 'P2', 0: 'Unknown'})
    
    out_cols = [
        'rally_id', 'start_frame', 'end_frame', 'server', 
        'actual_winner', 'predicted_winner', 'correct', 
        'score', 'p1_win_probability', 'p2_win_probability'
    ]
    
    out_df = merged[out_cols].copy()
    out_df.to_csv(out_file, index=False)
    print(f"Exported {len(out_df)} rallies with precise frame boundaries to {out_file}")
    
if __name__ == '__main__':
    match_id = sys.argv[1] if len(sys.argv) > 1 else 'v008'
    export_dashboard(match_id)
