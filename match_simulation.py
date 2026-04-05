import pandas as pd
import numpy as np
import os

import sys

MATCH_ID = sys.argv[1] if len(sys.argv) > 1 else 'v010'

MULTITASK_OOF = f'models/rally_predictions_multitask_oof_{MATCH_ID}.csv'
SINGLETASK_OOF = f'models/rally_predictions_lstm_oof_{MATCH_ID}.csv'
FEATURES_CSV = f'rally_data/rally_features_{MATCH_ID}.csv'
SCORES_CSV = f'scores_{MATCH_ID}.csv'
OUTPUT_CSV = f'models/match_simulation_lstm_{MATCH_ID}.csv'

POINT_LABELS = ['0', '15', '30', '40']


def load_predictions() -> pd.DataFrame:
    for path, label in [(MULTITASK_OOF, 'multi-task'), (SINGLETASK_OOF, 'single-task')]:
        if os.path.exists(path):
            print(f"  Using {label} OOF: {path}")
            return pd.read_csv(path).sort_values('Rally_ID').reset_index(drop=True)
    raise FileNotFoundError("No OOF predictions found.")


def build_running_score(feat_df: pd.DataFrame, pred_df: pd.DataFrame) -> list:
    # Build lookup from predictions
    pred_lookup = {}
    for _, row in pred_df.iterrows():
        rid = int(row['Rally_ID'])
        pred_lookup[rid] = {
            'pred_winner': row['Predicted_Winner'],
            'p1_rally_prob': row.get('P1_Rally_Prob', None),
            'p1_win_prob': row.get('P1_Win_Prob', None),
        }
    merged = pd.merge(feat_df, pred_df, left_on='rally_id', right_on='Rally_ID')
    
    # Running score state
    p1_sets, p2_sets = 0, 0
    p1_games, p2_games = 0, 0

    # Identify game boundaries using OCR (when score resets to 0-0)
    # A new game starts when p1_score == 0 and p2_score == 0
    # Exception: the very first rally is always the start of Game 1
    games_list = []
    current_game = []
    
    for _, row in merged.iterrows():
        p1_s = row.get('p1_score', 0)
        p2_s = row.get('p2_score', 0)
        
        # Detect new game boundary
        # It's a new game if it's 0-0 AND we already have played points in the current game
        # OR if it's the very first rally of the match
        if (p1_s == 0 and p2_s == 0 and len(current_game) > 0):
            games_list.append(current_game)
            current_game = []
            
        current_game.append(row)
        
    if len(current_game) > 0:
        games_list.append(current_game)
        
    results = []
    p1_pts, p2_pts = 0, 0
    p1_games_sim, p2_games_sim = 0, 0
    p1_sets_sim, p2_sets_sim = 0, 0
    
    for game_idx, game_rallies in enumerate(games_list):
        p1_pts, p2_pts = 0, 0
        
        for i, row in enumerate(game_rallies):
            rid = int(row['rally_id'])
            actual_winner = 'P1' if int(row['winner']) == 1 else 'P2'
            pred_winner = row['Predicted_Winner']
            
            p1_rally_prob = row.get('P1_Rally_Prob')
            p1_win_prob = row.get('P1_Win_Prob', 0.5)
            
            # Format current score
            # Output real score string, not divergent string
            p1_true = int(row.get('p1_score', 0))
            p2_true = int(row.get('p2_score', 0))
            p1_true_games = int(row.get('p1_games', 0))
            p2_true_games = int(row.get('p2_games', 0))
            p1_true_sets = int(row.get('p1_sets', 0))
            p2_true_sets = int(row.get('p2_sets', 0))
            
            # Map -1 to 0 if it wasn't valid in OCR
            p1_true_games = max(0, p1_true_games)
            p2_true_games = max(0, p2_true_games)
            p1_true_sets = max(0, p1_true_sets)
            p2_true_sets = max(0, p2_true_sets)
            
            score_str = f"Sets: {p1_true_sets}-{p2_true_sets} | Games: {p1_true_games}-{p2_true_games} | Points: {POINT_LABELS[min(p1_true, 3)]}-{POINT_LABELS[min(p2_true, 3)]}"
            if p1_true > 3 or p2_true > 3:
                if p1_true == p2_true: score_str = score_str.replace(f"Points: {POINT_LABELS[min(p1_true, 3)]}-{POINT_LABELS[min(p2_true, 3)]}", "Points: Deuce-Deuce")
                elif p1_true > p2_true: score_str = score_str.replace(f"Points: {POINT_LABELS[min(p1_true, 3)]}-{POINT_LABELS[min(p2_true, 3)]}", "Points: Ad-40")
                else: score_str = score_str.replace(f"Points: {POINT_LABELS[min(p1_true, 3)]}-{POINT_LABELS[min(p2_true, 3)]}", "Points: 40-Ad")
            
            wp = float(p1_win_prob) if p1_win_prob is not None and not pd.isna(p1_win_prob) else 0.5
            
            result = {
                'Rally_ID': rid,
                'Score': score_str,
                'Actual_Winner': actual_winner,
                'Predicted_Winner': pred_winner,
                'Correct': pred_winner == actual_winner,
                'P1_Win_Percent': f"{wp * 100:.1f}%",
                'P2_Win_Percent': f"{(1 - wp) * 100:.1f}%",
                'P1_Win_Probability': round(wp, 4),
                'P1_Rally_Prob': round(float(p1_rally_prob), 4) if p1_rally_prob is not None and not pd.isna(p1_rally_prob) else None,
            }
            
            # If this is the LAST rally of the game, annotate it
            if i == len(game_rallies) - 1:
                # Decide who actually won the game (based on final point winner!)
                gw = actual_winner
                if gw == 'P1': p1_games_sim += 1
                else: p2_games_sim += 1
                
                # We'll still add the simulated marker annotation but without forcefully mutating the string
                # If you want to simulate set end boundaries:
                if (p1_games_sim >= 6 and (p1_games_sim - p2_games_sim) >= 2) or p1_games_sim == 7:
                    p1_sets_sim += 1
                    result['Score'] += f" ({gw} SET)"
                    p1_games_sim, p2_games_sim = 0, 0
                elif (p2_games_sim >= 6 and (p2_games_sim - p1_games_sim) >= 2) or p2_games_sim == 7:
                    p2_sets_sim += 1
                    result['Score'] += f" ({gw} SET)"
                    p1_games_sim, p2_games_sim = 0, 0
                else:
                    result['Score'] += f" ({gw} GAME)"
                    
            results.append(result)
            
    return results


def main():
    print(f"  MATCH SIMULATION — {MATCH_ID.upper()}")

    print("\nLoading predictions...")
    pred_df = load_predictions()
    print(f"  Loaded {len(pred_df)} predictions")

    print("Loading features...")
    feat_df = pd.read_csv(FEATURES_CSV)
    print(f"  Rallies: {len(feat_df)}")

    print("\nBuilding running score with predictions...")
    results = build_running_score(feat_df, pred_df)

    # Save
    os.makedirs('models', exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Summary 
    n = len(results)
    correct = sum(1 for r in results if r['Correct'])

    # Count games from actual score
    game_events = [r for r in results if 'GAME' in r['Score'] or 'SET' in r['Score']]
    game_winners = []
    pred_game_winners = []

    # Group rallies by game for game-level accuracy
    current_game_rallies = []
    for r in results:
        current_game_rallies.append(r)
        if 'GAME' in r['Score'] or 'SET' in r['Score']:
            # Determine actual and predicted game winner by majority
            actual_p1 = sum(1 for x in current_game_rallies if x['Actual_Winner'] == 'P1')
            actual_p2 = sum(1 for x in current_game_rallies if x['Actual_Winner'] == 'P2')
            pred_p1 = sum(1 for x in current_game_rallies if x['Predicted_Winner'] == 'P1')
            pred_p2 = sum(1 for x in current_game_rallies if x['Predicted_Winner'] == 'P2')
            actual_gw = 'P1' if actual_p1 > actual_p2 else 'P2'
            pred_gw = 'P1' if pred_p1 > pred_p2 else 'P2'
            game_winners.append(actual_gw)
            pred_game_winners.append(pred_gw)
            current_game_rallies = []

    game_correct = sum(1 for a, p in zip(game_winners, pred_game_winners) if a == p)
    n_games = len(game_winners)

    # Set accuracy
    set_events = [r for r in results if 'SET' in r['Score']]

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {MATCH_ID.upper()}")
    print(f"{'=' * 60}")
    print(f"\n  POINT ACCURACY: {correct}/{n} ({correct/n*100:.1f}%)")
    if n_games > 0:
        print(f"  GAME ACCURACY:  {game_correct}/{n_games} ({game_correct/n_games*100:.1f}%)")
    print(f"  GAMES COMPLETED: {n_games}")
    print(f"  SETS COMPLETED:  {len(set_events)}")

    # Final score
    last = results[-1]
    # Parse final score from last entry
    final_score = last['Score'].split('|')[0].strip()
    print(f"  FINAL: {final_score}")

    # Match winner
    actual_mw = 'P2' if sum(1 for w in game_winners if w == 'P2') > sum(1 for w in game_winners if w == 'P1') else 'P1'
    pred_mw = 'P2' if sum(1 for w in pred_game_winners if w == 'P2') > sum(1 for w in pred_game_winners if w == 'P1') else 'P1'
    print(f"  MATCH WINNER: {pred_mw} ({'✓' if actual_mw == pred_mw else '✗'}) (actual: {actual_mw})")

    # Game-by-game
    if n_games > 0:
        print(f"\n{'─' * 50}")
        print(f"  GAME-BY-GAME")
        print(f"{'─' * 50}")
        gi = 0
        cg_rallies = []
        for r in results:
            cg_rallies.append(r)
            if 'GAME' in r['Score'] or 'SET' in r['Score']:
                gi += 1
                ann = 'GAME' if 'GAME' in r['Score'] else 'SET'
                a_p1 = sum(1 for x in cg_rallies if x['Actual_Winner'] == 'P1')
                a_p2 = sum(1 for x in cg_rallies if x['Actual_Winner'] == 'P2')
                p_p1 = sum(1 for x in cg_rallies if x['Predicted_Winner'] == 'P1')
                p_p2 = sum(1 for x in cg_rallies if x['Predicted_Winner'] == 'P2')
                aw = 'P1' if a_p1 > a_p2 else 'P2'
                pw = 'P1' if p_p1 > p_p2 else 'P2'
                ok = '✓' if aw == pw else '✗'
                score_part = r['Score'].split('|')[1].strip()
                print(f"  {gi:>2}. {score_part:>15} | {len(cg_rallies)} pts | "
                      f"actual={aw} ({a_p1}-{a_p2}) pred={pw} ({p_p1}-{p_p2}) {ok} [{ann}]")
                cg_rallies = []

    # Win prob trajectory
    wps = [r['P1_Win_Probability'] for r in results]
    print(f"\n{'─' * 50}")
    print(f"  WIN PROB: {wps[0]:.1%} → {wps[-1]:.1%} (range [{min(wps):.1%}, {max(wps):.1%}])")

    print(f"\n  Saved → {OUTPUT_CSV}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()