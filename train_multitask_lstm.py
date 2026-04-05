
#   python src/18_train_multitask_lstm.py --match v010

import pandas as pd
import numpy as np
import os
import json
import argparse
from typing import Tuple, Dict, List

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

ALL_MATCHES = ['v007', 'v008', 'v010']

def paths_for(vid: str) -> Dict[str, str]:
    return {
        'sequences': f'rally_data/rally_sequences_{vid}.npz',
        'features': f'rally_data/rally_features_{vid}.csv',
        'scores': f'scores_{vid}.csv',
        'shot': f'shot_{vid}.csv',
        'rf_oof': f'models/rally_predictions_oof_{vid}.csv',
        'model': f'models/multitask_lstm_{vid}.keras',
        'meta': f'models/multitask_lstm_{vid}.json',
        'oof': f'models/rally_predictions_multitask_oof_{vid}.csv',
    }

N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 16
DROPOUT = 0.3
LR = 0.001
RALLY_LOSS_W = 1.0
WIN_PROB_LOSS_W = 0.3


# Feature Engineering (per-match)
def compute_match_context(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    ctx = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        if i == 0:
            ctx[i] = [0.5, 0.5, 0.0, 0.5]
        else:
            past = labels[:i]
            p1r = np.mean(past == 1)
            ctx[i, 0] = p1r
            ctx[i, 1] = 1.0 - p1r
            ctx[i, 2] = i / n
            ctx[i, 3] = np.mean((past[-5:] if len(past) >= 5 else past) == 1)
    return ctx


def load_score_context(feat: pd.DataFrame, scores_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    n = len(feat)
    game_data = np.zeros((n, 4), dtype=np.float32)
    if not os.path.exists(scores_csv):
        return game_data / np.array([6, 6, 2, 2], dtype=np.float32), game_data

    scores = pd.read_csv(scores_csv, low_memory=False)
    scores.columns = scores.columns.str.strip()
    scores = scores.sort_values('frame_id').reset_index(drop=True)
    scores['p1_points'] = scores['p1_points'].astype(str).str.strip()
    scores['p2_points'] = scores['p2_points'].astype(str).str.strip()
    scores['score_changed'] = (
        (scores['p1_points'] != scores['p1_points'].shift(1)) |
        (scores['p2_points'] != scores['p2_points'].shift(1))
    )
    scores.loc[scores.index[0], 'score_changed'] = False
    scores['rally_id'] = scores['score_changed'].cumsum()
    rally_first = scores.groupby('rally_id').first().reset_index()

    for idx, row in feat.iterrows():
        rid = row['rally_id']
        m = rally_first[rally_first['rally_id'] == rid]
        if len(m) > 0:
            r = m.iloc[0]
            for ci, col in enumerate(['p1_games', 'p2_games', 'p1_sets', 'p2_sets']):
                if col in r and pd.notna(r.get(col)):
                    game_data[idx, ci] = float(r[col])

    score_ctx = game_data.copy()
    score_ctx[:, 0] /= 6.0
    score_ctx[:, 1] /= 6.0
    score_ctx[:, 2] /= 2.0
    score_ctx[:, 3] /= 2.0
    return score_ctx, game_data


def load_shot_features(feat: pd.DataFrame, shot_csv: str, scores_csv: str) -> np.ndarray:
    n = len(feat)
    feats = np.zeros((n, 10), dtype=np.float32)
    if not os.path.exists(shot_csv) or not os.path.exists(scores_csv):
        return feats

    shots = pd.read_csv(shot_csv)
    shots.columns = shots.columns.str.strip()
    scores = pd.read_csv(scores_csv, low_memory=False)
    scores.columns = scores.columns.str.strip()
    scores = scores.sort_values('frame_id').reset_index(drop=True)
    scores['p1_points'] = scores['p1_points'].astype(str).str.strip()
    scores['p2_points'] = scores['p2_points'].astype(str).str.strip()
    scores['score_changed'] = (
        (scores['p1_points'] != scores['p1_points'].shift(1)) |
        (scores['p2_points'] != scores['p2_points'].shift(1))
    )
    scores.loc[scores.index[0], 'score_changed'] = False
    scores['rally_id'] = scores['score_changed'].cumsum()
    rally_bounds = scores.groupby('rally_id')['frame_id'].agg(['min', 'max'])

    for col in ['wrist_vel', 'arm_angle', 'ball_vel', 'wrist_ball_dist']:
        if col in shots.columns:
            shots[col] = pd.to_numeric(shots[col], errors='coerce')

    for idx, row in feat.iterrows():
        rid = row['rally_id']
        if rid not in rally_bounds.index:
            continue
        fmin, fmax = rally_bounds.loc[rid, 'min'], rally_bounds.loc[rid, 'max']
        rs = shots[(shots['frame_id'] >= fmin) & (shots['frame_id'] <= fmax)]
        if len(rs) == 0:
            continue
        for pid, off in [(1, 0), (2, 5)]:
            ps = rs[rs['player_id'] == pid]
            if len(ps) == 0:
                continue
            for i, col in enumerate(['wrist_vel', 'arm_angle', 'ball_vel', 'wrist_ball_dist']):
                if col in ps.columns:
                    vals = ps[col].dropna()
                    if len(vals) > 0:
                        feats[idx, off + i] = vals.mean()
            if 'is_contact' in ps.columns:
                contacts = pd.to_numeric(ps['is_contact'], errors='coerce')
                feats[idx, off + 4] = (contacts > 0).mean()

    for c in range(feats.shape[1]):
        mx = np.abs(feats[:, c]).max()
        if mx > 0:
            feats[:, c] /= mx
    return feats


def compute_win_prob_targets(feat: pd.DataFrame, game_data: np.ndarray) -> np.ndarray:
    n = len(feat)
    targets = np.full(n, 0.5, dtype=np.float32)
    score_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3.5}
    for idx, row in feat.iterrows():
        p1_pts = score_map.get(int(row['p1_score']), 0)
        p2_pts = score_map.get(int(row['p2_score']), 0)
        g1, g2 = game_data[idx, 0], game_data[idx, 1]
        s1, s2 = game_data[idx, 2], game_data[idx, 3]
        adv = (s1 * 24 + g1 * 2 + p1_pts * 0.5) - (s2 * 24 + g2 * 2 + p2_pts * 0.5)
        targets[idx] = 1.0 / (1.0 + np.exp(-adv / 8.0))
    return np.clip(targets, 0.01, 0.99)

# Mirroring (train only)
def mirror_batch(X, labels, win_probs, n_base=22):
    Xm = X.copy()
    nf = X.shape[2]
    for a, b in [(0, 2), (1, 3), (6, 8), (7, 9), (12, 13), (20, 21)]:
        if a < nf and b < nf:
            Xm[:, :, a], Xm[:, :, b] = X[:, :, b].copy(), X[:, :, a].copy()
    off = n_base
    if off < nf: Xm[:, :, off] = 1.0 - X[:, :, off]
    off += 1
    if off + 3 < nf:
        Xm[:, :, off], Xm[:, :, off+1] = X[:, :, off+1].copy(), X[:, :, off].copy()
        Xm[:, :, off+3] = 1.0 - X[:, :, off+3]
    off += 4
    if off + 3 < nf:
        Xm[:, :, off], Xm[:, :, off+1] = X[:, :, off+1].copy(), X[:, :, off].copy()
        Xm[:, :, off+2], Xm[:, :, off+3] = X[:, :, off+3].copy(), X[:, :, off+2].copy()
    off += 4
    for i in range(5):
        a, b = off + i, off + 5 + i
        if a < nf and b < nf:
            Xm[:, :, a], Xm[:, :, b] = X[:, :, b].copy(), X[:, :, a].copy()
    return (
        np.concatenate([X, Xm], axis=0),
        np.concatenate([labels, 1 - labels]),
        np.concatenate([win_probs, 1.0 - win_probs]),
    )

# Model
def build_model(input_shape):
    inp = layers.Input(shape=input_shape, name='rally_input')
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True, name='shared_lstm_1')(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(64, return_sequences=True, name='shared_lstm_2')(x)
    x = layers.Dropout(DROPOUT)(x)
    shared = layers.LSTM(32, return_sequences=False, name='shared_lstm_3')(x)
    shared = layers.Dropout(DROPOUT)(shared)

    r = layers.Dense(24, activation='relu', name='rally_dense')(shared)
    r = layers.Dropout(0.2)(r)
    rally_out = layers.Dense(2, activation='softmax', name='rally_winner')(r)

    w = layers.Dense(24, activation='relu', name='win_dense')(shared)
    w = layers.Dropout(0.2)(w)
    win_out = layers.Dense(1, activation='sigmoid', name='win_prob')(w)

    return Model(inputs=inp, outputs=[rally_out, win_out])

# Train One Match
def train_match(vid: str, epochs: int, dry_run: bool):
    p = paths_for(vid)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {vid.upper()}")
    print(f"{'=' * 60}")

    data = np.load(p['sequences'])
    X_raw = data['sequences'].astype(np.float32)
    labels = data['labels']
    feat = pd.read_csv(p['features'])
    n = len(labels)
    n_base = X_raw.shape[2]

    print(f"  Rallies: {n} | P1: {np.sum(labels==1)} | P2: {np.sum(labels==0)}")

    # Features
    rf_probs = np.full(n, 0.5, dtype=np.float32)
    if os.path.exists(p['rf_oof']):
        rf = pd.read_csv(p['rf_oof']).sort_values('Rally_ID').reset_index(drop=True)
        rf_probs = rf['P1_Win_Prob'].values[:n].astype(np.float32)
        print(f"  RF OOF: mean={rf_probs.mean():.3f}")

    match_ctx = compute_match_context(labels)
    score_ctx, game_data = load_score_context(feat, p['scores'])
    shot_feats = load_shot_features(feat, p['shot'], p['scores'])
    win_probs = compute_win_prob_targets(feat, game_data)

    X = np.concatenate([
        X_raw,
        np.tile(rf_probs[:, None, None], (1, X_raw.shape[1], 1)),
        np.tile(match_ctx[:, None, :], (1, X_raw.shape[1], 1)),
        np.tile(score_ctx[:, None, :], (1, X_raw.shape[1], 1)),
        np.tile(shot_feats[:, None, :], (1, X_raw.shape[1], 1)),
    ], axis=2)

    print(f"  Input shape: {X.shape} | Win prob: [{win_probs.min():.3f}, {win_probs.max():.3f}]")

    rally_onehot = np.zeros((n, 2), dtype=np.float32)
    rally_onehot[labels == 1, 0] = 1.0
    rally_onehot[labels == 0, 1] = 1.0

    # CV
    min_class = min(np.sum(labels == 0), np.sum(labels == 1))
    n_folds = min(N_FOLDS, min_class)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_rally = np.full((n, 2), -1.0)
    oof_wp = np.full(n, -1.0)
    fold_accs = []

    for fold, (tr, va) in enumerate(cv.split(X, labels)):
        X_tr, lbl_tr, wp_tr = mirror_batch(X[tr], labels[tr], win_probs[tr], n_base)
        r_tr = np.zeros((len(lbl_tr), 2), dtype=np.float32)
        r_tr[lbl_tr == 1, 0] = 1.0
        r_tr[lbl_tr == 0, 1] = 1.0

        model = build_model((X.shape[1], X.shape[2]))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss={'rally_winner': 'categorical_crossentropy', 'win_prob': 'binary_crossentropy'},
            loss_weights={'rally_winner': RALLY_LOSS_W, 'win_prob': WIN_PROB_LOSS_W},
            metrics={'rally_winner': 'accuracy', 'win_prob': 'mae'},
        )

        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ]

        n_tr = len(lbl_tr)
        print(f"\n  Fold {fold+1}/{n_folds} (train={n_tr}, val={len(va)})")

        model.fit(
            X_tr, {'rally_winner': r_tr, 'win_prob': wp_tr},
            validation_data=(X[va], {'rally_winner': rally_onehot[va], 'win_prob': win_probs[va]}),
            epochs=epochs, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0,
        )

        ev = model.evaluate(X[va], {'rally_winner': rally_onehot[va], 'win_prob': win_probs[va]}, verbose=0)
        acc = ev[3]
        mae = ev[4]
        print(f"    Rally acc: {acc:.1%} | Win prob MAE: {mae:.4f}")
        fold_accs.append(acc)

        preds = model.predict(X[va], verbose=0)
        oof_rally[va] = preds[0]
        oof_wp[va] = preds[1].flatten()

        if dry_run:
            print("  [DRY RUN] 1 fold only.")
            break

    # OOF results
    if not dry_run:
        valid = oof_rally[:, 0] >= 0
        pred_cls = (np.argmax(oof_rally[valid], axis=1) == 0).astype(int)
        actual = labels[valid]
        oof_acc = (pred_cls == actual).mean()
        mean_acc = np.mean(fold_accs)

        print(f"\n  {'─' * 40}")
        print(f"  {vid.upper()} RESULTS")
        print(f"  {'─' * 40}")
        print(f"  Mean CV accuracy: {mean_acc:.1%}")
        print(f"  OOF accuracy:     {oof_acc:.1%}")
        print(f"  Fold accuracies:  {[f'{a:.1%}' for a in fold_accs]}")
        print(f"\n{classification_report(actual, pred_cls, target_names=['P2 wins', 'P1 wins'])}")

    # Train final model
    print(f"  Training final model...")
    X_full, lbl_full, wp_full = mirror_batch(X, labels, win_probs, n_base)
    r_full = np.zeros((len(lbl_full), 2), dtype=np.float32)
    r_full[lbl_full == 1, 0] = 1.0
    r_full[lbl_full == 0, 1] = 1.0

    final = build_model((X.shape[1], X.shape[2]))
    final.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={'rally_winner': 'categorical_crossentropy', 'win_prob': 'binary_crossentropy'},
        loss_weights={'rally_winner': RALLY_LOSS_W, 'win_prob': WIN_PROB_LOSS_W},
        metrics={'rally_winner': 'accuracy', 'win_prob': 'mae'},
    )
    final.fit(
        X_full, {'rally_winner': r_full, 'win_prob': wp_full},
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)],
    )

    os.makedirs('models', exist_ok=True)
    final.save(p['model'])

    # Save meta
    meta = {
        'model': f'MultiTask_LSTM_{vid.upper()}',
        'match': vid,
        'rallies': int(n),
        'architecture': 'LSTM(128→64→32) + 2 heads',
        'input_shape': list(X.shape[1:]),
        'features': {'base': 22, 'rf': 1, 'ctx': 4, 'score': 4, 'shot': 10, 'total': int(X.shape[2])},
    }
    if not dry_run:
        meta['cv_rally_accuracy'] = float(mean_acc)
        meta['oof_rally_accuracy'] = float(oof_acc)
        meta['fold_accuracies'] = [float(a) for a in fold_accs]

    with open(p['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    # Save OOF
    if not dry_run:
        oof_df = pd.DataFrame({
            'Rally_ID': feat['rally_id'].values,
            'Actual_Winner': ['P1' if w == 1 else 'P2' for w in labels],
            'Predicted_Winner': ['P1' if r[0] > r[1] else 'P2' for r in oof_rally],
            'P1_Rally_Prob': oof_rally[:, 0],
            'P2_Rally_Prob': oof_rally[:, 1],
            'P1_Win_Prob': oof_wp,
            'Correct': [
                (('P1' if r[0] > r[1] else 'P2') == ('P1' if w == 1 else 'P2'))
                for r, w in zip(oof_rally, labels)
            ],
        })
        oof_df.to_csv(p['oof'], index=False)
        print(f"  ✓ OOF → {p['oof']}")

    print(f"  ✓ Model → {p['model']}")
    print(f"  ✓ Meta  → {p['meta']}")

    return float(mean_acc) if not dry_run else 0.0

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--match', type=str, default=None, help='Train specific match only (e.g. v010)')
    args = parser.parse_args()
    epochs = 1 if args.dry_run else args.epochs

    matches = [args.match] if args.match else ALL_MATCHES

    print(f"  Matches: {', '.join(m.upper() for m in matches)}")

    results = {}
    for vid in matches:
        p = paths_for(vid)
        if not os.path.exists(p['sequences']):
            print(f"\n  [{vid.upper()}] Sequences not found, skipping")
            continue
        acc = train_match(vid, epochs, args.dry_run)
        results[vid] = acc

    if not args.dry_run and len(results) > 1:
        print(f"  SUMMARY")
        for vid, acc in results.items():
            print(f"  {vid.upper()}: {acc:.1%}")
        print(f"  Overall: {np.mean(list(results.values())):.1%}")
    print(f"  DONE")

if __name__ == '__main__':
    main()
