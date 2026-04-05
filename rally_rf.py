import pandas as pd
import numpy as np
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import json
import joblib

# Accept match ID from command line: python src/08a_rally_rf.py v007
MATCH_ID = sys.argv[1] if len(sys.argv) > 1 else 'v010'
print(f"\n=== Rally RF for {MATCH_ID.upper()} ===")

FEATURES_CSV = f'rally_data/rally_features_{MATCH_ID}.csv'
MODEL_SAVE = f'models/rally_best_{MATCH_ID}.json'
MODEL_PICKLE = f'models/rally_best_{MATCH_ID}.pkl'

FEATURE_COLS = [
    'rally_length', 'n_shots', 'server',
    'p1_distance', 'p2_distance', 'dist_ratio',
    'p1_avg_speed', 'p2_avg_speed', 'p1_max_speed', 'p2_max_speed', 'late_speed_diff',
    'p1_avg_accel', 'p2_avg_accel', 'ball_avg_speed', 'accel_diff',
    'p1_net_proximity', 'p2_net_proximity', 'net_diff',
    'p1_court_coverage', 'p2_court_coverage',
    'p1_late_spread', 'p2_late_spread', 'speed_momentum',
    'p1_score', 'p2_score',
]

def train_and_evaluate():
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: {FEATURES_CSV} not found!")
        return
    
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rallies")
    
    X = df[FEATURE_COLS].values
    y = df['winner'].values
    
    print(f"Features: {X.shape[1]}")
    print(f"P1 wins: {np.sum(y == 1)} | P2 wins: {np.sum(y == 0)}")
    
    min_class_count = min(np.sum(y == 1), np.sum(y == 0))
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate scale_pos_weight for XGBoost to handle class imbalance natively
    # scale_pos_weight = sum(negative instances) / sum(positive instances)
    # y=0 is P2 (25), y=1 is P1 (48). So scale_pos_weight = 25 / 48 = ~0.52
    scale_pos = np.sum(y == 0) / np.sum(y == 1)

    # Define pipelines with feature selection (to prevent the curse of dimensionality)
    pipelines = {
        'RandomForest': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)),
                ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            {
                'selector__k': [10, 15, 18, 'all'],
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7]
            }
        ),
    }

    print(f"\nSearching for >70% Accuracy using Pipeline GridSearch ({n_splits}-fold CV) ")
    
    best_name = None
    best_score = 0.0
    best_model = None
    best_params = None
    
    results = {}
    
    for name, (pipeline, params) in pipelines.items():
        # n_jobs=1: Windows + multiprocessing + XGBoost = access violations
        grid = GridSearchCV(pipeline, params, cv=cv, scoring='accuracy', n_jobs=1)
        grid.fit(X, y)
        
        mean_acc = grid.best_score_
        print(f"{name:20s}: {mean_acc:.1%} (Best Params: {grid.best_params_})")
        
        results[name] = float(mean_acc)
        
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            
    print(f"** BEST MODEL: {best_name} ({best_score:.1%}) **")
    
    # Train best model on full data
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    
    print("\nClassification Report (Full Data)")
    print(classification_report(y, y_pred, target_names=['P2 wins', 'P1 wins']))
    
    # Extract selected features
    selector = best_model.named_steps['selector']
    if selector.k != 'all':
        mask = selector.get_support()
        selected_features = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if mask[i]]
        print(f"\nTop {selector.k} Features Selected")
        for f in selected_features:
            print(f"  - {f}")
            
    # Save Model Metadata
    os.makedirs('models', exist_ok=True)
    with open(MODEL_SAVE, 'w') as f:
        json.dump({'model': best_name, 'accuracy': float(best_score), 'best_params': best_params, 'all_results': results, 'feature_cols': FEATURE_COLS}, f, indent=2)

    # Save trained model (for predicting on NEW data)
    joblib.dump(best_model, MODEL_PICKLE)
    print(f"\n✓ Trained model saved to {MODEL_PICKLE}")

    # Out-of-Fold Predictions (honest evaluation) 
    print("\nGenerating Out-of-Fold Predictions")
    best_pipeline, best_params_dict = pipelines[best_name]
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_probs = np.full(len(y), -1.0, dtype=float)

    for train_idx, val_idx in cv.split(X, y):
        fold_pipeline, fold_params = pipelines[best_name]
        fold_pipeline.set_params(**best_params)
        fold_pipeline.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = fold_pipeline.predict(X[val_idx])
        oof_probs[val_idx] = fold_pipeline.predict_proba(X[val_idx])[:, 1]

    oof_accuracy = (oof_preds == y).mean()
    print(f"Out-of-fold accuracy: {oof_accuracy:.1%} ({(oof_preds == y).sum()}/{len(y)})")

    oof_df = pd.DataFrame({
        'Rally_ID': df['rally_id'].values,
        'Actual_Winner': ['P1' if win == 1 else 'P2' for win in y],
        'Predicted_Winner': ['P1' if pred == 1 else 'P2' for pred in oof_preds],
        'P1_Win_Prob': oof_probs,
        'Correct': (oof_preds == y),
        'Server': ['P1' if s == 1 else 'P2' for s in df['server'].values],
    })

    oof_csv = f"models/rally_predictions_oof_{MATCH_ID}.csv"
    oof_df.to_csv(oof_csv, index=False)
    print(f"Out-of-fold predictions saved to {oof_csv}")

    # Output detailed rally predictions to CSV for Phase 3: Match Scaling
    print("\nGenerating Point-by-Point Playback (Full Data)")
    
    # We need the rally_id's to map this properly
    rally_ids = df['rally_id'].values
    servers = df['server'].values
    
    predictions_df = pd.DataFrame({
        'Rally_ID': rally_ids,
        'Server': ['P1' if s == 1 else 'P2' for s in servers],
        'Actual_Winner': ['P1' if win == 1 else 'P2' for win in y],
        'Predicted_Winner': ['P1' if pred == 1 else 'P2' for pred in y_pred],
        'Correct': (y == y_pred)
    })
    
    csv_out = f'models/rally_predictions_{MATCH_ID}.csv'
    predictions_df.to_csv(csv_out, index=False)
    print(f"Point-by-point predictions saved to {csv_out}")
    print(f"(True Accuracy on Full Data: {predictions_df['Correct'].mean():.1%})")

if __name__ == '__main__':
    train_and_evaluate()
