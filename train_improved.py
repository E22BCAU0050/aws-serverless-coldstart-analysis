import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('cold_start_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
print(f"Shape: {df.shape}")
print(f"Cold starts: {df['cold_start_flag'].sum()} / {len(df)} ({df['cold_start_flag'].mean()*100:.3f}%)")

# ── 2. FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\nEngineering features...")

# Sort by function identity + time (critical for lag features)
df = df.sort_values(['function_type', 'timestamp']).reset_index(drop=True)

# Time since last invocation (per function type) — strongest cold start predictor
df['time_since_last_inv'] = (
    df.groupby('function_type')['timestamp']
    .diff()
    .dt.total_seconds()
    .fillna(9999)   # first invocation = very stale
)

# Rolling invocation rate (last 10 rows per function)
df['rolling_inv_rate'] = (
    df.groupby('function_type')['cold_start_flag']
    .transform(lambda x: x.rolling(10, min_periods=1).mean())
)

# Memory × package size interaction
df['mem_pkg_interaction'] = df['memory_size_mb'] * df['package_size_kb']

# Encode traffic_phase (currently unused!)
phase_map = {
    'morning_ramp': 0,
    'morning_peak': 1,
    'midday_peak': 2,
    'afternoon': 3
}
df['traffic_phase_enc'] = df['traffic_phase'].map(phase_map).fillna(-1)

# Target encode function_type (mean cold start rate per type)
target_enc = df.groupby('function_type')['cold_start_flag'].mean()
df['function_type_target_enc'] = df['function_type'].map(target_enc)

# Exponential decay of recency (how stale is the function)
df['staleness_decay'] = np.exp(-df['time_since_last_inv'] / 300)  # 5-min half-life

# Drop noisy features identified by ablation
DROP_COLS = ['vpc_flag', 'simulated_traffic_level', 'request_id',
             'timestamp', 'function_type', 'traffic_phase']

FEATURES = [c for c in df.columns if c not in DROP_COLS + 
            ['cold_start_flag', 'duration_ms', 'init_duration_ms',
             'is_tail_p95', 'is_tail_p99']]

print(f"Features ({len(FEATURES)}): {FEATURES}")

# ── 3. SPLIT ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X = df[FEATURES]
y_clf = df['cold_start_flag']               # Stage 1: classify cold vs warm
y_reg = df['init_duration_ms']              # Stage 2: regress init latency

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\nTrain cold starts: {yc_train.sum()} / {len(yc_train)}")
print(f"Test cold starts:  {yc_test.sum()} / {len(yc_test)}")

# ── 4. STAGE 1 — COLD START CLASSIFIER ────────────────────────────────────────
print("\n── Stage 1: Cold Start Classifier ──")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Handle imbalance with scale_pos_weight
neg = (yc_train == 0).sum()
pos = (yc_train == 1).sum()
spw = neg / pos
print(f"scale_pos_weight: {spw:.1f}")

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=spw,      # handles 1648:1 imbalance
    subsample=0.8,
    colsample_bytree=0.8,

    eval_metric='aucpr',       # PR-AUC better than ROC for imbalanced
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, yc_train,
        eval_set=[(X_test, yc_test)],
        verbose=50)

yc_pred = clf.predict(X_test)
yc_prob = clf.predict_proba(X_test)[:, 1]

print("\nClassifier Report:")
print(classification_report(yc_test, yc_pred))
print(f"ROC-AUC: {roc_auc_score(yc_test, yc_prob):.4f}")

# ── 5. STAGE 2 — COLD START REGRESSOR (cold-only subset) ──────────────────────
print("\n── Stage 2: Cold Start Latency Regressor ──")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train regressor ONLY on actual cold start rows
cold_mask_train = yc_train == 1
cold_mask_test  = yc_test  == 1

X_cold_train = X_train[cold_mask_train]
X_cold_test  = X_test[cold_mask_test]
y_cold_train = yr_train[cold_mask_train]
y_cold_test  = yr_test[cold_mask_test]

print(f"Cold-only train: {len(X_cold_train)}, test: {len(X_cold_test)}")

reg = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
reg.fit(X_cold_train, y_cold_train)

yr_pred = reg.predict(X_cold_test)

mae  = mean_absolute_error(y_cold_test, yr_pred)
rmse = np.sqrt(mean_squared_error(y_cold_test, yr_pred))
r2   = r2_score(y_cold_test, yr_pred)
mape = np.mean(np.abs((y_cold_test - yr_pred) / (y_cold_test + 1e-9))) * 100

print(f"\nCold-only Regressor:")
print(f"  MAE:  {mae:.2f} ms")
print(f"  RMSE: {rmse:.2f} ms")
print(f"  R²:   {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# ── 6. OPTUNA TUNING (Gradient Boosting) ──────────────────────────────────────
print("\n── Optuna Tuning ──")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 200, 800),
            'max_depth':       trial.suggest_int('max_depth', 3, 8),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_leaf':trial.suggest_int('min_samples_leaf', 5, 50),
        }
        m = GradientBoostingRegressor(**params, random_state=42)
        m.fit(X_cold_train, y_cold_train)
        return mean_absolute_error(y_cold_test, m.predict(X_cold_test))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"\nBest MAE: {study.best_value:.2f} ms")
    print(f"Best params: {study.best_params}")

    # Retrain with best params
    best_reg = GradientBoostingRegressor(**study.best_params, random_state=42)
    best_reg.fit(X_cold_train, y_cold_train)
    bp = best_reg.predict(X_cold_test)
    print(f"Tuned R²: {r2_score(y_cold_test, bp):.4f}")
    print(f"Tuned MAE: {mean_absolute_error(y_cold_test, bp):.2f} ms")

except ImportError:
    print("Optuna not installed — run: pip install optuna")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
print("\n── Top Feature Importances (Regressor) ──")
importances = pd.Series(reg.feature_importances_, index=FEATURES)
print(importances.sort_values(ascending=False).head(10).to_string())

print("\nDone.")

# ── 8. SMOTE + BETTER REGRESSOR ───────────────────────────────────────────────
print("\n── SMOTE Oversampling + LightGBM ──")
try:
    from imblearn.over_sampling import SMOTE
    import lightgbm as lgb

    # Oversample cold starts to 5000 rows for regressor
    sm = SMOTE(sampling_strategy={1: 5000}, random_state=42, k_neighbors=5)
    X_res, y_res_clf = sm.fit_resample(X_train, yc_train)
    y_res_reg = pd.Series(
        yr_train.values[np.arange(len(X_train))], index=range(len(X_train))
    )

    # Filter to cold-only after resampling
    cold_mask_res = y_res_clf == 1
    X_cold_res = X_res[cold_mask_res]
    y_cold_res = y_res_reg.reindex(range(len(X_res))).fillna(
        yr_train[yc_train==1].mean()
    )[cold_mask_res]

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )
    lgb_reg.fit(X_cold_res, y_cold_res,
                eval_set=[(X_cold_test, y_cold_test)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    lgb_pred = lgb_reg.predict(X_cold_test)
    print(f"LightGBM MAE:  {mean_absolute_error(y_cold_test, lgb_pred):.2f} ms")
    print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_cold_test, lgb_pred)):.2f} ms")
    print(f"LightGBM R²:   {r2_score(y_cold_test, lgb_pred):.4f}")

except ImportError:
    print("Run: pip install imbalanced-learn lightgbm")
