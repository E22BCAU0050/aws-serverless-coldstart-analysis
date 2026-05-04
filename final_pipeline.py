import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=== Loading and Engineering Features ===")
df = pd.read_csv('cold_start_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
df = df.sort_values(['function_type','timestamp']).reset_index(drop=True)

df['time_since_last_inv'] = df.groupby('function_type')['timestamp'].diff().dt.total_seconds().fillna(9999)
df['rolling_inv_rate'] = df.groupby('function_type')['cold_start_flag'].transform(lambda x: x.rolling(10, min_periods=1).mean())
df['staleness_decay'] = np.exp(-df['time_since_last_inv'] / 300)
df['mem_pkg_interaction'] = df['memory_size_mb'] * df['package_size_kb']
df['traffic_phase_enc'] = df['traffic_phase'].map({'morning_ramp':0,'morning_peak':1,'midday_peak':2,'afternoon':3}).fillna(-1)
df['function_type_target_enc'] = df.groupby('function_type')['cold_start_flag'].transform('mean')
df['cw_p999_p50_ratio'] = df['cw_p999_duration_ms'] / (df['cw_p50_duration_ms'] + 1)
df['cw_p99_p50_ratio']  = df['cw_p99_duration_ms']  / (df['cw_p50_duration_ms'] + 1)
df['cw_tail_spread']    = df['cw_p999_duration_ms']  - df['cw_p50_duration_ms']
df['cw_init_ratio']     = df['cw_avg_init_ms']       / (df['cw_avg_duration_ms'] + 1)

cold = df[df['cold_start_flag']==1].copy()

# GMM regime detection
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(cold['init_duration_ms'].values.reshape(-1,1))
cold['regime'] = gmm.predict(cold['init_duration_ms'].values.reshape(-1,1))

FEATURES = [
    'cw_p999_duration_ms','cw_p99_duration_ms','cw_p95_duration_ms',
    'cw_avg_duration_ms','cw_p50_duration_ms',
    'day_of_week','function_type_enc','function_type_target_enc',
    'hour_sin','hour_of_day','hour_cos',
    'cw_max_concurrent_execs','package_size_kb','dep_count',
    'simulated_traffic_level','cw_total_invocations',
    'time_since_last_inv','rolling_inv_rate','staleness_decay',
    'traffic_phase_enc','mem_pkg_interaction',
    'cw_p999_p50_ratio','cw_p99_p50_ratio','cw_tail_spread','cw_init_ratio',
    'cw_avg_init_ms','cw_p95_init_ms','cw_p99_init_ms',
]

X = cold[FEATURES]
y = cold['init_duration_ms']
reg = cold['regime']

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, reg, test_size=0.2, random_state=42, stratify=reg
)

print(f"Cold-only: {len(cold)} total, {len(X_train)} train, {len(X_test)} test")
print(f"Regime 0 train/test: {(r_train==0).sum()}/{(r_test==0).sum()}")
print(f"Regime 1 train/test: {(r_train==1).sum()}/{(r_test==1).sum()}")

# ── Baseline models (single, no regime split) ─────────────────────────────────
print("\n=== Baseline Models (no regime split) ===")
baselines = {
    'Ridge': Ridge(alpha=1.0),
    'GradBoost': GradientBoostingRegressor(n_estimators=486, max_depth=3,
        learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=500, max_depth=4,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbosity=0),
    'RandomForest': RandomForestRegressor(n_estimators=500, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1),
}
baseline_results = {}
for name, model in baselines.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    baseline_results[name] = {
        'MAE':  mean_absolute_error(y_test, p),
        'RMSE': np.sqrt(mean_squared_error(y_test, p)),
        'R2':   r2_score(y_test, p),
        'MAPE': np.mean(np.abs((y_test - p) / (y_test + 1e-9))) * 100
    }
    print(f"  {name:15s} MAE:{baseline_results[name]['MAE']:6.2f}  RMSE:{baseline_results[name]['RMSE']:6.2f}  R2:{baseline_results[name]['R2']:.4f}  MAPE:{baseline_results[name]['MAPE']:.2f}%")

# ── Regime-aware models ───────────────────────────────────────────────────────
print("\n=== Regime-Aware Two-Stage Models ===")
def regime_predict(model_class, model_kwargs, X_tr, y_tr, r_tr, X_te, r_te):
    preds = np.zeros(len(X_te))
    for r in [0, 1]:
        mask_tr = r_tr == r
        mask_te = r_te == r
        m = model_class(**model_kwargs)
        m.fit(X_tr[mask_tr], y_tr[mask_tr])
        preds[mask_te.values] = m.predict(X_te[mask_te])
    return preds

regime_models = {
    'GradBoost+Regime': (GradientBoostingRegressor, dict(n_estimators=486, max_depth=3,
        learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)),
    'XGBoost+Regime': (xgb.XGBRegressor, dict(n_estimators=500, max_depth=4,
        learning_rate=0.02, subsample=0.7, colsample_bytree=0.7,
        random_state=42, n_jobs=-1, verbosity=0)),
    'RandomForest+Regime': (RandomForestRegressor, dict(n_estimators=500, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1)),
}
regime_results = {}
for name, (cls, kwargs) in regime_models.items():
    p = regime_predict(cls, kwargs, X_train, y_train, r_train, X_test, r_test)
    regime_results[name] = {
        'MAE':  mean_absolute_error(y_test, p),
        'RMSE': np.sqrt(mean_squared_error(y_test, p)),
        'R2':   r2_score(y_test, p),
        'MAPE': np.mean(np.abs((y_test - p) / (y_test + 1e-9))) * 100
    }
    print(f"  {name:25s} MAE:{regime_results[name]['MAE']:6.2f}  RMSE:{regime_results[name]['RMSE']:6.2f}  R2:{regime_results[name]['R2']:.4f}  MAPE:{regime_results[name]['MAPE']:.2f}%")

# ── Per-regime breakdown ──────────────────────────────────────────────────────
print("\n=== Per-Regime Breakdown (GradBoost+Regime) ===")
for r in [0, 1]:
    mask_tr = r_train == r
    mask_te = r_test  == r
    m = GradientBoostingRegressor(n_estimators=486, max_depth=3,
        learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)
    m.fit(X_train[mask_tr], y_train[mask_tr])
    p = m.predict(X_test[mask_te])
    print(f"  Regime {r} (n={mask_te.sum()})  MAE:{mean_absolute_error(y_test[mask_te],p):.2f}  R2:{r2_score(y_test[mask_te],p):.4f}  mean_init:{y_test[mask_te].mean():.1f}ms")

# ── Feature importance ────────────────────────────────────────────────────────
print("\n=== Top Feature Importances (GradBoost, Regime 0) ===")
m0 = GradientBoostingRegressor(n_estimators=486, max_depth=3,
    learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)
m0.fit(X_train[r_train==0], y_train[r_train==0])
imp = pd.Series(m0.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(10).to_string())

print("\n=== Top Feature Importances (GradBoost, Regime 1) ===")
m1 = GradientBoostingRegressor(n_estimators=486, max_depth=3,
    learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)
m1.fit(X_train[r_train==1], y_train[r_train==1])
imp1 = pd.Series(m1.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp1.head(10).to_string())

# ── Summary table for paper ───────────────────────────────────────────────────
print("\n=== PAPER TABLE: Model Comparison ===")
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE':>8}")
print("-" * 60)
all_results = {**baseline_results, **regime_results}
for name, m in all_results.items():
    print(f"{name:<25} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.4f} {m['MAPE']:>7.2f}%")
