import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

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

# ── Check if init_duration_ms is bimodal ──────────────────────────────────────
print("=== Init Duration Distribution Analysis ===")
y = cold['init_duration_ms']
print(f"Percentiles:")
for p in [5,10,25,50,75,90,95,99]:
    print(f"  p{p:02d}: {np.percentile(y, p):.1f} ms")

# Fit GMM to detect regimes
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(y.values.reshape(-1,1))
labels = gmm.predict(y.values.reshape(-1,1))
cold['regime'] = labels

print(f"\nGMM Regime 0: n={sum(labels==0)}, mean={y[labels==0].mean():.1f}, std={y[labels==0].std():.1f}")
print(f"GMM Regime 1: n={sum(labels==1)}, mean={y[labels==1].mean():.1f}, std={y[labels==1].std():.1f}")

# ── What features separate the regimes? ──────────────────────────────────────
print("\n=== Feature means by regime ===")
regime_cols = ['cw_tail_spread','cw_p999_p50_ratio','package_size_kb',
               'dep_count','function_type_enc','time_since_last_inv',
               'simulated_traffic_level','day_of_week','hour_of_day']
print(cold.groupby('regime')[regime_cols].mean().T.to_string())

# ── Train separate regressors per regime ─────────────────────────────────────
print("\n=== Per-Regime Regressor ===")
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
    X, y, reg, test_size=0.2, random_state=42
)

# Baseline — single model
gb_base = GradientBoostingRegressor(n_estimators=486, max_depth=3,
    learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)
gb_base.fit(X_train, y_train)
p_base = gb_base.predict(X_test)
print(f"Baseline (single model) MAE:{mean_absolute_error(y_test,p_base):.2f}  R2:{r2_score(y_test,p_base):.4f}")

# Per-regime models
preds = np.zeros(len(y_test))
for r in [0, 1]:
    mask_tr = r_train == r
    mask_te = r_test  == r
    if mask_tr.sum() < 20:
        preds[mask_te.values] = y_train[mask_tr].mean()
        continue
    m = GradientBoostingRegressor(n_estimators=486, max_depth=3,
        learning_rate=0.024, subsample=0.64, min_samples_leaf=12, random_state=42)
    m.fit(X_train[mask_tr], y_train[mask_tr])
    preds[mask_te.values] = m.predict(X_test[mask_te])
    sub_mae = mean_absolute_error(y_test[mask_te], m.predict(X_test[mask_te]))
    sub_r2  = r2_score(y_test[mask_te], m.predict(X_test[mask_te]))
    print(f"  Regime {r} (n_test={mask_te.sum()})  MAE:{sub_mae:.2f}  R2:{sub_r2:.4f}")

overall_mae  = mean_absolute_error(y_test, preds)
overall_rmse = np.sqrt(mean_squared_error(y_test, preds))
overall_r2   = r2_score(y_test, preds)
print(f"Per-regime combined  MAE:{overall_mae:.2f}  RMSE:{overall_rmse:.2f}  R2:{overall_r2:.4f}")
