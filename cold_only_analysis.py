import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('cold_start_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
df = df.sort_values(['function_type','timestamp']).reset_index(drop=True)

df['time_since_last_inv'] = (
    df.groupby('function_type')['timestamp']
    .diff().dt.total_seconds().fillna(9999)
)
df['rolling_inv_rate'] = (
    df.groupby('function_type')['cold_start_flag']
    .transform(lambda x: x.rolling(10, min_periods=1).mean())
)
df['staleness_decay'] = np.exp(-df['time_since_last_inv'] / 300)
df['mem_pkg_interaction'] = df['memory_size_mb'] * df['package_size_kb']
phase_map = {'morning_ramp':0,'morning_peak':1,'midday_peak':2,'afternoon':3}
df['traffic_phase_enc'] = df['traffic_phase'].map(phase_map).fillna(-1)
target_enc = df.groupby('function_type')['cold_start_flag'].mean()
df['function_type_target_enc'] = df['function_type'].map(target_enc)

df['cw_p999_p50_ratio'] = df['cw_p999_duration_ms'] / (df['cw_p50_duration_ms'] + 1)
df['cw_p99_p50_ratio']  = df['cw_p99_duration_ms']  / (df['cw_p50_duration_ms'] + 1)
df['cw_tail_spread']    = df['cw_p999_duration_ms']  - df['cw_p50_duration_ms']
df['cw_init_ratio']     = df['cw_avg_init_ms']       / (df['cw_avg_duration_ms'] + 1)

cold = df[df['cold_start_flag']==1].copy()

FEATURES = [
    'cw_p999_duration_ms', 'cw_p99_duration_ms', 'cw_p95_duration_ms',
    'cw_avg_duration_ms', 'cw_p50_duration_ms',
    'day_of_week', 'function_type_enc', 'function_type_target_enc',
    'hour_sin', 'hour_of_day', 'hour_cos',
    'cw_max_concurrent_execs', 'package_size_kb', 'dep_count',
    'simulated_traffic_level',
    'cw_total_invocations',
    'time_since_last_inv', 'rolling_inv_rate', 'staleness_decay',
    'traffic_phase_enc', 'mem_pkg_interaction',
    'cw_p999_p50_ratio', 'cw_p99_p50_ratio', 'cw_tail_spread', 'cw_init_ratio',
    'cw_avg_init_ms', 'cw_p95_init_ms', 'cw_p99_init_ms',
]

X = cold[FEATURES]
y = cold['init_duration_ms']

print(f'Cold-only dataset: {len(cold)} rows, {len(FEATURES)} features')
print(f'Target std: {y.std():.2f}, mean: {y.mean():.2f}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

gb = GradientBoostingRegressor(
    n_estimators=486, max_depth=3,
    learning_rate=0.024, subsample=0.64,
    min_samples_leaf=12, random_state=42
)
gb.fit(X_train, y_train)
p = gb.predict(X_test)
results['GradBoost'] = {
    'MAE': mean_absolute_error(y_test, p),
    'RMSE': np.sqrt(mean_squared_error(y_test, p)),
    'R2': r2_score(y_test, p)
}

xgb_r = xgb.XGBRegressor(
    n_estimators=500, max_depth=4,
    learning_rate=0.02, subsample=0.7,
    colsample_bytree=0.7, random_state=42,
    n_jobs=-1, verbosity=0
)
xgb_r.fit(X_train, y_train)
p = xgb_r.predict(X_test)
results['XGBoost'] = {
    'MAE': mean_absolute_error(y_test, p),
    'RMSE': np.sqrt(mean_squared_error(y_test, p)),
    'R2': r2_score(y_test, p)
}

rf = RandomForestRegressor(
    n_estimators=500, max_depth=8,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
p = rf.predict(X_test)
results['RandomForest'] = {
    'MAE': mean_absolute_error(y_test, p),
    'RMSE': np.sqrt(mean_squared_error(y_test, p)),
    'R2': r2_score(y_test, p)
}

print('\n── Results on Cold-Start-Only Subset ──')
for name, m in results.items():
    print(f'{name:15s}  MAE:{m["MAE"]:6.2f}  RMSE:{m["RMSE"]:6.2f}  R2:{m["R2"]:.4f}')

print('\n── Feature Importances (GradBoost) ──')
imp = pd.Series(gb.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(12).to_string())

print('\n── Feature Importances (XGBoost) ──')
imp_xgb = pd.Series(xgb_r.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp_xgb.head(12).to_string())
