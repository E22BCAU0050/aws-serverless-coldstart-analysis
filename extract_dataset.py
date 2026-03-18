#!/usr/bin/env python3
"""
extract_dataset.py
──────────────────────────────────────────────────────────────────
Extracts cold start metrics from DynamoDB + CloudWatch into a
structured CSV dataset for ML/DL training.

New features vs v1:
  - Tail latency: p95, p99, p999 per function type
  - Package size + dependency count as features
  - Sine wave hour-of-day encoding (sin/cos for cyclical features)
  - VPC flag, provisioned flag, container flag
  - CloudWatch aggregates: init duration, concurrent executions
  - Cross-variant comparison rows

Usage:
  python extract_dataset.py
  python extract_dataset.py --hours 48 --output my_dataset.csv
  python extract_dataset.py --synthetic  # generate synthetic data only
"""

import boto3
import csv
import json
import math
import random
import argparse
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────
PROJECT      = 'coldstart-research'
REGION       = 'us-east-1'
METRICS_TABLE = f'{PROJECT}-metrics'
OUTPUT_CSV    = 'cold_start_dataset.csv'

FUNCTION_TYPES = ['non-vpc', 'vpc', 'provisioned', 'small-pkg', 'large-pkg']

FUNCTION_META = {
    'non-vpc':     {'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0, 'package_size_kb': 48,  'dep_count': 3,  'enc': 0},
    'vpc':         {'vpc_flag': 1, 'provisioned_flag': 0, 'container_flag': 0, 'package_size_kb': 48,  'dep_count': 3,  'enc': 1},
    'provisioned': {'vpc_flag': 0, 'provisioned_flag': 1, 'container_flag': 0, 'package_size_kb': 48,  'dep_count': 3,  'enc': 2},
    'small-pkg':   {'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0, 'package_size_kb': 32,  'dep_count': 1,  'enc': 3},
    'large-pkg':   {'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0, 'package_size_kb': 512, 'dep_count': 15, 'enc': 4},
}

API_METHOD_ENC = {'GET': 0, 'POST': 1, 'PUT': 2, 'DELETE': 3}

CSV_COLUMNS = [
    # ── Identifiers
    'request_id', 'timestamp', 'function_type',
    # ── Raw measurements
    'duration_ms', 'init_duration_ms', 'cold_start_flag',
    # ── Features: function config
    'memory_size_mb', 'vpc_flag', 'provisioned_flag', 'container_flag',
    'package_size_kb', 'dep_count',
    # ── Features: request context
    'hour_of_day', 'day_of_week', 'hour_sin', 'hour_cos',
    'function_type_enc', 'api_method_enc',
    # ── CloudWatch aggregates (per function, per 5-min window)
    'cw_avg_duration_ms', 'cw_p50_duration_ms', 'cw_p95_duration_ms',
    'cw_p99_duration_ms', 'cw_p999_duration_ms',
    'cw_avg_init_ms', 'cw_p95_init_ms', 'cw_p99_init_ms',
    'cw_total_invocations', 'cw_max_concurrent_execs', 'cw_error_count',
    # ── Tail latency flags (derived)
    'is_tail_p95', 'is_tail_p99',
    # ── Traffic context (from sine wave model)
    'simulated_traffic_level', 'traffic_phase',
]


# ── SINE WAVE TRAFFIC MODEL ─────────────────────────────────────
def sine_wave_traffic(hour: float) -> float:
    """Returns 0.0–1.0 representing traffic level at given hour."""
    primary   = math.sin((hour - 7) * math.pi / 12)
    secondary = 0.4 * math.sin((hour - 13) * math.pi / 8)
    return max(0.05, min(1.0, primary + secondary))


def traffic_phase(hour: float) -> str:
    """Label the time-of-day traffic phase."""
    if   hour < 6:   return 'night'
    elif hour < 9:   return 'morning_ramp'
    elif hour < 12:  return 'morning_peak'
    elif hour < 14:  return 'midday_peak'
    elif hour < 17:  return 'afternoon'
    elif hour < 20:  return 'evening_peak'
    else:            return 'night'


def hour_encoding(hour: float):
    """Cyclical encoding of hour — avoids 23→0 discontinuity."""
    return (
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
    )


# ── CLOUDWATCH HELPERS ──────────────────────────────────────────
def get_cw_metric(cw, fn_name: str, metric: str, stat: str,
                  start: datetime, end: datetime, period: int = 300) -> float:
    """Fetch a single CloudWatch metric statistic."""
    try:
        std_stats = ('Average','Sum','Maximum','Minimum','SampleCount')
        is_ext = stat not in std_stats
        kwargs = dict(
            Namespace='AWS/Lambda', MetricName=metric,
            Dimensions=[{'Name': 'FunctionName', 'Value': fn_name}],
            StartTime=start, EndTime=end, Period=period,
        )
        if is_ext:
            kwargs['ExtendedStatistics'] = [stat]
        else:
            kwargs['Statistics'] = [stat]
        resp   = cw.get_metric_statistics(**kwargs)
        points = resp.get('Datapoints', [])
        if not points:
            return 0.0
        points.sort(key=lambda x: x['Timestamp'])
        latest = points[-1]
        if is_ext:
            return float(latest.get('ExtendedStatistics', {}).get(stat, 0))
        else:
            return float(latest.get(stat, 0))
    except Exception as e:
        print(f'  [cw_warn] {fn_name}/{metric}/{stat}: {e}')
        return 0.0


def fetch_cw_aggregates(cw, fn_name: str, start: datetime, end: datetime) -> dict:
    """Fetch all CloudWatch aggregates for a function over a time window."""
    fn_label = f'coldstart-research-api-{fn_name}'
    # Auto-scale period to stay under CloudWatch 1440 datapoint limit
    hours  = (end - start).total_seconds() / 3600
    period = max(300, int(math.ceil(hours * 3600 / 1440 / 60)) * 60)
    return {
        'cw_avg_duration_ms':     get_cw_metric(cw, fn_label, 'Duration', 'Average', start, end, period),
        'cw_p50_duration_ms':     get_cw_metric(cw, fn_label, 'Duration', 'p50', start, end, period),
        'cw_p95_duration_ms':     get_cw_metric(cw, fn_label, 'Duration', 'p95', start, end, period),
        'cw_p99_duration_ms':     get_cw_metric(cw, fn_label, 'Duration', 'p99', start, end, period),
        'cw_p999_duration_ms':    get_cw_metric(cw, fn_label, 'Duration', 'p99.9', start, end, period),
        'cw_avg_init_ms':         get_cw_metric(cw, fn_label, 'InitDuration', 'Average', start, end, period),
        'cw_p95_init_ms':         get_cw_metric(cw, fn_label, 'InitDuration', 'p95', start, end, period),
        'cw_p99_init_ms':         get_cw_metric(cw, fn_label, 'InitDuration', 'p99', start, end, period),
        'cw_total_invocations':   get_cw_metric(cw, fn_label, 'Invocations', 'Sum', start, end, period),
        'cw_max_concurrent_execs':get_cw_metric(cw, fn_label, 'ConcurrentExecutions','Maximum',start, end),
        'cw_error_count':         get_cw_metric(cw, fn_label, 'Errors', 'Sum', start, end, period),
    }


# ── DYNAMODB SCAN ───────────────────────────────────────────────
def scan_metrics_table(dynamodb, hours_back: int) -> list:
    """Scan DynamoDB metrics table for recent records."""
    table     = dynamodb.Table(METRICS_TABLE)
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    records   = []
    kwargs    = {'FilterExpression': boto3.dynamodb.conditions.Attr('timestamp').gte(cutoff_ts)}

    print(f'  Scanning {METRICS_TABLE} (last {hours_back}h)...')
    while True:
        resp = table.scan(**kwargs)
        records.extend(resp.get('Items', []))
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']

    print(f'  Found {len(records)} records')
    return records


# ── PER-FUNCTION PERCENTILE COMPUTATION ─────────────────────────
def compute_percentiles(values: list) -> dict:
    """Compute p50/p95/p99/p999 from a list of values."""
    if not values:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = max(0, min(n-1, int(math.ceil(p/100 * n)) - 1))
        return s[idx]
    return {'p50': pct(50), 'p95': pct(95), 'p99': pct(99), 'p999': pct(99.9)}


# ── REAL DATA EXTRACTION ────────────────────────────────────────
def extract_real_data(hours_back: int) -> list:
    print('\n[extract] Connecting to AWS...')
    session  = boto3.Session(region_name=REGION)
    dynamodb = session.resource('dynamodb')
    cw       = session.client('cloudwatch')

    now   = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)

    # Fetch DynamoDB records
    records = scan_metrics_table(dynamodb, hours_back)
    if not records:
        print('  [warn] No DynamoDB records found — try running ./run_all.sh test first')
        return []

    # Group records by function type for percentile computation
    by_fn = defaultdict(list)
    for r in records:
        fn = str(r.get('functionType', 'unknown'))
        by_fn[fn].append(float(r.get('durationMs', 0)))

    fn_percentiles = {fn: compute_percentiles(vals) for fn, vals in by_fn.items()}

    # Fetch CW aggregates per function (cache — one fetch per function)
    print('  Fetching CloudWatch aggregates...')
    cw_cache = {}
    for fn in FUNCTION_TYPES:
        cw_cache[fn] = fetch_cw_aggregates(cw, fn, start, now)
        print(f'    {fn}: avg={cw_cache[fn]["cw_avg_duration_ms"]:.1f}ms  '
              f'p95={cw_cache[fn]["cw_p95_duration_ms"]:.1f}ms  '
              f'invocations={cw_cache[fn]["cw_total_invocations"]:.0f}')

    # Build dataset rows
    rows = []
    for r in records:
        fn       = str(r.get('functionType', 'unknown'))
        meta     = FUNCTION_META.get(fn, FUNCTION_META['non-vpc'])
        ts_str   = str(r.get('timestamp', ''))
        duration = float(r.get('durationMs', 0))
        init_ms  = float(r.get('initDurationMs', 0))
        cold     = bool(r.get('isColdStart', False))

        try:
            ts  = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            hod = ts.hour + ts.minute / 60
            dow = ts.weekday()
        except Exception:
            hod, dow = 12.0, 0

        pcts  = fn_percentiles.get(fn, {'p50':0,'p95':0,'p99':0,'p999':0})
        h_sin, h_cos = hour_encoding(hod)
        cw    = cw_cache.get(fn, {k: 0 for k in ['cw_avg_duration_ms','cw_p50_duration_ms',
                    'cw_p95_duration_ms','cw_p99_duration_ms','cw_p999_duration_ms',
                    'cw_avg_init_ms','cw_p95_init_ms','cw_p99_init_ms',
                    'cw_total_invocations','cw_max_concurrent_execs','cw_error_count']})

        rows.append({
            'request_id':          str(r.get('requestId', '')),
            'timestamp':           ts_str,
            'function_type':       fn,
            'duration_ms':         round(duration, 3),
            'init_duration_ms':    round(init_ms, 3),
            'cold_start_flag':     int(cold),
            'memory_size_mb':      int(r.get('memoryMB', 128)),
            'vpc_flag':            meta['vpc_flag'],
            'provisioned_flag':    meta['provisioned_flag'],
            'container_flag':      meta['container_flag'],
            'package_size_kb':     int(r.get('packageSizeKb', meta['package_size_kb'])),
            'dep_count':           int(r.get('depCount', meta['dep_count'])),
            'hour_of_day':         round(hod, 4),
            'day_of_week':         dow,
            'hour_sin':            round(h_sin, 6),
            'hour_cos':            round(h_cos, 6),
            'function_type_enc':   meta['enc'],
            'api_method_enc':      API_METHOD_ENC.get(str(r.get('method', 'GET')), 0),
            **cw,
            'is_tail_p95':         int(duration > pcts['p95']),
            'is_tail_p99':         int(duration > pcts['p99']),
            'simulated_traffic_level': round(sine_wave_traffic(hod), 4),
            'traffic_phase':       traffic_phase(hod),
        })

    print(f'\n  Built {len(rows)} dataset rows from real data')
    return rows


# ── SYNTHETIC DATA GENERATION ───────────────────────────────────
def generate_synthetic_data(n_samples: int = 2000) -> list:
    """
    Generate realistic synthetic cold start data for model development
    when real AWS data isn't available yet.
    Uses distributions calibrated from published Lambda benchmarks.
    """
    print(f'\n[synthetic] Generating {n_samples} synthetic samples...')
    rng   = random.Random(42)
    rows  = []

    # Cold start distributions (mean, std) per function type (ms)
    COLD_START_DIST = {
        'non-vpc':     (280,  80),
        'vpc':         (950, 200),
        'provisioned': (12,    5),
        'small-pkg':   (180,  50),
        'large-pkg':   (620, 150),
    }
    WARM_START_DIST = {
        'non-vpc':     (8,   3),
        'vpc':         (12,  4),
        'provisioned': (7,   2),
        'small-pkg':   (6,   2),
        'large-pkg':   (15,  5),
    }

    for i in range(n_samples):
        fn    = rng.choice(FUNCTION_TYPES)
        meta  = FUNCTION_META[fn]
        hod   = rng.uniform(0, 24)
        dow   = rng.randint(0, 6)
        traffic = sine_wave_traffic(hod)

        # Cold start probability influenced by traffic (low traffic → more cold starts)
        base_cold_prob = {'non-vpc': 0.15, 'vpc': 0.12, 'provisioned': 0.01,
                          'small-pkg': 0.18, 'large-pkg': 0.10}
        cold_prob = base_cold_prob[fn] * (1 + (1 - traffic) * 0.5)
        cold      = rng.random() < cold_prob

        if cold:
            mu, sigma = COLD_START_DIST[fn]
            init_ms   = max(1.0, rng.gauss(mu, sigma))
            duration  = init_ms + max(1.0, rng.gauss(WARM_START_DIST[fn][0], WARM_START_DIST[fn][1]))
        else:
            init_ms  = 0.0
            mu, sigma = WARM_START_DIST[fn]
            duration  = max(1.0, rng.gauss(mu, sigma))

        # Occasional tail latency spike (simulates GC pause, throttling)
        if rng.random() < 0.02:
            duration *= rng.uniform(3, 10)

        h_sin, h_cos = hour_encoding(hod)
        ts = datetime.now(timezone.utc) - timedelta(hours=rng.uniform(0, 24))

        # Synthetic CW aggregates (slightly noisier than raw)
        cw_avg  = rng.gauss(WARM_START_DIST[fn][0], WARM_START_DIST[fn][1] * 2)
        cw_p95  = cw_avg * rng.uniform(2.5, 5.0) if cold else cw_avg * rng.uniform(1.5, 3.0)
        cw_p99  = cw_p95 * rng.uniform(1.2, 2.0)
        cw_p999 = cw_p99 * rng.uniform(1.5, 3.0)

        rows.append({
            'request_id':           f'synthetic-{i:06d}',
            'timestamp':            ts.isoformat(),
            'function_type':        fn,
            'duration_ms':          round(duration, 3),
            'init_duration_ms':     round(init_ms, 3),
            'cold_start_flag':      int(cold),
            'memory_size_mb':       128,
            'vpc_flag':             meta['vpc_flag'],
            'provisioned_flag':     meta['provisioned_flag'],
            'container_flag':       meta['container_flag'],
            'package_size_kb':      meta['package_size_kb'],
            'dep_count':            meta['dep_count'],
            'hour_of_day':          round(hod, 4),
            'day_of_week':          dow,
            'hour_sin':             round(h_sin, 6),
            'hour_cos':             round(h_cos, 6),
            'function_type_enc':    meta['enc'],
            'api_method_enc':       rng.choice([0, 0, 0, 1]),  # weighted GET
            'cw_avg_duration_ms':   round(max(0, cw_avg), 3),
            'cw_p50_duration_ms':   round(max(0, cw_avg * 0.9), 3),
            'cw_p95_duration_ms':   round(max(0, cw_p95), 3),
            'cw_p99_duration_ms':   round(max(0, cw_p99), 3),
            'cw_p999_duration_ms':  round(max(0, cw_p999), 3),
            'cw_avg_init_ms':       round(max(0, rng.gauss(COLD_START_DIST[fn][0]*0.3, 20)), 3) if cold else 0,
            'cw_p95_init_ms':       round(max(0, COLD_START_DIST[fn][0] * 1.2), 3),
            'cw_p99_init_ms':       round(max(0, COLD_START_DIST[fn][0] * 1.8), 3),
            'cw_total_invocations': rng.randint(10, 500),
            'cw_max_concurrent_execs': rng.randint(1, 20),
            'cw_error_count':       rng.randint(0, 3),
            'is_tail_p95':          int(duration > cw_p95),
            'is_tail_p99':          int(duration > cw_p99),
            'simulated_traffic_level': round(traffic, 4),
            'traffic_phase':        traffic_phase(hod),
        })

    cold_count = sum(r['cold_start_flag'] for r in rows)
    print(f'  Cold starts: {cold_count}/{n_samples} ({100*cold_count/n_samples:.1f}%)')
    for fn in FUNCTION_TYPES:
        fn_rows = [r for r in rows if r['function_type'] == fn]
        if fn_rows:
            avg_init = sum(r['init_duration_ms'] for r in fn_rows if r['cold_start_flag']) / max(1, sum(r['cold_start_flag'] for r in fn_rows))
            print(f'  {fn:15s}: {len(fn_rows):4d} samples  avg_init={avg_init:.1f}ms')
    return rows


# ── WRITE CSV ───────────────────────────────────────────────────
def write_csv(rows: list, output_path: str):
    if not rows:
        print('[warn] No rows to write.')
        return
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[extract] ✓ Wrote {len(rows)} rows → {output_path}')


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Extract cold start dataset')
    parser.add_argument('--hours',      type=int,   default=24,
                        help='Hours of history to pull from DynamoDB (default: 24)')
    parser.add_argument('--output',     type=str,   default=OUTPUT_CSV,
                        help=f'Output CSV path (default: {OUTPUT_CSV})')
    parser.add_argument('--synthetic',  action='store_true',
                        help='Generate synthetic data only (no AWS calls)')
    parser.add_argument('--n-synthetic',type=int,   default=2000,
                        help='Number of synthetic samples (default: 2000)')
    parser.add_argument('--merge',      action='store_true',
                        help='Merge real + synthetic data')
    args = parser.parse_args()

    print('═' * 60)
    print('  Cold Start Dataset Extractor')
    print('═' * 60)
    print(f'  Output:    {args.output}')
    print(f'  Mode:      {"synthetic" if args.synthetic else "real AWS"}')
    if not args.synthetic:
        print(f'  Lookback:  {args.hours}h')
    print('═' * 60)

    rows = []

    if args.synthetic:
        rows = generate_synthetic_data(args.n_synthetic)
    else:
        try:
            rows = extract_real_data(args.hours)
        except Exception as e:
            print(f'\n[error] AWS extraction failed: {e}')
            print('[fallback] Generating synthetic data instead...')
            rows = generate_synthetic_data(args.n_synthetic)

    if args.merge and not args.synthetic:
        synthetic_rows = generate_synthetic_data(args.n_synthetic // 2)
        rows.extend(synthetic_rows)
        print(f'[merge] Combined: {len(rows)} total rows')

    write_csv(rows, args.output)

    # Print quick summary
    if rows:
        cold = sum(r['cold_start_flag'] for r in rows)
        print(f'\n  Total rows:      {len(rows)}')
        print(f'  Cold starts:     {cold} ({100*cold/len(rows):.1f}%)')
        print(f'  Function types:  {sorted(set(r["function_type"] for r in rows))}')
        print(f'  Time range:      {min(r["timestamp"] for r in rows)[:19]}')
        print(f'               →  {max(r["timestamp"] for r in rows)[:19]}')
        print(f'\n  Ready for: python train_models.py --input {args.output}')


if __name__ == '__main__':
    main()
