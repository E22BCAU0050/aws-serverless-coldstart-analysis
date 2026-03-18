"""
Parses InitDuration from CloudWatch REPORT logs for all 5 Lambda variants
and enriches cold_start_dataset.csv with real init_duration_ms values.
"""
import boto3, re, csv, json
from datetime import datetime, timezone, timedelta

REGION   = 'us-east-1'
LOG_GROUPS = {
    'non-vpc':     '/aws/lambda/coldstart-research-api-non-vpc',
    'vpc':         '/aws/lambda/coldstart-research-api-vpc',
    'provisioned': '/aws/lambda/coldstart-research-api-provisioned',
    'small-pkg':   '/aws/lambda/coldstart-research-api-small-pkg',
    'large-pkg':   '/aws/lambda/coldstart-research-api-large-pkg',
}

client    = boto3.client('logs', region_name=REGION)
init_map  = {}  # requestId -> init_duration_ms

start_ms = int((datetime.now(timezone.utc) - timedelta(hours=168)).timestamp() * 1000)

for fn, lg in LOG_GROUPS.items():
    print(f'Parsing {fn}...')
    kwargs = dict(
        logGroupName=lg,
        filterPattern='Init Duration',
        startTime=start_ms,
        limit=10000,
    )
    count = 0
    while True:
        resp   = client.filter_log_events(**kwargs)
        for e in resp.get('events', []):
            msg = e['message']
            # Extract RequestId and Init Duration
            rid   = re.search(r'RequestId:\s+([\w-]+)', msg)
            init  = re.search(r'Init Duration:\s+([\d.]+)', msg)
            dur   = re.search(r'Duration:\s+([\d.]+)', msg)
            if rid and init:
                init_map[rid.group(1)] = {
                    'init_duration_ms': float(init.group(1)),
                    'duration_ms':      float(dur.group(1)) if dur else 0.0,
                    'function_type':    fn,
                    'cold_start_flag':  1,
                }
                count += 1
        token = resp.get('nextToken')
        if not token:
            break
        kwargs['nextToken'] = token
    print(f'  {fn}: {count} cold start REPORT lines found')

print(f'\nTotal cold start records from logs: {len(init_map)}')

# Enrich the CSV
in_rows  = []
with open('cold_start_dataset.csv') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rid = row['request_id']
        if rid in init_map:
            row['init_duration_ms'] = init_map[rid]['init_duration_ms']
            row['cold_start_flag']  = 1
        in_rows.append(row)

# Write enriched CSV
with open('cold_start_dataset.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(in_rows)

cold_count = sum(1 for r in in_rows if int(r['cold_start_flag']) == 1)
print(f'Enriched CSV: {len(in_rows)} rows, {cold_count} cold starts with real InitDuration')
print(f'\nSample init durations:')
samples = [(r["function_type"], r["init_duration_ms"]) for r in in_rows if float(r["init_duration_ms"]) > 0][:10]
for fn, ms in samples:
    print(f'  {fn}: {ms}ms')
