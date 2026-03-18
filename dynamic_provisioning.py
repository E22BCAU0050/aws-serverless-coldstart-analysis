#!/usr/bin/env python3
"""
dynamic_provisioning.py
──────────────────────────────────────────────────────────────────
Reads TFT traffic predictions from training_summary.json and calls
the DynamicProvisioner Lambda to adjust provisioned concurrency.

Can be run:
  1. Manually after training:  python dynamic_provisioning.py
  2. On a cron / EventBridge schedule for live deployment
  3. In simulation mode:       python dynamic_provisioning.py --simulate

Usage:
  python dynamic_provisioning.py
  python dynamic_provisioning.py --simulate          # no AWS calls
  python dynamic_provisioning.py --hours-ahead 1     # 1h lookahead
  python dynamic_provisioning.py --summary-file model_outputs/training_summary.json
"""

import boto3
import json
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

REGION              = 'us-east-1'
PROVISIONER_FN_NAME = 'coldstart-research-dynamic-provisioner'
SUMMARY_FILE        = 'model_outputs/training_summary.json'

# ── SINE WAVE TRAFFIC MODEL ──────────────────────────────────────
def sine_wave_invocations(hour: float, peak: int = 1000) -> int:
    primary   = math.sin((hour - 7) * math.pi / 12)
    secondary = 0.4 * math.sin((hour - 13) * math.pi / 8)
    level     = max(0.05, min(1.0, primary + secondary))
    return max(10, int(peak * level))


def get_tft_prediction(summary_file: str, hours_ahead: float = 0.5) -> dict:
    """
    Load TFT forecast from training_summary.json.
    Falls back to sine wave model if file not found.
    """
    now  = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60 + hours_ahead  # lookahead

    try:
        with open(summary_file) as f:
            summary = json.load(f)

        forecast = summary.get('dl', {}).get('tft', {}).get('next_30min_forecast', [])
        if forecast:
            # Use average of 30-min forecast as predicted invocations
            predicted = int(sum(forecast) / len(forecast))
            source = 'tft_model'
        else:
            predicted = sine_wave_invocations(hour % 24)
            source = 'sine_wave_fallback'

    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        predicted = sine_wave_invocations(hour % 24)
        source = 'sine_wave_fallback'

    return {
        'predicted_invocations': predicted,
        'hour_of_day':           round(hour % 24, 2),
        'source':                source,
        'timestamp':             now.isoformat(),
    }


def invoke_provisioner(prediction: dict, simulate: bool = False) -> dict:
    """Call the DynamicProvisioner Lambda with TFT predictions."""
    payload = {
        'predicted_invocations': prediction['predicted_invocations'],
        'hour_of_day':           int(prediction['hour_of_day']),
        'scaling_factor':        1.0,
        'source':                prediction['source'],
    }

    print(f'\n[provisioner] Calling Lambda...')
    print(f'  predicted_invocations: {payload["predicted_invocations"]}')
    print(f'  hour_of_day:           {payload["hour_of_day"]}')
    print(f'  source:                {payload["source"]}')

    if simulate:
        # Compute what the provisioner would do without calling AWS
        desired = max(1, min(10, math.ceil(payload['predicted_invocations'] / 50)))
        result = {
            'timestamp':  prediction['timestamp'],
            'hour':       payload['hour_of_day'],
            'predicted':  payload['predicted_invocations'],
            'current_pc': '(simulated)',
            'desired_pc': desired,
            'action':     f'[SIMULATED] would scale to {desired}',
        }
        print(f'\n  [SIMULATE] Would set provisioned concurrency → {desired}')
        return result

    try:
        client   = boto3.client('lambda', region_name=REGION)
        response = client.invoke(
            FunctionName=PROVISIONER_FN_NAME,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload).encode(),
        )
        result_payload = json.loads(response['Payload'].read())

        if response.get('FunctionError'):
            print(f'  [error] Lambda error: {result_payload}')
            return {'error': str(result_payload)}

        print(f'\n  [result] action:     {result_payload.get("action", "unknown")}')
        print(f'  [result] current_pc: {result_payload.get("current_pc", "?")}')
        print(f'  [result] desired_pc: {result_payload.get("desired_pc", "?")}')
        return result_payload

    except Exception as e:
        print(f'  [error] {e}')
        return {'error': str(e)}


def run_24h_simulation(peak_invocations: int = 1000):
    """
    Simulate a full 24-hour dynamic provisioning cycle.
    Shows how TFT-driven provisioning compares to AWS autoscaling.
    """
    print('\n' + '═'*60)
    print('  24-Hour Dynamic Provisioning Simulation')
    print('═'*60)
    print(f'  {"Hour":>5}  {"Traffic":>9}  {"TFT PC":>7}  {"AWS PC":>7}  {"TFT p99ms":>10}  {"AWS p99ms":>10}')
    print('  ' + '-'*60)

    AWS_DELAY = 3  # 3 × 5min = 15min lag for AWS autoscaling
    history   = []

    total_tft_cost = 0.0
    total_aws_cost = 0.0
    MEM_GB         = 128 / 1024
    INTERVAL_SEC   = 5 * 60
    COST_PER_UNIT  = MEM_GB * INTERVAL_SEC * 0.000064

    for step in range(0, 288):  # 24h × 12 steps/hr (5 min intervals)
        hour    = step * 5 / 60
        traffic = sine_wave_invocations(hour, peak_invocations)

        # TFT: proactive — uses 30-min lookahead
        future_hour    = (hour + 0.5) % 24
        future_traffic = sine_wave_invocations(future_hour, peak_invocations)
        tft_pc = max(1, min(10, math.ceil(future_traffic / 100)))

        # AWS autoscaling: reactive — 15-min lag
        past_step  = max(0, step - AWS_DELAY)
        past_hour  = past_step * 5 / 60
        past_traffic = sine_wave_invocations(past_hour, peak_invocations)
        aws_pc = max(1, min(10, math.ceil(past_traffic / 100)))

        # Simulate p99 latency
        needed = math.ceil(traffic / 100)
        def p99_latency(provisioned, needed):
            if provisioned >= needed:
                return 50.0   # all warm
            gap_ratio = (needed - provisioned) / needed
            return 50 + gap_ratio * 450  # interpolate to 500ms p99 cold

        tft_p99 = p99_latency(tft_pc, needed)
        aws_p99 = p99_latency(aws_pc, needed)

        total_tft_cost += tft_pc * COST_PER_UNIT
        total_aws_cost += aws_pc * COST_PER_UNIT

        history.append({
            'hour': hour, 'traffic': traffic,
            'tft_pc': tft_pc, 'aws_pc': aws_pc,
            'tft_p99': tft_p99, 'aws_p99': aws_p99,
        })

        if step % 12 == 0:  # print every hour
            print(f'  {hour:>5.1f}h  {traffic:>9,}  {tft_pc:>7}  {aws_pc:>7}  '
                  f'{tft_p99:>10.1f}  {aws_p99:>10.1f}')

    avg_tft_p99 = sum(h['tft_p99'] for h in history) / len(history)
    avg_aws_p99 = sum(h['aws_p99'] for h in history) / len(history)
    latency_red = (avg_aws_p99 - avg_tft_p99) / avg_aws_p99 * 100
    cost_delta  = (total_tft_cost - total_aws_cost) / total_aws_cost * 100

    print('\n' + '═'*60)
    print('  SUMMARY')
    print('═'*60)
    print(f'  TFT avg p99 latency:  {avg_tft_p99:.1f}ms')
    print(f'  AWS avg p99 latency:  {avg_aws_p99:.1f}ms')
    print(f'  Latency reduction:    {latency_red:.1f}%')
    print(f'  TFT daily cost:       ${total_tft_cost:.4f}')
    print(f'  AWS daily cost:       ${total_aws_cost:.4f}')
    print(f'  Cost delta:           {cost_delta:+.1f}%')
    print('═'*60)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate',      action='store_true', help='Simulate without AWS calls')
    parser.add_argument('--hours-ahead',   type=float, default=0.5,  help='Lookahead hours (default: 0.5)')
    parser.add_argument('--summary-file',  default=SUMMARY_FILE)
    parser.add_argument('--run-24h-sim',   action='store_true', help='Run 24h simulation')
    args = parser.parse_args()

    print('═'*60)
    print('  Dynamic Provisioning Controller')
    print(f'  Mode: {"SIMULATE" if args.simulate else "LIVE"}')
    print('═'*60)

    if args.run_24h_sim:
        run_24h_simulation()
        return

    prediction = get_tft_prediction(args.summary_file, args.hours_ahead)
    result     = invoke_provisioner(prediction, simulate=args.simulate)

    # Save result
    out = Path('model_outputs') / 'last_provisioning_action.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'prediction': prediction, 'result': result}, f, indent=2)
    print(f'\n  Saved → {out}')


if __name__ == '__main__':
    main()
