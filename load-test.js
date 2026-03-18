/**
 * Cold Start Research — k6 Load Test
 * ─────────────────────────────────────────────────────────────
 * Sine-wave 24-hour traffic simulation across all Lambda variants
 * Captures: cold starts, tail latency (p95/p99/p999), per-variant metrics
 *
 * Usage:
 *   k6 run load-test.js -e API_URL=https://xxxx.execute-api.us-east-1.amazonaws.com/dev
 *   k6 run load-test.js -e API_URL=... -e DURATION_MINUTES=60 -e PEAK_VUS=50
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend, Gauge } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// ── ENV CONFIG ────────────────────────────────────────────────
const API_URL          = __ENV.API_URL          || 'http://localhost:3000';
const DURATION_MINUTES = parseInt(__ENV.DURATION_MINUTES || '30');
const PEAK_VUS         = parseInt(__ENV.PEAK_VUS         || '30');
const START_HOUR       = parseInt(__ENV.START_HOUR        || '8');  // simulate starting at 8am

// ── CUSTOM METRICS ────────────────────────────────────────────
// Cold start detection
const coldStartCount   = new Counter('cold_starts_total');
const coldStartRate    = new Rate('cold_start_rate');
const coldStartLatency = new Trend('cold_start_init_ms', true);   // true = percentiles

// Per-variant latency trends (tail latency)
const latencyNonVpc    = new Trend('latency_non_vpc_ms',    true);
const latencyVpc       = new Trend('latency_vpc_ms',        true);
const latencyProvis    = new Trend('latency_provisioned_ms',true);
const latencySmallPkg  = new Trend('latency_small_pkg_ms',  true);
const latencyLargePkg  = new Trend('latency_large_pkg_ms',  true);

// Tail latency trackers (sampled every request)
const tailP95          = new Trend('tail_p95_ms',  true);
const tailP99          = new Trend('tail_p99_ms',  true);

// Traffic shape
const invocationsTotal = new Counter('invocations_total');
const errorRate        = new Rate('error_rate');

// Dynamic provisioning gauge
const provisioningLevel = new Gauge('provisioning_level');

// ── SINE WAVE TRAFFIC MODEL ───────────────────────────────────
/**
 * Simulates realistic 24-hour traffic pattern:
 *   - Low traffic at night (1-3am): ~5% of peak
 *   - Morning ramp-up (6-9am): rising
 *   - Midday peak (12-2pm): 100% of peak
 *   - Evening secondary peak (6-8pm): ~70% of peak
 *   - Night trough: back to low
 *
 * Returns VU count for a given simulated hour (0-23)
 */
function sineWaveVUs(simulatedHour, peakVUs) {
  // Primary sine: peaks at hour 13 (1pm)
  const primary   = Math.sin((simulatedHour - 7) * Math.PI / 12);
  // Secondary evening bump: peaks at hour 19 (7pm)
  const secondary = 0.4 * Math.sin((simulatedHour - 13) * Math.PI / 8);
  // Combine, clamp to [0,1]
  const raw = Math.max(0, primary + secondary);
  // Floor at 5% of peak (background traffic always exists)
  return Math.max(Math.floor(peakVUs * 0.05), Math.floor(peakVUs * raw));
}

/**
 * Get current simulated hour based on elapsed test time.
 * DURATION_MINUTES of test = 24 simulated hours
 */
function getSimulatedHour(elapsedSeconds) {
  const totalSeconds   = DURATION_MINUTES * 60;
  const progress       = elapsedSeconds / totalSeconds;          // 0→1
  const simulatedHours = (START_HOUR + progress * 24) % 24;     // wrap at 24
  return simulatedHours;
}

// ── VU STAGES (sine-approximated as k6 stages) ────────────────
// We approximate the sine wave using k6 stages since k6 doesn't
// support dynamic VU counts natively. Stages simulate the shape.
function buildSineStages(totalMinutes, peakVUs) {
  const stages = [];
  const numSteps = 24;  // one step per simulated hour
  const stepDuration = Math.floor((totalMinutes * 60) / numSteps);

  for (let i = 0; i < numSteps; i++) {
    const hour = (START_HOUR + i) % 24;
    const vus  = sineWaveVUs(hour, peakVUs);
    stages.push({ duration: `${stepDuration}s`, target: vus });
  }
  return stages;
}

// ── TEST OPTIONS ──────────────────────────────────────────────
export const options = {
  scenarios: {
    // ── Scenario 1: 24-hour sine wave traffic (main experiment)
    sine_wave_traffic: {
      executor:          'ramping-vus',
      startVUs:          Math.floor(PEAK_VUS * 0.05),
      stages:            buildSineStages(DURATION_MINUTES, PEAK_VUS),
      gracefulRampDown:  '10s',
      exec:              'sineWaveScenario',
      tags:              { scenario: 'sine_wave' },
    },

    // ── Scenario 2: Cold start burst (runs once at start)
    cold_start_burst: {
      executor:    'per-vu-iterations',
      vus:         20,
      iterations:  1,
      maxDuration: '60s',
      exec:        'coldStartBurst',
      tags:        { scenario: 'cold_start_burst' },
      startTime:   '0s',
    },

    // ── Scenario 3: Spike test (simulates sudden traffic surge)
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 0           },
        { duration: '20s', target: PEAK_VUS * 2 }, // 2x spike
        { duration: '30s', target: PEAK_VUS * 2 },
        { duration: '20s', target: 0           },
      ],
      exec:      'spikeScenario',
      tags:      { scenario: 'spike' },
      startTime: `${Math.floor(DURATION_MINUTES * 0.4)}m`,  // 40% into run
    },
  },

  thresholds: {
    'latency_non_vpc_ms':     ['p(95)<2000', 'p(99)<5000'],
    'latency_vpc_ms':         ['p(95)<4000'],
    'latency_provisioned_ms': ['p(95)<500'],
    'cold_start_rate':        ['rate<0.3'],
    'error_rate':             ['rate<0.05'],
    'http_req_duration':      ['p(95)<5000'],
    'http_req_failed':        ['rate<0.05'],
  },
};

// ── HELPERS ───────────────────────────────────────────────────
function parseHeaders(res) {
  return {
    cold:     (res.headers['X-Cold-Start'] || 'false').toLowerCase() === 'true',
    fnType:   res.headers['X-Function-Type'] || 'unknown',
    durationMs: parseFloat(res.headers['X-Duration-Ms'] || '0'),
    initMs:   parseFloat(res.headers['X-Init-Ms'] || '0'),
  };
}

function recordLatency(fnType, ms, cold, initMs) {
  invocationsTotal.add(1, { fn: fnType });
  coldStartRate.add(cold);

  if (cold) {
    coldStartCount.add(1, { fn: fnType });
    if (initMs > 0) coldStartLatency.add(initMs, { fn: fnType });
  }

  // Per-variant trends
  switch (fnType) {
    case 'non-vpc':     latencyNonVpc.add(ms);   break;
    case 'vpc':         latencyVpc.add(ms);      break;
    case 'provisioned': latencyProvis.add(ms);   break;
    case 'small-pkg':   latencySmallPkg.add(ms); break;
    case 'large-pkg':   latencyLargePkg.add(ms); break;
  }

  // Tail latency (all variants)
  tailP95.add(ms);
  tailP99.add(ms);
}

function invokeFunction(url, fnLabel, payload) {
  const headers = {
    'Content-Type':    'application/json',
    'X-Research-Test': 'true',
    'X-Function-Target': fnLabel,
  };
  const res = http.get(url, { headers, tags: { fn: fnLabel } });

  const ok = check(res, {
    [`${fnLabel} status 200`]: (r) => r.status === 200 || r.status === 201,
    [`${fnLabel} has body`]:   (r) => r.body && r.body.length > 0,
  });

  errorRate.add(!ok);

  const h = parseHeaders(res);
  recordLatency(fnLabel, res.timings.duration, h.cold, h.initMs);

  return { res, cold: h.cold, ms: res.timings.duration };
}

// ── BOOK IDs SHARED ARRAY (for GET requests) ──────────────────
// Pre-generate fake IDs — in real run seeder populates actual IDs
const fakeBookIds = new SharedArray('bookIds', function () {
  const ids = [];
  for (let i = 0; i < 20; i++) {
    ids.push(`book-${i}-placeholder`);
  }
  return ids;
});

// ── SCENARIO FUNCTIONS ────────────────────────────────────────

/**
 * Scenario 1: Main sine wave — all variants tested proportionally
 */
export function sineWaveScenario() {
  const endpoints = [
    { url: `${API_URL}/health`,    fn: 'non-vpc'    },
    { url: `${API_URL}/health`,    fn: 'vpc'        },
    { url: `${API_URL}/health`,    fn: 'provisioned'},
    { url: `${API_URL}/books`,     fn: 'non-vpc'    },
    { url: `${API_URL}/books`,     fn: 'vpc'        },
  ];

  // Each VU picks a random endpoint to hit
  const ep = endpoints[Math.floor(Math.random() * endpoints.length)];

  const headers = {
    'Content-Type':        'application/json',
    'X-Research-Test':     'true',
    'X-Function-Target':   ep.fn,
  };

  const res = http.get(ep.url, { headers, tags: { fn: ep.fn, scenario: 'sine_wave' } });

  const ok = check(res, {
    'status 2xx': (r) => r.status >= 200 && r.status < 300,
  });
  errorRate.add(!ok);

  const h = parseHeaders(res);
  recordLatency(ep.fn, res.timings.duration, h.cold, h.initMs);

  // Pacing: short sleep to not overwhelm
  sleep(Math.random() * 0.5 + 0.1);
}

/**
 * Scenario 2: Cold start burst — force cold starts by invoking all variants
 * simultaneously after they've been idle. Each VU hits a different variant.
 */
export function coldStartBurst() {
  const vuIndex = __VU % 5;

  const targets = [
    { url: `${API_URL}/health`, fn: 'non-vpc',     label: 'Non-VPC baseline'  },
    { url: `${API_URL}/health`, fn: 'vpc',          label: 'VPC + ENI'         },
    { url: `${API_URL}/health`, fn: 'provisioned',  label: 'Provisioned'       },
    { url: `${API_URL}/health`, fn: 'small-pkg',    label: 'Small package'     },
    { url: `${API_URL}/health`, fn: 'large-pkg',    label: 'Large package'     },
  ];

  const t = targets[vuIndex];
  const headers = {
    'Content-Type':      'application/json',
    'X-Research-Test':   'true',
    'X-Function-Target': t.fn,
    'X-Cold-Start-Test': 'burst',
  };

  // No sleep before — hit immediately to maximize cold start probability
  const res = http.get(t.url, { headers, tags: { fn: t.fn, scenario: 'cold_start_burst' } });

  check(res, {
    [`${t.label} responded`]:     (r) => r.status === 200,
    [`${t.label} body valid`]:    (r) => {
      try { JSON.parse(r.body); return true; } catch { return false; }
    },
  });

  const h = parseHeaders(res);
  recordLatency(t.fn, res.timings.duration, h.cold, h.initMs);

  console.log(`[cold_burst] fn=${t.fn} cold=${h.cold} ms=${res.timings.duration.toFixed(1)}`);
}

/**
 * Scenario 3: Spike test — sudden traffic surge to measure provisioning response
 */
export function spikeScenario() {
  const urls = [
    `${API_URL}/health`,
    `${API_URL}/books`,
    `${API_URL}/books`,  // weighted heavier
  ];

  const url = urls[Math.floor(Math.random() * urls.length)];
  const headers = {
    'Content-Type':    'application/json',
    'X-Research-Test': 'true',
    'X-Spike-Test':    'true',
  };

  const res = http.get(url, { headers, tags: { scenario: 'spike' } });

  check(res, {
    'spike: status 2xx': (r) => r.status >= 200 && r.status < 300,
    'spike: fast enough': (r) => r.timings.duration < 10000,
  });

  const h = parseHeaders(res);
  recordLatency(h.fnType || 'non-vpc', res.timings.duration, h.cold, h.initMs);

  sleep(0.05);
}

// ── SETUP: print test config ──────────────────────────────────
export function setup() {
  console.log('═'.repeat(60));
  console.log('  Cold Start Research — k6 Load Test');
  console.log('═'.repeat(60));
  console.log(`  API URL:          ${API_URL}`);
  console.log(`  Duration:         ${DURATION_MINUTES} minutes`);
  console.log(`  Peak VUs:         ${PEAK_VUS}`);
  console.log(`  Simulated start:  ${START_HOUR}:00`);
  console.log(`  Sine stages:      ${buildSineStages(DURATION_MINUTES, PEAK_VUS).length}`);
  console.log('═'.repeat(60));

  // Verify API is reachable
  const res = http.get(`${API_URL}/health`);
  if (res.status !== 200) {
    console.error(`[WARN] API health check failed: ${res.status} — ${res.body}`);
  } else {
    console.log(`  API health: OK (${res.timings.duration.toFixed(1)}ms)`);
  }
  console.log('');

  return {
    apiUrl:   API_URL,
    startTime: new Date().toISOString(),
    peakVUs:  PEAK_VUS,
  };
}

// ── TEARDOWN: summary ─────────────────────────────────────────
export function teardown(data) {
  console.log('');
  console.log('═'.repeat(60));
  console.log('  Test complete.');
  console.log(`  Started: ${data.startTime}`);
  console.log(`  Ended:   ${new Date().toISOString()}`);
  console.log('  Results written to k6_coldstart_summary.json');
  console.log('═'.repeat(60));
}

// ── handleSummary: export JSON for extract_dataset.py ─────────
export function handleSummary(data) {
  // Build per-function summary
  const summary = {
    test_config: {
      api_url:           API_URL,
      duration_minutes:  DURATION_MINUTES,
      peak_vus:          PEAK_VUS,
      start_hour:        START_HOUR,
      timestamp:         new Date().toISOString(),
    },
    metrics: {
      cold_starts_total:    data.metrics.cold_starts_total?.values?.count      || 0,
      cold_start_rate:      data.metrics.cold_start_rate?.values?.rate         || 0,
      cold_start_init_p50:  data.metrics.cold_start_init_ms?.values?.['p(50)'] || 0,
      cold_start_init_p95:  data.metrics.cold_start_init_ms?.values?.['p(95)'] || 0,
      cold_start_init_p99:  data.metrics.cold_start_init_ms?.values?.['p(99)'] || 0,
      latency_non_vpc: {
        p50:  data.metrics.latency_non_vpc_ms?.values?.['p(50)']  || 0,
        p95:  data.metrics.latency_non_vpc_ms?.values?.['p(95)']  || 0,
        p99:  data.metrics.latency_non_vpc_ms?.values?.['p(99)']  || 0,
        p999: data.metrics.latency_non_vpc_ms?.values?.['p(99.9)']|| 0,
        avg:  data.metrics.latency_non_vpc_ms?.values?.avg        || 0,
      },
      latency_vpc: {
        p50:  data.metrics.latency_vpc_ms?.values?.['p(50)']  || 0,
        p95:  data.metrics.latency_vpc_ms?.values?.['p(95)']  || 0,
        p99:  data.metrics.latency_vpc_ms?.values?.['p(99)']  || 0,
        p999: data.metrics.latency_vpc_ms?.values?.['p(99.9)']|| 0,
        avg:  data.metrics.latency_vpc_ms?.values?.avg        || 0,
      },
      latency_provisioned: {
        p50:  data.metrics.latency_provisioned_ms?.values?.['p(50)']  || 0,
        p95:  data.metrics.latency_provisioned_ms?.values?.['p(95)']  || 0,
        p99:  data.metrics.latency_provisioned_ms?.values?.['p(99)']  || 0,
        p999: data.metrics.latency_provisioned_ms?.values?.['p(99.9)']|| 0,
        avg:  data.metrics.latency_provisioned_ms?.values?.avg        || 0,
      },
      latency_small_pkg: {
        p50:  data.metrics.latency_small_pkg_ms?.values?.['p(50)']  || 0,
        p95:  data.metrics.latency_small_pkg_ms?.values?.['p(95)']  || 0,
        p99:  data.metrics.latency_small_pkg_ms?.values?.['p(99)']  || 0,
        avg:  data.metrics.latency_small_pkg_ms?.values?.avg        || 0,
      },
      latency_large_pkg: {
        p50:  data.metrics.latency_large_pkg_ms?.values?.['p(50)']  || 0,
        p95:  data.metrics.latency_large_pkg_ms?.values?.['p(95)']  || 0,
        p99:  data.metrics.latency_large_pkg_ms?.values?.['p(99)']  || 0,
        avg:  data.metrics.latency_large_pkg_ms?.values?.avg        || 0,
      },
      tail_p95:  data.metrics.tail_p95_ms?.values?.['p(95)'] || 0,
      tail_p99:  data.metrics.tail_p99_ms?.values?.['p(99)'] || 0,
      error_rate: data.metrics.error_rate?.values?.rate       || 0,
      invocations_total: data.metrics.invocations_total?.values?.count || 0,
    },
    thresholds_passed: Object.entries(data.metrics).reduce((acc, [k, v]) => {
      if (v?.thresholds) acc[k] = Object.values(v.thresholds).every(t => !t.ok === false);
      return acc;
    }, {}),
  };

  return {
    'k6_coldstart_summary.json': JSON.stringify(summary, null, 2),
    'k6_raw_summary.json':       JSON.stringify(data,    null, 2),
    stdout: '\n[k6] Results saved to k6_coldstart_summary.json\n',
  };
}
