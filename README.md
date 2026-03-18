# Cold Start Latency Research
## Hybrid ML/DL Framework for Serverless Prediction

> Quantile Regression • Temporal Fusion Transformer • SHAP Explainability • Dynamic Provisioning

---

## What This Does

This project measures AWS Lambda cold start latency across 5 deployment configurations, then trains a hybrid ML/DL model to predict and optimise it.

**Research contributions:**
1. **Tail latency prediction** — Quantile Regression predicts p10/p50/p90 cold start uncertainty
2. **SHAP explainability** — XGBoost + SHAP shows which features (VPC, package size, time-of-day) drive cold starts
3. **Temporal Fusion Transformer** — predicts future traffic from 24h sine wave pattern
4. **Ablation studies** — model accuracy with/without VPC feature, with/without TFT predictor
5. **Dynamic provisioning** — proves TFT-driven proactive scaling beats AWS reactive autoscaling on p99 latency

---

## File Structure

```
serverless-coldstart-analysis/
├── run_all.sh                  ← master script (start here)
├── main-stack.yaml             ← CloudFormation (5 Lambda variants + VPC + DynamoDB)
├── load-test.js                ← k6 load test (sine wave 24h traffic)
├── extract_dataset.py          ← DynamoDB + CloudWatch → CSV
├── train_models.py             ← Quantile Regression + XGBoost + TFT + SHAP
├── dynamic_provisioning.py     ← TFT predictions → Lambda concurrency
└── README.md
```

---

## Lambda Variants (5 experimental configurations)

| Function | VPC | Provisioned | Package | Deps | Expected Cold Start |
|----------|-----|-------------|---------|------|---------------------|
| `non-vpc` | ✗ | ✗ | 48KB | 3 | ~280ms |
| `vpc` | ✓ | ✗ | 48KB | 3 | ~950ms (ENI overhead) |
| `provisioned` | ✗ | ✓ (2) | 48KB | 3 | ~12ms |
| `small-pkg` | ✗ | ✗ | 32KB | 1 | ~180ms |
| `large-pkg` | ✗ | ✗ | 512KB | 15 | ~620ms |

---

## Quick Start (No AWS Required)

Test the full ML/DL pipeline locally using synthetic data:

```bash
# 1. Install Python dependencies
pip install scikit-learn xgboost shap matplotlib pandas numpy tensorflow

# 2. Run full pipeline on synthetic data
./run_all.sh quick-run
```

This generates 2000 synthetic samples and trains all models. Outputs go to `model_outputs/`.

---

## Full AWS Experiment (Step by Step)

### Prerequisites

```bash
# Install AWS CLI
brew install awscli          # macOS
# or: https://aws.amazon.com/cli/

# Install k6
brew install k6              # macOS
# or: https://k6.io/docs/get-started/installation/

# Install Python deps
pip install scikit-learn xgboost shap matplotlib pandas numpy tensorflow boto3

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, region=us-east-1, output=json

# Verify credentials
aws sts get-caller-identity
```

### One-time AWS account setup (run once, ever)

```bash
# Set CloudWatch Logs role for API Gateway (needed for API GW metrics)
aws iam create-role \
  --role-name APIGatewayCloudWatchLogsRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"apigateway.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

aws iam attach-role-policy \
  --role-name APIGatewayCloudWatchLogsRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonAPIGatewayPushToCloudWatchLogs

ROLE_ARN=$(aws iam get-role --role-name APIGatewayCloudWatchLogsRole --query 'Role.Arn' --output text)
aws apigateway update-account --patch-operations op=replace,path=/cloudwatchRoleArn,value=$ROLE_ARN --region us-east-1
```

---

### Step 1 — Deploy Infrastructure (~5-8 min)

```bash
./run_all.sh deploy
```

Deploys: VPC, 2 private subnets, NAT Gateway, 5 Lambda functions, DynamoDB, API Gateway.

**Verify:**
```bash
./run_all.sh status
```

---

### Step 2 — Seed DynamoDB

```bash
./run_all.sh seed
```

Populates `coldstart-research-books` with 20 books across 8 genres.

---

### Step 3 — Cold Start Experiment

```bash
./run_all.sh test
```

Invokes each Lambda directly (bypasses API GW) to capture `InitDuration` in CloudWatch logs. Repeat 2-3 times with 10-minute gaps to get multiple cold start samples.

**Force cold starts manually:**
```bash
# Wait for Lambda to go cold (~15 min idle), then:
aws lambda invoke \
  --function-name coldstart-research-api-non-vpc \
  --region us-east-1 \
  --payload '{"path":"/health","httpMethod":"GET"}' \
  --cli-binary-format raw-in-base64-out \
  --log-type Tail \
  /tmp/out.json \
  --query 'LogResult' --output text | base64 --decode | grep INIT
```

---

### Step 4 — Load Test (Sine Wave Traffic)

```bash
./run_all.sh load
```

Runs k6 with 3 scenarios:
- **sine_wave_traffic** — 30 min test simulating 24h traffic pattern
- **cold_start_burst** — 20 VUs hit all variants simultaneously at start
- **spike_test** — 2x traffic spike at 40% through the run

**Custom duration/load:**
```bash
DURATION_MINUTES=60 PEAK_VUS=50 ./run_all.sh load
```

---

### Step 5 — Extract Dataset

```bash
./run_all.sh extract
```

Pulls from DynamoDB + CloudWatch and writes `cold_start_dataset.csv` with:
- Raw measurements: `duration_ms`, `init_duration_ms`, `cold_start_flag`
- Features: `vpc_flag`, `provisioned_flag`, `package_size_kb`, `dep_count`, `hour_sin`, `hour_cos`
- CloudWatch aggregates: `cw_p95_duration_ms`, `cw_p99_duration_ms`, `cw_p999_duration_ms`
- Traffic context: `simulated_traffic_level`, `traffic_phase`

---

### Step 6 — Train Models

```bash
./run_all.sh train
```

Trains the full hybrid framework:

| Model | Purpose | Output |
|-------|---------|--------|
| Quantile Regression | p10/p50/p90 cold start prediction | prediction intervals |
| XGBoost + SHAP | magnitude prediction + feature explanation | shap importance chart |
| Gradient Boosting | ensemble baseline | comparison metrics |
| Ridge | linear baseline | comparison metrics |
| TFT | 30-min traffic forecast | next invocation count |

**Ablation studies run automatically:**
- Accuracy with VPC feature vs without
- Accuracy with traffic context (TFT) vs without
- Cross-model R²/MAE/RMSE comparison table

**Custom epochs:**
```bash
TFT_EPOCHS=50 ./run_all.sh train
```

---

### Step 7 — Dynamic Provisioning

```bash
# See what the provisioner would do (no AWS calls)
./run_all.sh provision --simulate

# Run 24-hour simulation
python3 dynamic_provisioning.py --run-24h-sim

# Actually call Lambda to adjust concurrency (requires trained model)
./run_all.sh provision
```

---

### Step 8 — View Results

```bash
# Open CloudWatch dashboard
./run_all.sh dashboard

# View training outputs
ls -lh model_outputs/
#   ml_results.png          — quantile regression + model comparison + SHAP + ablation
#   tft_results.png         — TFT prediction + training history + provisioning comparison
#   tail_latency.png        — boxplots + tail latency curves + CDF per variant
#   ablation_results.json   — VPC/TFT ablation numbers
#   provisioning_comparison.json  — TFT vs AWS autoscaling cost/latency
#   training_summary.json   — all metrics in one file
```

---

### Step 9 — Cleanup (important — NAT Gateway costs ~$0.045/hr)

```bash
./run_all.sh destroy
```

---

## Cost Estimate

| Resource | Cost | Notes |
|----------|------|-------|
| NAT Gateway | ~$0.045/hr | **Destroy between sessions** |
| Provisioned Concurrency (2×128MB) | ~$0.004/hr | Only while stack exists |
| Lambda invocations | ~$0.0000002 each | Negligible |
| DynamoDB (PAY_PER_REQUEST) | ~$0.00 | <1M requests free tier |
| API Gateway | ~$0.00 | <1M calls free tier |
| **Total for 4-hour session** | **~$0.20** | |

---

## Model Outputs Explained

### `ml_results.png` — 4 panels
1. **Quantile regression intervals** — actual cold start values vs predicted q10–q90 band
2. **Cross-model comparison** — MAE + R² bar chart (Ridge vs GradBoost vs XGBoost)
3. **SHAP feature importance** — top 8 features driving cold start duration
4. **Ablation study** — R² with/without VPC feature and TFT traffic context

### `tft_results.png` — 4 panels
1. **TFT prediction vs actual** — traffic forecast with ±15% uncertainty band
2. **TFT training loss** — convergence curve
3. **24h sine wave** — traffic pattern used as TFT input
4. **Dynamic provisioning** — TFT p99 vs AWS autoscaling p99 over 24h

### `tail_latency.png` — 3 panels
1. **Boxplot** — cold start distribution per variant (median, IQR, outliers)
2. **Tail latency curves** — p50/p90/p95/p99/p999 on log scale
3. **CDF** — cumulative probability vs latency (shows % of requests under SLA)

---

## Research Paper Structure (Suggested)

1. **Introduction** — cold start problem in serverless, business impact of tail latency
2. **Background** — Lambda execution model, VPC/ENI overhead, provisioned concurrency
3. **Methodology** — 5-variant experimental setup, sine wave traffic model, dataset schema
4. **ML Framework** — Quantile Regression for uncertainty, XGBoost for magnitude, SHAP for explainability
5. **DL Framework** — TFT architecture, attention mechanism, 30-min traffic forecasting
6. **Results** — cold start distributions, tail latency analysis, SHAP feature importance
7. **Ablation** — VPC feature impact (+X% R²), TFT traffic context impact (+Y% R²)
8. **Dynamic Provisioning** — TFT proactive vs AWS reactive: Z% latency reduction at W% cost delta
9. **Conclusion** — when to use VPC/provisioned/small-package, optimal deployment strategy

---

## Troubleshooting

**Stack deploy fails with ValidationError (ROLLBACK_COMPLETE)**
```bash
aws cloudformation delete-stack --stack-name coldstart-research --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name coldstart-research --region us-east-1
./run_all.sh deploy
```

**Early validation error on redeploy**
```bash
./run_all.sh clean-logs
./run_all.sh deploy
```

**No cold start data in DynamoDB**
```bash
# Force cold starts by waiting 15+ minutes then invoking
./run_all.sh test
sleep 60
./run_all.sh test  # repeat
```

**TFT training crashes (TensorFlow not available)**
The TFT automatically falls back to a numpy ARMA approximation. All other models still train. Install TF for full TFT: `pip install tensorflow`

**k6 not found**
```bash
brew install k6   # macOS
# Linux: https://k6.io/docs/get-started/installation/
```
