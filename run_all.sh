#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  Cold Start Research — Master Orchestration Script
#  Hybrid ML/DL Framework: Quantile Regression + TFT + SHAP
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

STACK_NAME="coldstart-research"
REGION="us-east-1"
TEMPLATE="./main-stack.yaml"
PROJECT="coldstart-research"

# ── Colors ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────
banner() { echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
           echo -e "${BOLD}${CYAN}  $1${NC}"
           echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
info()    { echo -e "${BLUE}[→]${NC} $1"; }
ok()      { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
err()     { echo -e "${RED}[✗]${NC} $1"; exit 1; }

require() {
  command -v "$1" &>/dev/null || err "$1 is required but not installed. Install: $2"
}

get_api_url() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' \
    --output text 2>/dev/null || echo ""
}

check_stack_ready() {
  local status
  status=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")
  if [[ "$status" != "CREATE_COMPLETE" && "$status" != "UPDATE_COMPLETE" ]]; then
    err "Stack is not ready (status: $status). Run: ./run_all.sh deploy"
  fi
}

# ══════════════════════════════════════════════════════════════════
#  COMMANDS
# ══════════════════════════════════════════════════════════════════

cmd_deploy() {
  banner "STEP 1 — Deploying CloudFormation Stack"
  require aws "https://aws.amazon.com/cli/"

  info "Stack    : $STACK_NAME"
  info "Account  : $(aws sts get-caller-identity --query Account --output text)"
  info "Region   : $REGION"
  info "Template : $TEMPLATE"

  [[ -f "$TEMPLATE" ]] || err "Template not found: $TEMPLATE"

  info "Verifying AWS credentials..."
  aws sts get-caller-identity --output json

  info "Deploying stack (this takes ~5-8 minutes for NAT Gateway)..."
  aws cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --template-file "$TEMPLATE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --no-fail-on-empty-changeset \
    --s3-bucket "coldstart-research-cfn-873166938412" \
    --parameter-overrides ProjectName="$PROJECT" ProvisionedConcurrency=2

  ok "Stack deployed!"

  local api_url
  api_url=$(get_api_url)
  ok "API URL: $api_url"
  echo "$api_url" > .api_url

  info "Waiting 10s for Lambda functions to be ready..."
  sleep 10

  info "Quick health check..."
  curl -s "$api_url/health" | python3 -m json.tool || warn "Health check failed — may need another 30s"
}

cmd_seed() {
  banner "STEP 2 — Seeding DynamoDB"
  check_stack_ready

  info "Invoking seeder Lambda (20 books)..."
  aws lambda invoke \
    --function-name "${PROJECT}-seeder" \
    --region "$REGION" \
    --payload '{"action":"seed"}' \
    --cli-binary-format raw-in-base64-out \
    /tmp/seed_response.json

  cat /tmp/seed_response.json | python3 -m json.tool
  ok "DynamoDB seeded!"
}

cmd_test() {
  banner "STEP 3 — Cold Start Experiment"
  check_stack_ready

  local api_url
  api_url=$(cat .api_url 2>/dev/null || get_api_url)
  [[ -z "$api_url" ]] && err "Cannot get API URL. Run: ./run_all.sh deploy"

  info "API URL: $api_url"
  info "Running cold start tests across all 5 Lambda variants..."

  # Force cold starts by sleeping between invocations
  FUNCTIONS=(
    "${PROJECT}-api-non-vpc"
    "${PROJECT}-api-vpc"
    "${PROJECT}-api-provisioned"
    "${PROJECT}-api-small-pkg"
    "${PROJECT}-api-large-pkg"
  )

  for fn in "${FUNCTIONS[@]}"; do
    info "Testing $fn..."
    # Direct Lambda invoke (bypasses API GW — captures InitDuration in logs)
    aws lambda invoke \
      --function-name "$fn" \
      --region "$REGION" \
      --payload '{"path":"/health","httpMethod":"GET"}' \
      --cli-binary-format raw-in-base64-out \
      --log-type Tail \
      /tmp/invoke_response.json \
      --query 'LogResult' --output text | base64 --decode | grep -E "REPORT|INIT" || true
    sleep 2
  done

  # Also hit via API Gateway (different init path — ENI attachment for VPC)
  info "Testing via API Gateway (triggers ENI attachment for VPC)..."
  for endpoint in "/health" "/books" "/health" "/books"; do
    curl -s -w "\n  HTTP %{http_code} | %{time_total}s\n" \
      "${api_url}${endpoint}" -o /dev/null || warn "Request failed"
    sleep 3
  done

  ok "Cold start test complete. Check CloudWatch for InitDuration metrics."
  info "Dashboard: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=${PROJECT}-dashboard"
}

cmd_load() {
  banner "STEP 4 — k6 Sine Wave Load Test"
  check_stack_ready
  require k6 "https://k6.io/docs/get-started/installation/"

  local api_url
  api_url=$(cat .api_url 2>/dev/null || get_api_url)
  [[ -z "$api_url" ]] && err "Cannot get API URL"

  info "API URL:  $api_url"
  info "Duration: ${DURATION_MINUTES:-30} minutes"
  info "Peak VUs: ${PEAK_VUS:-30}"
  info "Pattern:  24-hour sine wave (starts at 8am simulation)"

  k6 run load-test.js \
    -e API_URL="$api_url" \
    -e DURATION_MINUTES="${DURATION_MINUTES:-30}" \
    -e PEAK_VUS="${PEAK_VUS:-30}" \
    -e START_HOUR="${START_HOUR:-8}" \
    --out json=k6_raw_metrics.jsonl

  ok "Load test complete!"
  info "Output files:"
  ls -lh k6_coldstart_summary.json k6_raw_summary.json 2>/dev/null || warn "Summary files not found"
}

cmd_extract() {
  banner "STEP 5 — Extract Dataset"
  require python3 "https://python.org"

  info "Pulling metrics from DynamoDB + CloudWatch → CSV..."
  python3 extract_dataset.py \
    --hours "${LOOKBACK_HOURS:-24}" \
    --output cold_start_dataset.csv

  if [[ -f cold_start_dataset.csv ]]; then
    ok "Dataset ready: cold_start_dataset.csv"
    info "Rows: $(wc -l < cold_start_dataset.csv)"
    head -2 cold_start_dataset.csv
  fi
}

cmd_extract_synthetic() {
  banner "STEP 5 (Synthetic) — Generate Synthetic Dataset"
  require python3 "https://python.org"

  info "Generating 2000 synthetic samples (no AWS required)..."
  python3 extract_dataset.py --synthetic --n-synthetic 2000

  ok "Synthetic dataset: cold_start_dataset.csv"
}

cmd_train() {
  banner "STEP 6 — Train Hybrid ML/DL Models"
  require python3 "https://python.org"

  info "Checking Python dependencies..."
  python3 -c "import sklearn, xgboost, shap, matplotlib" 2>/dev/null || {
    warn "Installing Python dependencies..."
    pip install scikit-learn xgboost shap matplotlib pandas numpy tensorflow --quiet
  }

  [[ -f cold_start_dataset.csv ]] || {
    warn "cold_start_dataset.csv not found — generating synthetic data"
    python3 extract_dataset.py --synthetic
  }

  info "Training models..."
  info "  → Quantile Regression (p10/p50/p90)"
  info "  → XGBoost + SHAP explainability"
  info "  → Gradient Boosting baseline"
  info "  → Ridge Regression baseline"
  info "  → Temporal Fusion Transformer"
  info "  → Ablation studies (w/wo VPC, w/wo TFT)"
  info "  → Dynamic provisioning vs AWS autoscaling"

  python3 train_models.py \
    --input cold_start_dataset.csv \
    --output model_outputs/ \
    --epochs "${TFT_EPOCHS:-30}" \
    --ablation

  ok "Training complete! Outputs:"
  ls -lh model_outputs/ 2>/dev/null || warn "model_outputs/ not found"
}

cmd_train_synthetic() {
  banner "STEP 6 (Synthetic) — Train on Synthetic Data"
  require python3 "https://python.org"

  info "Training on synthetic dataset (no AWS required)..."
  python3 train_models.py --synthetic --output model_outputs/ --ablation

  ok "Synthetic training complete!"
  ls -lh model_outputs/ 2>/dev/null
}

cmd_provision() {
  banner "Dynamic Provisioning — Adjust Concurrency"

  if [[ "${1:-}" == "--simulate" ]]; then
    info "Running 24h simulation (no AWS calls)..."
    python3 dynamic_provisioning.py --run-24h-sim
  else
    check_stack_ready
    info "Invoking provisioner with TFT predictions..."
    python3 dynamic_provisioning.py \
      --summary-file model_outputs/training_summary.json \
      --hours-ahead 0.5
  fi
}

cmd_dashboard() {
  banner "CloudWatch Dashboard"
  local url="https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=${PROJECT}-dashboard"
  info "Opening: $url"
  open "$url" 2>/dev/null || xdg-open "$url" 2>/dev/null || echo "URL: $url"
}

cmd_status() {
  banner "Stack Status"
  require aws "https://aws.amazon.com/cli/"

  local status
  status=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")

  echo -e "  Stack:   ${BOLD}$STACK_NAME${NC}"
  echo -e "  Region:  $REGION"
  echo -e "  Status:  ${GREEN}$status${NC}"

  if [[ "$status" == "CREATE_COMPLETE" || "$status" == "UPDATE_COMPLETE" ]]; then
    local api_url
    api_url=$(get_api_url)
    echo -e "  API URL: $api_url"

    echo ""
    info "DynamoDB record counts..."
    for table in "${PROJECT}-books" "${PROJECT}-metrics"; do
      count=$(aws dynamodb scan --table-name "$table" \
        --select COUNT --region "$REGION" \
        --query 'Count' --output text 2>/dev/null || echo "error")
      echo "    $table: $count records"
    done

    echo ""
    info "Lambda function status..."
    for fn in non-vpc vpc provisioned small-pkg large-pkg; do
      state=$(aws lambda get-function \
        --function-name "${PROJECT}-api-${fn}" \
        --region "$REGION" \
        --query 'Configuration.State' \
        --output text 2>/dev/null || echo "not found")
      echo "    ${PROJECT}-api-${fn}: $state"
    done
  fi
}

cmd_destroy() {
  banner "⚠️  DESTROY — Deleting All Resources"
  warn "This will delete ALL resources including DynamoDB data!"
  read -rp "  Type 'yes' to confirm: " confirm
  [[ "$confirm" == "yes" ]] || { info "Cancelled."; exit 0; }

  info "Deleting CloudFormation stack..."
  aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"

  info "Waiting for deletion (~3 minutes)..."
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"

  ok "Stack deleted!"
  rm -f .api_url
  info "Note: CloudWatch log groups may need manual cleanup"
  info "  aws logs describe-log-groups --log-group-name-prefix /aws/lambda/coldstart-research"
}

cmd_clean_logs() {
  banner "Clean Leftover Log Groups"
  info "Deleting CloudWatch log groups..."
  for lg in \
    /aws/lambda/coldstart-research-api-non-vpc \
    /aws/lambda/coldstart-research-api-vpc \
    /aws/lambda/coldstart-research-api-provisioned \
    /aws/lambda/coldstart-research-api-small-pkg \
    /aws/lambda/coldstart-research-api-large-pkg \
    /aws/lambda/coldstart-research-seeder \
    /aws/lambda/coldstart-research-dynamic-provisioner; do
    aws logs delete-log-group --log-group-name "$lg" --region "$REGION" 2>/dev/null \
      && ok "Deleted: $lg" || info "Not found: $lg"
  done
}

cmd_full_run() {
  banner "FULL PIPELINE — Deploy → Seed → Test → Load → Extract → Train"
  warn "This runs everything end-to-end (~45-60 minutes)"
  read -rp "  Continue? (yes/no): " confirm
  [[ "$confirm" == "yes" ]] || exit 0

  cmd_deploy
  cmd_seed
  cmd_test
  cmd_load
  cmd_extract
  cmd_train
  ok "Full pipeline complete!"
}

cmd_quick_run() {
  banner "QUICK RUN — Synthetic data (no AWS required)"
  info "This runs ML/DL training on synthetic data without deploying to AWS"
  cmd_extract_synthetic
  cmd_train_synthetic
  python3 dynamic_provisioning.py --run-24h-sim
  ok "Quick run complete! Check model_outputs/"
}

cmd_help() {
  echo ""
  echo -e "${BOLD}Cold Start Research — Run Script${NC}"
  echo -e "Hybrid ML/DL: Quantile Regression + TFT + SHAP"
  echo ""
  echo -e "${BOLD}USAGE:${NC}  ./run_all.sh <command>"
  echo ""
  echo -e "${BOLD}AWS COMMANDS (require deployment):${NC}"
  echo -e "  ${GREEN}deploy${NC}             Deploy CloudFormation stack (~5-8 min)"
  echo -e "  ${GREEN}seed${NC}               Seed DynamoDB with 20 books"
  echo -e "  ${GREEN}test${NC}               Run cold start experiment via Lambda invoke"
  echo -e "  ${GREEN}load${NC}               Run k6 sine wave load test"
  echo -e "  ${GREEN}extract${NC}            Export metrics → cold_start_dataset.csv"
  echo -e "  ${GREEN}train${NC}              Train all ML/DL models"
  echo -e "  ${GREEN}provision${NC}          Run dynamic provisioner (TFT → Lambda)"
  echo -e "  ${GREEN}dashboard${NC}          Open CloudWatch dashboard"
  echo -e "  ${GREEN}status${NC}             Show stack + resource status"
  echo -e "  ${GREEN}destroy${NC}            Delete all AWS resources"
  echo -e "  ${GREEN}clean-logs${NC}         Delete leftover CloudWatch log groups"
  echo ""
  echo -e "${BOLD}LOCAL COMMANDS (no AWS required):${NC}"
  echo -e "  ${CYAN}quick-run${NC}          Extract synthetic data + train models locally"
  echo -e "  ${CYAN}extract-synthetic${NC}  Generate 2000 synthetic samples"
  echo -e "  ${CYAN}train-synthetic${NC}    Train models on synthetic data"
  echo -e "  ${CYAN}provision --simulate${NC}  Simulate 24h provisioning (no AWS)"
  echo ""
  echo -e "${BOLD}PIPELINE COMMANDS:${NC}"
  echo -e "  ${YELLOW}full-run${NC}           Run entire AWS pipeline end-to-end"
  echo ""
  echo -e "${BOLD}ENV OVERRIDES:${NC}"
  echo -e "  DURATION_MINUTES=60   k6 test duration (default: 30)"
  echo -e "  PEAK_VUS=50           k6 peak virtual users (default: 30)"
  echo -e "  START_HOUR=0          Simulated traffic start hour (default: 8)"
  echo -e "  LOOKBACK_HOURS=48     DynamoDB lookback for extract (default: 24)"
  echo -e "  TFT_EPOCHS=50         TFT training epochs (default: 30)"
  echo ""
  echo -e "${BOLD}EXAMPLES:${NC}"
  echo -e "  ./run_all.sh quick-run                       # No AWS, local only"
  echo -e "  ./run_all.sh deploy                          # Deploy to AWS"
  echo -e "  ./run_all.sh load DURATION_MINUTES=60        # 1h load test"
  echo -e "  ./run_all.sh provision --simulate            # 24h provisioning sim"
  echo ""
}

# ── DISPATCH ──────────────────────────────────────────────────────
case "${1:-help}" in
  deploy)             cmd_deploy ;;
  seed)               cmd_seed ;;
  test)               cmd_test ;;
  load)               cmd_load ;;
  extract)            cmd_extract ;;
  extract-synthetic)  cmd_extract_synthetic ;;
  train)              cmd_train ;;
  train-synthetic)    cmd_train_synthetic ;;
  provision)          cmd_provision "${2:-}" ;;
  dashboard)          cmd_dashboard ;;
  status)             cmd_status ;;
  destroy)            cmd_destroy ;;
  clean-logs)         cmd_clean_logs ;;
  full-run)           cmd_full_run ;;
  quick-run)          cmd_quick_run ;;
  help|--help|-h)     cmd_help ;;
  *)                  err "Unknown command: $1. Run: ./run_all.sh help" ;;
esac
