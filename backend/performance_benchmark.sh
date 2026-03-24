#!/bin/bash

# Performance Benchmarking Script for Model Validation
# Measures inference time, explainability endpoints, and logs results

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost:8000"
OUTPUT_FILE="performance_results.txt"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Test prompts
SIMPLE_PROMPT="What is 245 + 387?"
MULTI_STEP_PROMPT="Calculate (15 × 8) + 42"
PERCENTAGE_PROMPT="What is 30% of 250?"

echo "================================================"
echo "Performance Benchmarking - $TIMESTAMP"
echo "================================================"
echo ""

# Clear previous results
> $OUTPUT_FILE

# Function to measure endpoint performance
measure_endpoint() {
    local endpoint=$1
    local method=$2
    local data=$3
    local description=$4
    
    echo -n "Testing $description... "
    
    # Measure time using curl's built-in timing
    local start=$(date +%s.%N)
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{time_total}" -X POST "$BASE_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -s -w "\n%{time_total}" "$BASE_URL$endpoint")
    fi
    
    local end=$(date +%s.%N)
    
    # Extract time from curl output (last line)
    local curl_time=$(echo "$response" | tail -n 1)
    local body=$(echo "$response" | sed '$d')
    
    # Calculate total time
    local total_time=$(echo "$end - $start" | bc)
    
    # Check if request was successful
    if echo "$body" | grep -q "error"; then
        echo -e "${RED}FAILED${NC}"
        echo "$description: FAILED - $(echo $body | grep -o '"error":"[^"]*"')" >> $OUTPUT_FILE
    else
        echo -e "${GREEN}OK${NC} (${curl_time}s)"
        echo "$description: ${curl_time}s" >> $OUTPUT_FILE
    fi
    
    echo "$curl_time"
}

# ============================================================================
# Test 1: Inference Time - Base Model
# ============================================================================
echo ""
echo "Test 1: Base Model Inference"
echo "----------------------------"
INFERENCE_TIME_BASE=$(measure_endpoint \
    "/generate/" \
    "POST" \
    "{\"model_id\":\"gemma-base\",\"prompt\":\"$SIMPLE_PROMPT\",\"max_new_tokens\":512}" \
    "Base model inference")

# ============================================================================
# Test 2: Inference Time - Fine-tuned Model
# ============================================================================
echo ""
echo "Test 2: Fine-tuned Model Inference"
echo "-----------------------------------"
INFERENCE_TIME_FT=$(measure_endpoint \
    "/generate/" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\",\"max_new_tokens\":512}" \
    "Fine-tuned model inference")

# ============================================================================
# Test 3: Token Confidence
# ============================================================================
echo ""
echo "Test 3: Explainability - Token Confidence"
echo "-----------------------------------------"
CONFIDENCE_TIME=$(measure_endpoint \
    "/explain/confidence" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\",\"response\":\"Step 1: Add 245 and 387\"}" \
    "Token confidence analysis")

# ============================================================================
# Test 4: Attention Weights
# ============================================================================
echo ""
echo "Test 4: Explainability - Attention Weights"
echo "------------------------------------------"
ATTENTION_TIME=$(measure_endpoint \
    "/explain/attention" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\",\"attn_layer\":0,\"attn_head\":0}" \
    "Attention weights extraction")

# ============================================================================
# Test 5: Logit Lens
# ============================================================================
echo ""
echo "Test 5: Explainability - Logit Lens"
echo "-----------------------------------"
LOGIT_LENS_TIME=$(measure_endpoint \
    "/explain/logit-lens" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\",\"response\":\"Step 1: Add 245 and 387. Step 2: The sum is 632.\"}" \
    "Logit lens analysis")

# ============================================================================
# Test 6: Attribution
# ============================================================================
echo ""
echo "Test 6: Explainability - Attribution"
echo "------------------------------------"
ATTRIBUTION_TIME=$(measure_endpoint \
    "/explain/attribution" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\",\"response\":\"Step 1: Add 245 and 387\"}" \
    "Gradient attribution")

# ============================================================================
# Test 7: Hidden States
# ============================================================================
echo ""
echo "Test 7: Explainability - Hidden States"
echo "--------------------------------------"
HIDDEN_STATES_TIME=$(measure_endpoint \
    "/explain/hidden-states" \
    "POST" \
    "{\"model_id\":\"gemma-finetuned\",\"prompt\":\"$SIMPLE_PROMPT\"}" \
    "Hidden states analysis")

# ============================================================================
# Summary Table
# ============================================================================
echo ""
echo "================================================"
echo "PERFORMANCE SUMMARY"
echo "================================================"
echo ""

printf "%-30s %-10s %-10s %-10s\n" "Function" "Target" "Actual" "Status"
echo "────────────────────────────────────────────────────────────────"

# Function to check status
check_status() {
    local actual=$1
    local target=$2
    
    # Use awk for floating point comparison (more portable)
    local result=$(awk -v a="$actual" -v t="$target" 'BEGIN { print (a < t) ? "pass" : "fail" }')
    
    if [ "$result" = "pass" ]; then
        echo "✅"
    else
        echo "❌"
    fi
}

# Inference time check
INFERENCE_TARGET=10.0
INFERENCE_STATUS=$(check_status $INFERENCE_TIME_FT $INFERENCE_TARGET)
printf "%-30s %-10s %-10s %-10s\n" "Inference time" "<10s" "${INFERENCE_TIME_FT}s" "$INFERENCE_STATUS"

# Logit lens check
LOGIT_TARGET=5.0
LOGIT_STATUS=$(check_status $LOGIT_LENS_TIME $LOGIT_TARGET)
printf "%-30s %-10s %-10s %-10s\n" "Logit Lens render" "<5s" "${LOGIT_LENS_TIME}s" "$LOGIT_STATUS"

# Attention check
ATTENTION_TARGET=3.0
ATTENTION_STATUS=$(check_status $ATTENTION_TIME $ATTENTION_TARGET)
printf "%-30s %-10s %-10s %-10s\n" "Attention extraction" "<3s" "${ATTENTION_TIME}s" "$ATTENTION_STATUS"

# Attribution check
ATTRIBUTION_TARGET=5.0
ATTRIBUTION_STATUS=$(check_status $ATTRIBUTION_TIME $ATTRIBUTION_TARGET)
printf "%-30s %-10s %-10s %-10s\n" "Attribution analysis" "<5s" "${ATTRIBUTION_TIME}s" "$ATTRIBUTION_STATUS"

echo ""
echo "Full results saved to: $OUTPUT_FILE"
echo ""

# ============================================================================
# Save detailed results
# ============================================================================
{
    echo ""
    echo "================================================"
    echo "DETAILED PERFORMANCE RESULTS"
    echo "Timestamp: $TIMESTAMP"
    echo "================================================"
    echo ""
    echo "INFERENCE TIMES:"
    echo "  Base model:       ${INFERENCE_TIME_BASE}s"
    echo "  Fine-tuned model: ${INFERENCE_TIME_FT}s"
    echo ""
    echo "EXPLAINABILITY ENDPOINTS:"
    echo "  Token confidence: ${CONFIDENCE_TIME}s"
    echo "  Attention:        ${ATTENTION_TIME}s"
    echo "  Logit lens:       ${LOGIT_LENS_TIME}s"
    echo "  Attribution:      ${ATTRIBUTION_TIME}s"
    echo "  Hidden states:    ${HIDDEN_STATES_TIME}s"
    echo ""
    echo "VALIDATION TARGETS:"
    echo "  Inference time:   ${INFERENCE_STATUS} (target: <${INFERENCE_TARGET}s, actual: ${INFERENCE_TIME_FT}s)"
    echo "  Logit lens:       ${LOGIT_STATUS} (target: <${LOGIT_TARGET}s, actual: ${LOGIT_LENS_TIME}s)"
    echo "  Attention:        ${ATTENTION_STATUS} (target: <${ATTENTION_TARGET}s, actual: ${ATTENTION_TIME}s)"
    echo "  Attribution:      ${ATTRIBUTION_STATUS} (target: <${ATTRIBUTION_TARGET}s, actual: ${ATTRIBUTION_TIME}s)"
    echo ""
} >> $OUTPUT_FILE

echo "Benchmark complete!"
