#!/bin/bash
#
# Nova Medicina CLI Examples
# ==========================
#
# This script demonstrates all CLI commands with real-world examples
# for the Nova Medicina medical analysis system.
#
# Usage: bash cli-examples.sh
# Or run individual sections by uncommenting specific commands

set -e  # Exit on error

echo "============================================"
echo "Nova Medicina CLI Examples"
echo "============================================"
echo ""

# ============================================
# 1. Basic Analysis Commands
# ============================================

echo "1. BASIC ANALYSIS COMMANDS"
echo "=========================="
echo ""

# Simple analysis with minimal input
echo "Example 1.1: Simple symptom analysis"
echo "Command: medai analyze \"diabetes\" -s \"increased thirst\" \"frequent urination\""
echo ""
# medai analyze "diabetes" -s "increased thirst" "frequent urination"

# Analysis with patient context
echo "Example 1.2: Analysis with patient age"
echo "Command: medai analyze \"hypertension\" -s \"headache\" \"dizziness\" --age 55"
echo ""
# medai analyze "hypertension" -s "headache" "dizziness" --age 55

# Analysis with full patient context
echo "Example 1.3: Analysis with complete patient context"
echo "Command: medai analyze \"asthma\" -s \"wheezing\" \"shortness of breath\" --age 32 --gender female --history \"allergies\""
echo ""
# medai analyze "asthma" -s "wheezing" "shortness of breath" --age 32 --gender female --history "allergies"

# Emergency analysis
echo "Example 1.4: Emergency situation analysis"
echo "Command: medai analyze \"chest pain\" -s \"severe chest pain\" \"shortness of breath\" \"sweating\" --emergency"
echo ""
# medai analyze "chest pain" -s "severe chest pain" "shortness of breath" "sweating" --emergency

echo ""

# ============================================
# 2. Interactive Mode
# ============================================

echo "2. INTERACTIVE MODE"
echo "==================="
echo ""

echo "Example 2.1: Interactive analysis (guided prompts)"
echo "Command: medai analyze --interactive"
echo "This will guide you through:"
echo "  - Condition input"
echo "  - Symptom collection"
echo "  - Patient context"
echo "  - Emergency status"
echo ""
# medai analyze --interactive

echo ""

# ============================================
# 3. Output Formats
# ============================================

echo "3. OUTPUT FORMATS"
echo "================="
echo ""

# JSON output
echo "Example 3.1: JSON output for programmatic processing"
echo "Command: medai analyze \"flu\" -s \"fever\" \"cough\" --output json"
echo ""
# medai analyze "flu" -s "fever" "cough" --output json

# JSON with file output
echo "Example 3.2: Save JSON to file"
echo "Command: medai analyze \"pneumonia\" -s \"fever\" \"cough\" \"chest pain\" --output json > analysis-result.json"
echo ""
# medai analyze "pneumonia" -s "fever" "cough" "chest pain" --output json > analysis-result.json

# Human-readable text output (default)
echo "Example 3.3: Human-readable text output"
echo "Command: medai analyze \"migraine\" -s \"severe headache\" \"nausea\" --output text"
echo ""
# medai analyze "migraine" -s "severe headache" "nausea" --output text

echo ""

# ============================================
# 4. Verification Commands
# ============================================

echo "4. CONFIDENCE VERIFICATION"
echo "=========================="
echo ""

# Basic verification
echo "Example 4.1: Check confidence score"
echo "Command: medai verify ANALYSIS_ID"
echo ""
# medai verify abc123xyz

# Detailed verification
echo "Example 4.2: Detailed confidence breakdown"
echo "Command: medai verify ANALYSIS_ID --detailed"
echo ""
# medai verify abc123xyz --detailed

# Verification with JSON output
echo "Example 4.3: JSON confidence data"
echo "Command: medai verify ANALYSIS_ID --output json"
echo ""
# medai verify abc123xyz --output json

echo ""

# ============================================
# 5. Provider Commands
# ============================================

echo "5. PROVIDER INTERACTION"
echo "======================="
echo ""

# Submit provider review
echo "Example 5.1: Approve analysis"
echo "Command: medai provider review ANALYSIS_ID --decision approved --comments \"Diagnosis confirmed\""
echo ""
# medai provider review abc123xyz --decision approved --comments "Diagnosis confirmed"

# Reject analysis
echo "Example 5.2: Reject with modifications"
echo "Command: medai provider review ANALYSIS_ID --decision rejected --comments \"Consider alternative diagnosis\""
echo ""
# medai provider review abc123xyz --decision rejected --comments "Consider alternative diagnosis"

# Modify analysis
echo "Example 5.3: Modify with suggestions"
echo "Command: medai provider review ANALYSIS_ID --decision modified --comments \"Updated treatment plan\" --modifications '{\"treatment\": \"Added antibiotic\"}'"
echo ""
# medai provider review abc123xyz --decision modified --comments "Updated treatment plan" --modifications '{"treatment": "Added antibiotic"}'

# Notify provider
echo "Example 5.4: Notify provider about analysis"
echo "Command: medai provider notify ANALYSIS_ID --provider-id PROVIDER_123"
echo ""
# medai provider notify abc123xyz --provider-id PROVIDER_123

# Urgent notification
echo "Example 5.5: Urgent provider notification"
echo "Command: medai provider notify ANALYSIS_ID --provider-id PROVIDER_123 --urgent"
echo ""
# medai provider notify abc123xyz --provider-id PROVIDER_123 --urgent

# List pending reviews
echo "Example 5.6: List pending provider reviews"
echo "Command: medai provider list --status pending"
echo ""
# medai provider list --status pending

echo ""

# ============================================
# 6. Configuration Commands
# ============================================

echo "6. CONFIGURATION MANAGEMENT"
echo "==========================="
echo ""

# Show current configuration
echo "Example 6.1: Display current configuration"
echo "Command: medai config show"
echo ""
# medai config show

# Set configuration values
echo "Example 6.2: Set minimum confidence threshold"
echo "Command: medai config set antiHallucination.minConfidence 0.80"
echo ""
# medai config set antiHallucination.minConfidence 0.80

# Set provider notification preference
echo "Example 6.3: Configure provider notifications"
echo "Command: medai config set provider.notificationChannels email,sms"
echo ""
# medai config set provider.notificationChannels email,sms

# Set API configuration
echo "Example 6.4: Configure API server"
echo "Command: medai config set api.port 3001"
echo ""
# medai config set api.port 3001

echo ""

# ============================================
# 7. Batch Processing
# ============================================

echo "7. BATCH PROCESSING"
echo "==================="
echo ""

# Batch analysis from file
echo "Example 7.1: Batch analysis from JSON file"
echo "Command: cat patients.json | medai analyze --batch"
echo ""
echo "Sample patients.json format:"
cat << 'EOF'
[
  {
    "condition": "diabetes",
    "symptoms": ["increased thirst", "frequent urination"],
    "patientContext": { "age": 45 }
  },
  {
    "condition": "hypertension",
    "symptoms": ["headache", "dizziness"],
    "patientContext": { "age": 60 }
  }
]
EOF
echo ""
# cat patients.json | medai analyze --batch

# Batch with parallel processing
echo "Example 7.2: Parallel batch processing"
echo "Command: medai analyze --batch --parallel 5 < patients.json"
echo ""
# medai analyze --batch --parallel 5 < patients.json

echo ""

# ============================================
# 8. Advanced Options
# ============================================

echo "8. ADVANCED OPTIONS"
echo "==================="
echo ""

# Analysis with medications
echo "Example 8.1: Include current medications"
echo "Command: medai analyze \"drug interaction\" -s \"nausea\" \"dizziness\" --medications \"warfarin,aspirin\""
echo ""
# medai analyze "drug interaction" -s "nausea" "dizziness" --medications "warfarin,aspirin"

# Analysis with medical history
echo "Example 8.2: Include medical history"
echo "Command: medai analyze \"chest pain\" -s \"chest pain\" \"shortness of breath\" --history \"heart disease,diabetes\""
echo ""
# medai analyze "chest pain" -s "chest pain" "shortness of breath" --history "heart disease,diabetes"

# Save analysis ID for later
echo "Example 8.3: Save analysis ID for verification"
echo "Command: ANALYSIS_ID=\$(medai analyze \"condition\" -s \"symptom\" --output json | jq -r '.id')"
echo ""
# ANALYSIS_ID=$(medai analyze "condition" -s "symptom" --output json | jq -r '.id')
# echo "Saved analysis ID: $ANALYSIS_ID"

echo ""

# ============================================
# 9. Piping and Chaining
# ============================================

echo "9. PIPING AND CHAINING"
echo "======================"
echo ""

# Analyze and verify in one command
echo "Example 9.1: Analyze and verify confidence"
echo "Command: medai analyze \"flu\" -s \"fever\" | tee analysis.txt && medai verify \$(cat analysis.txt | grep 'ID:' | awk '{print \$2}')"
echo ""

# Analyze and auto-notify provider if low confidence
echo "Example 9.2: Auto-notify on low confidence"
cat << 'EOF'
#!/bin/bash
RESULT=$(medai analyze "condition" -s "symptom" --output json)
CONFIDENCE=$(echo $RESULT | jq -r '.confidenceScore.overall')
ANALYSIS_ID=$(echo $RESULT | jq -r '.id')

if (( $(echo "$CONFIDENCE < 0.75" | bc -l) )); then
  echo "Low confidence detected, notifying provider..."
  medai provider notify $ANALYSIS_ID --provider-id PROVIDER_123 --urgent
fi
EOF
echo ""

# Filter high-confidence analyses
echo "Example 9.3: Filter analyses by confidence"
echo "Command: medai analyze --batch < patients.json | jq 'select(.confidenceScore.overall > 0.85)'"
echo ""

echo ""

# ============================================
# 10. Real-World Workflow Examples
# ============================================

echo "10. REAL-WORLD WORKFLOWS"
echo "========================"
echo ""

echo "Example 10.1: Emergency Room Triage"
cat << 'EOF'
#!/bin/bash
# Quick triage for emergency patients
medai analyze "trauma" \
  -s "severe bleeding" "altered consciousness" \
  --age 35 \
  --emergency \
  --provider-id ER_DOC_001 \
  --notify-on-complete
EOF
echo ""

echo "Example 10.2: Telemedicine Consultation"
cat << 'EOF'
#!/bin/bash
# Analyze patient symptoms remotely
ANALYSIS=$(medai analyze "respiratory infection" \
  -s "persistent cough" "fever" "fatigue" \
  --age 42 \
  --gender female \
  --history "asthma" \
  --medications "albuterol" \
  --output json)

# Check if provider review needed
if [ $(echo $ANALYSIS | jq -r '.requiresProviderReview') == "true" ]; then
  echo "Scheduling video consultation..."
  medai provider notify $(echo $ANALYSIS | jq -r '.id') --urgent
else
  echo "Self-care recommendations provided"
fi
EOF
echo ""

echo "Example 10.3: Clinical Decision Support"
cat << 'EOF'
#!/bin/bash
# Support clinical decision-making
medai analyze "differential diagnosis" \
  -s "fever" "cough" "chest pain" "shortness of breath" \
  --age 68 \
  --history "COPD,smoking history" \
  --medications "tiotropium,prednisone" \
  --output json | jq '{
    primaryDiagnosis: .diagnosis.condition,
    confidence: .confidenceScore.overall,
    differentials: .differentialDiagnoses,
    urgentActions: [.recommendations[] | select(.priority == "urgent")]
  }'
EOF
echo ""

# ============================================
# Help and Documentation
# ============================================

echo ""
echo "GETTING HELP"
echo "============"
echo ""
echo "View help for any command:"
echo "  medai --help"
echo "  medai analyze --help"
echo "  medai verify --help"
echo "  medai provider --help"
echo ""
echo "View version information:"
echo "  medai --version"
echo ""
echo "View configuration:"
echo "  medai config show"
echo ""

echo "============================================"
echo "Examples script completed!"
echo "============================================"
echo ""
echo "To run actual commands, uncomment the lines in this script."
echo "For more examples, see:"
echo "  - examples/basic-usage.js"
echo "  - examples/api-client.js"
echo "  - examples/advanced-workflows.js"
echo "  - docs/TUTORIALS.md"
echo ""
