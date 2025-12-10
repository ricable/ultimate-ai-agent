#!/bin/bash
# Simple threshold calculator using bash and awk
# Analyzes CSV data to calculate basic statistical thresholds

CSV_FILE="$1"
if [ -z "$CSV_FILE" ] || [ ! -f "$CSV_FILE" ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

echo "Analyzing CSV file: $CSV_FILE"
echo "Total lines: $(wc -l < "$CSV_FILE")"

# Extract header
HEADER=$(head -1 "$CSV_FILE")
echo "Columns found: $(echo "$HEADER" | tr ';' '\n' | wc -l)"

# Create output directory
mkdir -p calculated_thresholds
OUTPUT_DIR="calculated_thresholds"

# Function to calculate statistics for a column
calculate_column_stats() {
    local col_num=$1
    local col_name=$2
    
    echo "Analyzing column $col_num: $col_name"
    
    # Extract column data (skip header, remove zeros and empty values)
    tail -n +2 "$CSV_FILE" | cut -d';' -f"$col_num" | \
    awk '$1 != "" && $1 != 0 && $1 !~ /^[^0-9]*$/ {print $1}' | \
    sort -n > "${OUTPUT_DIR}/col_${col_num}_data.tmp"
    
    local data_file="${OUTPUT_DIR}/col_${col_num}_data.tmp"
    local count=$(wc -l < "$data_file")
    
    if [ "$count" -lt 10 ]; then
        echo "  Skipping $col_name - insufficient data ($count points)"
        rm -f "$data_file"
        return
    fi
    
    # Calculate basic statistics
    local min=$(head -1 "$data_file")
    local max=$(tail -1 "$data_file")
    local median_line=$(( (count + 1) / 2 ))
    local median=$(sed -n "${median_line}p" "$data_file")
    local q25_line=$(( count / 4 ))
    local q75_line=$(( 3 * count / 4 ))
    local q95_line=$(( 95 * count / 100 ))
    
    local q25=$(sed -n "${q25_line}p" "$data_file")
    local q75=$(sed -n "${q75_line}p" "$data_file")
    local q95=$(sed -n "${q95_line}p" "$data_file")
    
    # Calculate mean and std using awk
    local stats=$(awk '{sum+=$1; sumsq+=$1*$1} END {
        mean=sum/NR; 
        std=sqrt((sumsq-sum*sum/NR)/(NR-1)); 
        printf "%.6f %.6f", mean, std
    }' "$data_file")
    
    local mean=$(echo "$stats" | cut -d' ' -f1)
    local std=$(echo "$stats" | cut -d' ' -f2)
    
    # Calculate thresholds based on column type
    local normal_min normal_max warning_threshold critical_threshold anomaly_threshold
    
    if [[ "$col_name" == *"AVAILABILITY"* ]] || [[ "$col_name" == *"_SR" ]] || [[ "$col_name" == *"SUCCESS"* ]]; then
        # Availability/Success Rate thresholds
        normal_min=$(echo "$q25 90" | awk '{print ($1 > $2) ? $1 : $2}')
        normal_max="100.0"
        warning_threshold=$(echo "$q75 95" | awk '{print ($1 > $2) ? $1 : $2}')
        critical_threshold=$(echo "$q25 90" | awk '{print ($1 > $2) ? $1 : $2}')
        anomaly_threshold=$(echo "$mean $std" | awk '{print $1 - 2*$2}')
        
    elif [[ "$col_name" == *"DROP"* ]] || [[ "$col_name" == *"ERROR"* ]] || [[ "$col_name" == *"LOSS"* ]] || [[ "$col_name" == *"BLER"* ]]; then
        # Error Rate thresholds
        normal_min="0.0"
        normal_max=$(echo "$q75 5" | awk '{print ($1 < $2) ? $1 : $2}')
        warning_threshold=$(echo "$q75 3" | awk '{print ($1 < $2) ? $1 : $2}')
        critical_threshold=$(echo "$q95 5" | awk '{print ($1 < $2) ? $1 : $2}')
        anomaly_threshold=$(echo "$mean $std" | awk '{print $1 + 2*$2}')
        
    elif [[ "$col_name" == *"SINR"* ]]; then
        # SINR thresholds based on LTE standards
        normal_min="0.0"
        normal_max="30.0"
        warning_threshold="5.0"
        critical_threshold="3.0"
        anomaly_threshold="1.0"
        
    elif [[ "$col_name" == *"RSSI"* ]]; then
        # RSSI thresholds
        normal_min="-130.0"
        normal_max="-80.0"
        warning_threshold="-120.0"
        critical_threshold="-125.0"
        anomaly_threshold="-130.0"
        
    else
        # Generic statistical thresholds
        local iqr=$(echo "$q75 $q25" | awk '{print $1 - $2}')
        normal_min=$(echo "$q25 $iqr" | awk '{print $1 - 1.5*$2}')
        normal_max=$(echo "$q75 $iqr" | awk '{print $1 + 1.5*$2}')
        warning_threshold="$q95"
        critical_threshold=$(echo "$mean $std" | awk '{print $1 + 2*$2}')
        anomaly_threshold=$(echo "$mean $std" | awk '{print $1 + 3*$2}')
    fi
    
    # Ensure minimum value is not negative for certain metrics
    if [[ "$col_name" != *"RSSI"* ]] && [[ "$col_name" != *"LATENCY"* ]]; then
        normal_min=$(echo "$normal_min 0" | awk '{print ($1 > $2) ? $1 : $2}')
        anomaly_threshold=$(echo "$anomaly_threshold 0" | awk '{print ($1 > $2) ? $1 : $2}')
    fi
    
    echo "  Stats: count=$count, min=$min, max=$max, mean=$mean, std=$std"
    echo "  Quartiles: Q25=$q25, median=$median, Q75=$q75, Q95=$q95"
    echo "  Thresholds: normal=[$normal_min, $normal_max], warning=$warning_threshold, critical=$critical_threshold, anomaly=$anomaly_threshold"
    
    # Output to JSON-like format
    cat >> "${OUTPUT_DIR}/thresholds.json" << EOF
    "$col_name": {
        "statistics": {
            "count": $count,
            "min": $min,
            "max": $max,
            "mean": $mean,
            "std": $std,
            "q25": $q25,
            "median": $median,
            "q75": $q75,
            "q95": $q95
        },
        "thresholds": {
            "normal_min": $normal_min,
            "normal_max": $normal_max,
            "warning_threshold": $warning_threshold,
            "critical_threshold": $critical_threshold,
            "anomaly_threshold": $anomaly_threshold
        }
    },
EOF
    
    # Clean up temporary file
    rm -f "$data_file"
}

# Initialize JSON output
echo "{" > "${OUTPUT_DIR}/thresholds.json"

# Process each column
col_num=1
echo "$HEADER" | tr ';' '\n' | while IFS= read -r col_name; do
    if [ "$col_num" -gt 7 ]; then  # Skip identifier columns (first 7)
        calculate_column_stats "$col_num" "$col_name"
    fi
    col_num=$((col_num + 1))
done

# Close JSON
sed -i '' '$s/,$//' "${OUTPUT_DIR}/thresholds.json"  # Remove last comma
echo "}" >> "${OUTPUT_DIR}/thresholds.json"

echo ""
echo "Analysis complete! Results saved to:"
echo "  ${OUTPUT_DIR}/thresholds.json"