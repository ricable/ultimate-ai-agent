#!/bin/bash

# Example script for scraping clinical trials using auto-browser

# Initialize config if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Initializing config file..."
    ./auto-browser init config.yaml
fi

# Example trial IDs to scrape
TRIAL_IDS=(
    "2022-500814-24-00"
    "2023-509462-38-00"
    "2023-501632-27-00"
)

# Function to process a trial with retries
process_trial() {
    local trial_id=$1
    local max_attempts=3
    local attempt=1
    local wait_time=5

    while [ $attempt -le $max_attempts ]; do
        echo "Processing trial: $trial_id (Attempt $attempt of $max_attempts)"
        
        if ./auto-browser process \
            --site clinical_trials \
            --output "${trial_id}" \
            --verbose \
            "https://euclinicaltrials.eu/ctis-public/view/$trial_id"; then
            echo "Successfully processed trial: $trial_id"
            return 0
        else
            echo "Attempt $attempt failed for trial: $trial_id"
            if [ $attempt -lt $max_attempts ]; then
                echo "Waiting $wait_time seconds before retry..."
                sleep $wait_time
                wait_time=$((wait_time * 2))
            fi
        fi
        attempt=$((attempt + 1))
    done

    echo "Failed to process trial after $max_attempts attempts: $trial_id"
    return 1
}

# Process each trial
failed_trials=()
for id in "${TRIAL_IDS[@]}"; do
    if ! process_trial "$id"; then
        failed_trials+=("$id")
    fi
done

# Report results
echo "Processing complete!"
if [ ${#failed_trials[@]} -eq 0 ]; then
    echo "All trials processed successfully"
else
    echo "Failed to process the following trials:"
    printf '%s\n' "${failed_trials[@]}"
fi

echo "Check output/trials/ for results"
