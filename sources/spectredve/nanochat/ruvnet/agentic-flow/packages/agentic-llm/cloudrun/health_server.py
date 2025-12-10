#!/usr/bin/env python3
"""
Health check server for Cloud Run
Provides status endpoint for monitoring
Starts training in background thread after server is healthy
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

STATUS_FILE = Path("/app/results/training_status.json")
TRAINING_STARTED = False


@app.route("/")
def index():
    """Root endpoint"""
    return jsonify({
        "service": "phi4-finetuning",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health")
def health():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200


@app.route("/status")
def status():
    """Training status endpoint"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            status_data = json.load(f)
        return jsonify(status_data), 200
    else:
        return jsonify({
            "status": "initializing",
            "message": "Training not yet started"
        }), 200


@app.route("/results")
def results():
    """Get training results if available"""
    results_dir = Path("/app/benchmarks/results")

    if not results_dir.exists():
        return jsonify({"error": "Results not available yet"}), 404

    available_files = list(results_dir.glob("*.json"))

    return jsonify({
        "results_available": len(available_files) > 0,
        "files": [f.name for f in available_files],
        "path": str(results_dir)
    }), 200


def start_training_async():
    """Start training in background thread after 30 second delay"""
    global TRAINING_STARTED

    # Wait 30 seconds to ensure health checks pass
    print("⏳ Waiting 30 seconds before starting training...")
    time.sleep(30)

    print("🚀 Starting training in background...")
    TRAINING_STARTED = True

    try:
        # Run training script
        process = subprocess.Popen(
            ["python3", "/app/cloudrun/cloud_runner.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output to log file
        with open("/app/logs/training.log", "w") as log_file:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

        process.wait()
        print(f"✅ Training completed with exit code: {process.returncode}")

    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    # Start training in background thread
    training_thread = threading.Thread(target=start_training_async, daemon=True)
    training_thread.start()
    print(f"✅ Health server starting on port {port}")
    print(f"📋 Training will start in 30 seconds...")

    # Run Flask server in main thread
    app.run(host="0.0.0.0", port=port, debug=False)
