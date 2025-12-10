#!/usr/bin/env python3
"""Simple test for enhanced SAFLA endpoints."""

import requests
import json

def test_analyze_text():
    """Test just the analyze_text endpoint."""
    url = "https://safla.fly.dev/api/safla"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "analyze_text",
        "params": {
            "text": "SAFLA is amazing!",
            "analysis_type": "sentiment"
        }
    }
    
    print("Testing analyze_text endpoint...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response text: {response.text[:500]}")
        else:
            print(f"Error response: {response.text[:500]}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_analyze_text()