#!/usr/bin/env python3
"""
Test script for Qwen2.5-1.5B-Instruct chat functionality
"""

from mlx_lm import load, generate
import time
import json

def test_qwen_chat():
    """Test Qwen model with various chat scenarios"""
    
    # Load model
    print("Loading Qwen2.5-1.5B-Instruct...")
    start_time = time.time()
    model, tokenizer = load('./models/mlx/qwen2.5-1.5b-instruct')
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Test scenarios
    test_cases = [
        {
            "name": "AI Explanation",
            "prompt": "Explain artificial intelligence in simple terms",
            "max_tokens": 128
        },
        {
            "name": "Programming Help", 
            "prompt": "Write a simple Python function to calculate fibonacci numbers",
            "max_tokens": 200
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short story about a robot learning to paint",
            "max_tokens": 300
        },
        {
            "name": "Math Problem",
            "prompt": "Solve this math problem step by step: What is 15% of 240?",
            "max_tokens": 150
        },
        {
            "name": "Conversational",
            "prompt": "Hi! How are you doing today? What's your favorite hobby?",
            "max_tokens": 128
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        
        # Generate response
        start_gen = time.time()
        response = generate(
            model, 
            tokenizer, 
            test_case['prompt'], 
            max_tokens=test_case['max_tokens']
        )
        gen_time = time.time() - start_gen
        
        # Count tokens (approximate)
        response_tokens = len(response.split())
        tokens_per_sec = response_tokens / gen_time if gen_time > 0 else 0
        
        print(f"Response: {response}")
        print(f"Generation time: {gen_time:.2f}s")
        print(f"Tokens/sec: {tokens_per_sec:.2f}")
        print("-" * 50)
        
        # Store results
        results.append({
            "test_name": test_case['name'],
            "prompt": test_case['prompt'],
            "response": response,
            "generation_time": gen_time,
            "tokens_per_sec": tokens_per_sec,
            "response_tokens": response_tokens
        })
    
    # Calculate averages
    avg_gen_time = sum(r['generation_time'] for r in results) / len(results)
    avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
    total_tokens = sum(r['response_tokens'] for r in results)
    
    summary = {
        "model_name": "Qwen2.5-1.5B-Instruct",
        "load_time": load_time,
        "test_count": len(test_cases),
        "average_generation_time": avg_gen_time,
        "average_tokens_per_sec": avg_tokens_per_sec,
        "total_tokens_generated": total_tokens,
        "results": results
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {summary['model_name']}")
    print(f"Load time: {summary['load_time']:.2f}s")
    print(f"Tests run: {summary['test_count']}")
    print(f"Average generation time: {summary['average_generation_time']:.2f}s")
    print(f"Average tokens/sec: {summary['average_tokens_per_sec']:.2f}")
    print(f"Total tokens generated: {summary['total_tokens_generated']}")
    
    # Save results
    with open('qwen_chat_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to qwen_chat_test_results.json")
    return summary

if __name__ == "__main__":
    test_qwen_chat()