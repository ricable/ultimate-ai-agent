#!/usr/bin/env python3
"""
MCP Tool Calling Validator
Validates fine-tuned model on MCP tool accuracy
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import re
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPToolValidator:
    """Validate MCP tool calling accuracy"""

    def __init__(self, model_path: str, base_model: str = "microsoft/Phi-4"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check if this is a PEFT model
        peft_config_path = Path(self.model_path) / "adapter_config.json"

        if peft_config_path.exists():
            # Load base model first
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # Load PEFT adapter
            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.model = self.model.merge_and_unload()  # Merge for inference
        else:
            # Load full model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def parse_tool_call(self, response: str) -> Dict:
        """Parse tool call from response"""
        # Look for <tool_use>...</tool_use> pattern
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)

        if not match:
            return {"found": False}

        tool_call_str = match.group(1).strip()

        # Extract tool name and parameters
        # Format: tool_name(param1='value1', param2='value2')
        tool_match = re.match(r'(\w+)\((.*)\)', tool_call_str)

        if not tool_match:
            return {"found": True, "parsed": False}

        tool_name = tool_match.group(1)
        params_str = tool_match.group(2)

        # Parse parameters
        params = {}
        if params_str:
            # Simple parameter parsing
            param_pattern = r"(\w+)=(['\"])(.*?)\2"
            for param_match in re.finditer(param_pattern, params_str):
                key = param_match.group(1)
                value = param_match.group(3)
                params[key] = value

        return {
            "found": True,
            "parsed": True,
            "tool_name": tool_name,
            "parameters": params
        }

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response from model"""
        formatted_prompt = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response

    def validate_tool_accuracy(self, test_cases: List[Dict]) -> Dict:
        """Validate tool calling accuracy on test cases"""
        logger.info(f"Validating on {len(test_cases)} test cases...")

        results = {
            "total": len(test_cases),
            "correct_tool": 0,
            "correct_params": 0,
            "tool_found": 0,
            "tool_parsed": 0,
            "accuracy": 0.0,
            "param_accuracy": 0.0,
            "details": []
        }

        for i, test_case in enumerate(test_cases):
            query = test_case["query"]
            expected_tool = test_case["expected_tool"]
            expected_params = test_case.get("expected_params", {})

            # Generate response
            response = self.generate_response(query)

            # Parse tool call
            parsed = self.parse_tool_call(response)

            # Check correctness
            tool_found = parsed.get("found", False)
            tool_parsed = parsed.get("parsed", False)
            correct_tool = tool_parsed and parsed.get("tool_name") == expected_tool
            correct_params = False

            if correct_tool and expected_params:
                # Check if all expected params are present
                actual_params = parsed.get("parameters", {})
                correct_params = all(
                    key in actual_params for key in expected_params.keys()
                )

            # Update counts
            if tool_found:
                results["tool_found"] += 1
            if tool_parsed:
                results["tool_parsed"] += 1
            if correct_tool:
                results["correct_tool"] += 1
            if correct_params:
                results["correct_params"] += 1

            # Record details
            results["details"].append({
                "query": query,
                "expected_tool": expected_tool,
                "response": response,
                "parsed": parsed,
                "correct_tool": correct_tool,
                "correct_params": correct_params
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_cases)} test cases")

        # Calculate final metrics
        results["accuracy"] = (results["correct_tool"] / results["total"]) * 100
        results["param_accuracy"] = (results["correct_params"] / results["total"]) * 100
        results["parse_rate"] = (results["tool_parsed"] / results["tool_found"]) * 100 if results["tool_found"] > 0 else 0

        logger.info("\nValidation Results:")
        logger.info(f"  Tool Detection: {results['tool_found']}/{results['total']} ({results['tool_found']/results['total']*100:.1f}%)")
        logger.info(f"  Tool Parsing: {results['tool_parsed']}/{results['tool_found']} ({results['parse_rate']:.1f}%)")
        logger.info(f"  Tool Accuracy: {results['correct_tool']}/{results['total']} ({results['accuracy']:.1f}%)")
        logger.info(f"  Param Accuracy: {results['correct_params']}/{results['total']} ({results['param_accuracy']:.1f}%)")

        return results


def main():
    """Run validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate MCP tool calling")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str, default="microsoft/Phi-4", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="/app/validation/results", help="Output directory")

    args = parser.parse_args()

    # Create validator
    validator = MCPToolValidator(
        model_path=args.model_path,
        base_model=args.base_model
    )

    # Load test cases
    test_cases = [
        {"query": "Read the package.json file", "expected_tool": "read", "expected_params": {"file_path": "package.json"}},
        {"query": "Create a new utils file", "expected_tool": "write", "expected_params": {"file_path": "utils.ts"}},
        {"query": "Run the tests", "expected_tool": "bash", "expected_params": {"command": "npm test"}},
        {"query": "Search for TODO comments", "expected_tool": "grep", "expected_params": {"pattern": "TODO"}},
        {"query": "List all TypeScript files", "expected_tool": "glob", "expected_params": {"pattern": "**/*.ts"}},
        {"query": "Update the port to 8080", "expected_tool": "edit", "expected_params": {"old_string": "3000", "new_string": "8080"}},
        {"query": "Show me the config file", "expected_tool": "read", "expected_params": {"file_path": "config"}},
        {"query": "Install dependencies", "expected_tool": "bash", "expected_params": {"command": "npm install"}},
        {"query": "Find all API endpoints", "expected_tool": "grep", "expected_params": {"pattern": "api"}},
        {"query": "Show all JavaScript files", "expected_tool": "glob", "expected_params": {"pattern": "**/*.js"}},
    ]

    # Validate
    results = validator.validate_tool_accuracy(test_cases)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "mcp_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Tool Accuracy: {results['accuracy']:.1f}%")
    logger.info(f"Parameter Accuracy: {results['param_accuracy']:.1f}%")

    if results['accuracy'] >= 85:
        logger.info("✅ EXCELLENT - Model is ready for deployment")
    elif results['accuracy'] >= 75:
        logger.info("✓ GOOD - Model performs well")
    elif results['accuracy'] >= 65:
        logger.info("⚠ ACCEPTABLE - Consider more training")
    else:
        logger.info("❌ NEEDS IMPROVEMENT - More training required")


if __name__ == "__main__":
    main()
