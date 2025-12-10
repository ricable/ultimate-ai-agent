#!/usr/bin/env python3
"""
Claude Code Polyglot Benchmark
Adapted from aider's benchmark.py to work with Claude Code CLI
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import re

# Configuration
BENCHMARK_DNAME = Path("tmp.claude_benchmarks")
EXERCISES_DIR = "polyglot-benchmark"

def create_benchmark_dir(name="claude-polyglot"):
    """Create a timestamped benchmark directory"""
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dirname = BENCHMARK_DNAME / f"{now}--{name}"
    return dirname

def get_exercise_dirs(base_dir, languages=None):
    """Get all exercise directories for specified languages"""
    base_dir = Path(base_dir)
    
    # Get available language dirs
    lang_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    # Filter to requested languages if specified
    if languages:
        requested = set(lang.strip().lower() for lang in languages.split(","))
        lang_dirs = [d for d in lang_dirs if d.name.lower() in requested]
        
    if not lang_dirs:
        print(f"No matching language directories found for: {languages}")
        return []
    
    # Get all exercise dirs under exercises/practice for each language
    exercise_dirs = []
    for lang_dir in lang_dirs:
        practice_dir = lang_dir / "exercises" / "practice"
        if practice_dir.exists():
            exercise_dirs.extend(d for d in practice_dir.iterdir() if d.is_dir())
    
    return exercise_dirs

def setup_test_directory(original_dname, testdir, exercise_dirs):
    """Copy exercise files to test directory"""
    if testdir.exists():
        shutil.rmtree(testdir)
    
    print(f"Setting up test directory: {testdir}")
    os.makedirs(testdir, exist_ok=True)
    
    # Copy exercise structures
    for lang_dir in original_dname.iterdir():
        if not lang_dir.is_dir():
            continue
        practice_dir = lang_dir / "exercises" / "practice"
        if practice_dir.exists():
            dest_lang_dir = testdir / lang_dir.name / "exercises" / "practice"
            os.makedirs(dest_lang_dir.parent, exist_ok=True)
            shutil.copytree(practice_dir, dest_lang_dir)

def load_exercise_config(exercise_dir):
    """Load exercise configuration from .meta/config.json"""
    config_file = exercise_dir / ".meta/config.json"
    if not config_file.exists():
        raise ValueError(f"No config file found: {config_file}")
    
    with open(config_file) as f:
        config = json.loads(f.read())
    
    # Get file sets from config
    test_files = config.get("files", {}).get("test", [])
    example_files = config.get("files", {}).get("example", [])
    solution_files = set(config.get("files", {}).get("solution", []))
    
    # Files to ignore (not for LLM to edit)
    ignore_files = {
        "CMakeLists.txt", "Cargo.toml", "package.json", "build.gradle"
    }
    
    # Add .meta and .docs directories to ignore
    ignore_files.update(str(p.relative_to(exercise_dir)) for p in exercise_dir.glob(".meta/**/*"))
    ignore_files.update(str(p.relative_to(exercise_dir)) for p in exercise_dir.glob(".docs/**/*"))
    ignore_files.update(test_files)
    ignore_files.update(example_files)
    
    # Remove ignore files from solution set
    solution_files.difference_update(ignore_files)
    
    return {
        "solution_files": list(solution_files),
        "test_files": test_files,
        "example_files": example_files,
        "ignore_files": ignore_files
    }

def get_instructions(exercise_dir):
    """Get exercise instructions from .docs files"""
    instructions = ""
    
    introduction = exercise_dir / ".docs/introduction.md"
    if introduction.exists():
        instructions += introduction.read_text() + "\n\n"
    
    instructions_file = exercise_dir / ".docs/instructions.md"
    if instructions_file.exists():
        instructions += instructions_file.read_text() + "\n\n"
    
    instructions_append = exercise_dir / ".docs/instructions.append.md"
    if instructions_append.exists():
        instructions += instructions_append.read_text() + "\n\n"
    
    return instructions

def run_claude_code(instructions, solution_files, exercise_dir, model="sonnet"):
    """Run Claude Code on the exercise"""
    
    # Create the prompt with file contents
    file_list = ", ".join(Path(f).name for f in solution_files)
    
    # Read current file contents
    file_contents = {}
    for file_path in solution_files:
        full_path = exercise_dir / file_path
        if full_path.exists():
            try:
                file_contents[file_path] = full_path.read_text()
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                file_contents[file_path] = "# File could not be read"
    
    # Build comprehensive prompt
    full_instructions = f"""
{instructions}

Please implement the solution by modifying these files: {file_list}

Current file contents:
"""
    
    for file_path, content in file_contents.items():
        full_instructions += f"\n--- {file_path} ---\n{content}\n"
    
    full_instructions += """

Make sure your implementation:
1. Follows the exact function/class signatures expected by the tests
2. Handles all edge cases mentioned in the instructions  
3. Is syntactically correct and follows language conventions
4. Will pass all the provided unit tests
5. Provides complete, working implementations (not just comments or placeholders)

Please provide the complete updated file contents for each file that needs changes.
"""
    
    print(f"Running Claude Code on {exercise_dir.name}...")
    print(f"Solution files: {file_list}")
    
    # Build Claude Code command  
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--dangerously-skip-permissions",
        full_instructions
    ]
    
    # Run Claude Code
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=exercise_dir,
            timeout=300  # 5 minute timeout
        )
        duration = time.time() - start_time
        
        # Parse Claude's response and update files
        if result.returncode == 0:
            try:
                parse_and_update_files(result.stdout, solution_files, exercise_dir)
            except Exception as e:
                print(f"Warning: Failed to parse Claude response: {e}")
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Claude Code execution timed out",
            "duration": 300,
            "returncode": -1
        }

def parse_and_update_files(claude_output, solution_files, exercise_dir):
    """Parse Claude's output and update solution files"""
    # This is a simple parser - Claude usually provides code in ```language blocks
    import re
    
    for file_path in solution_files:
        file_name = Path(file_path).name
        
        # Look for code blocks that mention this file
        pattern = rf"```(?:python|rust|go|javascript|java|cpp|c\+\+)?\s*(?:#.*?{re.escape(file_name)}.*?)?\n(.*?)\n```"
        matches = re.findall(pattern, claude_output, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Use the last match (most likely to be the final version)
            new_content = matches[-1].strip()
            
            # Write updated content
            full_path = exercise_dir / file_path
            if full_path.exists():
                full_path.write_text(new_content)
                print(f"Updated {file_path}")
        else:
            # Try to find content by file name mention
            lines = claude_output.split('\n')
            in_file_section = False
            file_content_lines = []
            
            for line in lines:
                if file_name in line and ('---' in line or 'File:' in line or file_name + ':' in line):
                    in_file_section = True
                    file_content_lines = []
                    continue
                elif in_file_section and line.startswith('---'):
                    break
                elif in_file_section:
                    file_content_lines.append(line)
            
            if file_content_lines:
                new_content = '\n'.join(file_content_lines).strip()
                full_path = exercise_dir / file_path
                if full_path.exists() and new_content:
                    full_path.write_text(new_content)
                    print(f"Updated {file_path} (via text parsing)")

def run_unit_tests(exercise_dir, test_files):
    """Run unit tests for the exercise"""
    timeout = 60 * 3
    
    # Map of file extensions to test commands
    TEST_COMMANDS = {
        ".py": ["python", "-m", "pytest", "-v"],
        ".rs": ["cargo", "test", "--", "--include-ignored"],
        ".go": ["go", "test", "-v", "./..."],
        ".js": ["npm", "test"],
        ".cpp": ["cmake", "-B", "build", "&&", "cd", "build", "&&", "make", "&&", "ctest"],
        ".java": ["./gradlew", "test"],
    }
    
    # Get unique file extensions from test files
    extensions = {Path(f).suffix for f in test_files}
    
    # Find matching test command
    command = None
    for ext in extensions:
        if ext in TEST_COMMANDS:
            command = TEST_COMMANDS[ext]
            break
    
    if not command:
        return f"No test command found for files with extensions: {extensions}", False
    
    print(f"Running tests: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=exercise_dir,
            encoding="utf-8",
            errors="replace",
        )
        
        success = result.returncode == 0
        output = result.stdout
        
        # Clean up output (remove timing info and absolute paths)
        output = re.sub(r"\bin \d+\.\d+s\b", "", output)
        output = output.replace(str(exercise_dir), exercise_dir.name)
        
        return output, success
        
    except subprocess.TimeoutExpired:
        return "Tests timed out!", False

def run_single_exercise(exercise_dir, original_dir, model="sonnet", tries=2):
    """Run benchmark on a single exercise"""
    print(f"\n{'='*60}")
    print(f"Testing: {exercise_dir.name}")
    print(f"Language: {exercise_dir.parts[-4]}")  # e.g., "python"
    
    # Load configuration
    try:
        config = load_exercise_config(exercise_dir)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {"success": False, "error": str(e)}
    
    # Get instructions
    instructions = get_instructions(exercise_dir)
    
    # Restore original solution files
    for file_path in config["solution_files"]:
        src = exercise_dir / file_path
        original_file = original_dir / exercise_dir.relative_to(original_dir.parent) / file_path
        
        if original_file.exists() and src.parent.exists():
            shutil.copy(original_file, src)
    
    # Track results
    test_outcomes = []
    total_duration = 0
    
    for attempt in range(tries):
        print(f"\nAttempt {attempt + 1}/{tries}")
        
        # Run Claude Code
        claude_result = run_claude_code(
            instructions, 
            config["solution_files"], 
            exercise_dir,
            model
        )
        
        total_duration += claude_result["duration"]
        
        if not claude_result["success"]:
            print(f"Claude Code failed!")
            print(f"Return code: {claude_result.get('returncode', 'unknown')}")
            print(f"STDERR: {claude_result['stderr']}")
            print(f"STDOUT: {claude_result['stdout'][:500]}...")  # First 500 chars
            test_outcomes.append(False)
            continue
        
        # Run unit tests
        test_output, test_success = run_unit_tests(exercise_dir, config["test_files"])
        test_outcomes.append(test_success)
        
        if test_success:
            print("✅ Tests passed!")
            break
        else:
            print("❌ Tests failed:")
            print(test_output[-500:])  # Show last 500 chars
    
    # Save results
    results = {
        "exercise": exercise_dir.name,
        "language": exercise_dir.parts[-4],
        "model": model,
        "test_outcomes": test_outcomes,
        "duration": total_duration,
        "final_success": test_outcomes[-1] if test_outcomes else False,
        "attempts": len(test_outcomes)
    }
    
    results_file = exercise_dir / ".claude.results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def summarize_results(test_dir):
    """Summarize benchmark results"""
    results_files = list(test_dir.glob("**/exercises/practice/*/.claude.results.json"))
    
    if not results_files:
        print("No results found!")
        return
    
    all_results = []
    for results_file in results_files:
        with open(results_file) as f:
            all_results.append(json.load(f))
    
    # Calculate statistics
    total_exercises = len(all_results)
    successful = sum(1 for r in all_results if r["final_success"])
    success_rate = successful / total_exercises if total_exercises > 0 else 0
    
    # Group by language
    by_language = {}
    for result in all_results:
        lang = result["language"]
        if lang not in by_language:
            by_language[lang] = {"total": 0, "passed": 0}
        by_language[lang]["total"] += 1
        if result["final_success"]:
            by_language[lang]["passed"] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total exercises: {total_exercises}")
    print(f"Successful: {successful}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average duration: {sum(r['duration'] for r in all_results) / total_exercises:.1f}s")
    
    print(f"\nBy Language:")
    for lang, stats in sorted(by_language.items()):
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {lang}: {stats['passed']}/{stats['total']} ({rate:.1%})")
    
    return {
        "total_exercises": total_exercises,
        "successful": successful,
        "success_rate": success_rate,
        "by_language": by_language
    }

def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Code Polyglot Benchmark")
    parser.add_argument("--languages", "-l", help="Comma-separated list of languages to test")
    parser.add_argument("--model", "-m", default="sonnet", help="Claude model to use")
    parser.add_argument("--num-tests", "-n", type=int, default=-1, help="Number of tests to run (-1 for all)")
    parser.add_argument("--tries", "-r", type=int, default=2, help="Number of attempts per exercise")
    parser.add_argument("--name", default="claude-polyglot", help="Benchmark run name")
    
    args = parser.parse_args()
    
    # Setup directories
    original_dir = Path(EXERCISES_DIR)
    if not original_dir.exists():
        print(f"Error: {EXERCISES_DIR} not found!")
        sys.exit(1)
    
    test_dir = create_benchmark_dir(args.name)
    
    # Get exercises
    exercise_dirs = get_exercise_dirs(original_dir, args.languages)
    if not exercise_dirs:
        print("No exercises found!")
        sys.exit(1)
    
    print(f"Found {len(exercise_dirs)} exercises")
    
    # Limit number of tests if requested
    if args.num_tests > 0:
        exercise_dirs = exercise_dirs[:args.num_tests]
        print(f"Running {len(exercise_dirs)} exercises")
    
    # Setup test directory
    setup_test_directory(original_dir, test_dir, exercise_dirs)
    
    # Run benchmark
    print(f"Starting Claude Code benchmark...")
    print(f"Model: {args.model}")
    print(f"Test directory: {test_dir}")
    
    start_time = time.time()
    
    for exercise_dir in exercise_dirs:
        test_exercise_dir = test_dir / exercise_dir.relative_to(original_dir)
        run_single_exercise(test_exercise_dir, original_dir, args.model, args.tries)
    
    total_time = time.time() - start_time
    
    # Summarize results
    print(f"\nBenchmark completed in {total_time:.1f} seconds")
    summarize_results(test_dir)

if __name__ == "__main__":
    main()