#!/usr/bin/env python3

import subprocess
import os
import sys
from pathlib import Path
import logging
import difflib
import shlex
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import stat
from .security_cli import COLORS, DECORATORS

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add file handler to save logs
try:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"fix_cycle_{datetime.now():%Y%m%d}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Failed to set up file logging: {e}")

class FixCycle:
    def __init__(self, files=None, message=None, max_attempts=3, report_path=None):
        self.max_attempts = max_attempts
        self.original_contents = {}
        
        if report_path:
            self.findings = self.parse_security_report(report_path)
            self.files = list(set(finding['file'] for finding in self.findings))
        else:
            self.files = files
            self.message = message
            self.findings = []

        # Sanitize and validate file paths
        self.files = [str(Path(f).resolve()) for f in self.files]
        for file in self.files:
            if not os.path.exists(file):
                raise ValueError(f"File not found: {file}")
            if not os.access(file, os.R_OK | os.W_OK):
                raise PermissionError(f"Insufficient permissions for file: {file}")
        
        self._store_original_contents()

    def _store_original_contents(self):
        """Store original file contents for later comparison"""
        for file in self.files:
            try:
                with open(file, 'r') as f:
                    self.original_contents[file] = f.readlines()
            except Exception as e:
                logger.error(f"Failed to read original content of {file}: {e}")

    def _generate_diff_summary(self, file_path):
        """Generate a human-readable summary of changes made to a file"""
        try:
            with open(file_path, 'r') as f:
                new_content = f.readlines()

            original = self.original_contents.get(file_path, [])
            diff = list(difflib.unified_diff(original, new_content, lineterm=''))
            
            if not diff:
                return "No changes made to file."

            changes = {
                'added': [],
                'removed': [],
                'modified': []
            }

            for line in diff:
                if line.startswith('+') and not line.startswith('+++'):
                    changes['added'].append(line[1:].strip())
                elif line.startswith('-') and not line.startswith('---'):
                    changes['removed'].append(line[1:].strip())

            # Identify modified lines (lines that were both removed and added)
            for r_line in changes['removed']:
                for a_line in changes['added']:
                    if difflib.SequenceMatcher(None, r_line, a_line).ratio() > 0.5:
                        changes['modified'].append((r_line, a_line))

            summary = []
            if changes['added']:
                summary.append("\nAdded:")
                for line in changes['added']:
                    if not any(line in mod[1] for mod in changes['modified']):
                        if 'import' in line:
                            summary.append(f"  - New import: {line}")
                        elif 'class' in line:
                            summary.append(f"  - New class: {line.split('class ')[1].split('(')[0]}")
                        elif 'def' in line:
                            summary.append(f"  - New function: {line.split('def ')[1].split('(')[0]}")
                        else:
                            summary.append(f"  - {line[:100]}...")

            if changes['modified']:
                summary.append("\nModified:")
                for old, new in changes['modified']:
                    if 'def' in old and 'def' in new:
                        summary.append(f"  - Updated function: {old.split('def ')[1].split('(')[0]}")
                    elif len(old) < 100:
                        summary.append(f"  - Changed: {old} → {new}")
                    else:
                        summary.append(f"  - Modified long line containing: {old[:50]}...")

            if changes['removed']:
                summary.append("\nRemoved:")
                for line in changes['removed']:
                    if not any(line in mod[0] for mod in changes['modified']):
                        if len(line) < 100:
                            summary.append(f"  - {line}")
                        else:
                            summary.append(f"  - {line[:50]}...")

            return "\n".join(summary)

        except Exception as e:
            return f"Failed to generate diff summary: {e}"

    VALID_FINDING_TYPES = {
        'command_injection', 'xxe', 'insecure_deserialization', 
        'xss', 'weak_crypto', 'sql_injection', 'path_traversal'
    }
    
    def parse_security_report(self, report_path: str, min_severity: str = None) -> List[Dict]:
        """Parse a security report markdown file and extract findings
        
        Args:
            report_path: Path to the markdown report file
            min_severity: Minimum severity to include ('low', 'medium', 'high')
            
        Returns:
            List of finding dictionaries
            
        Raises:
            ValueError: If report format is invalid or required fields are missing
        """
        findings = []
        current_file = None
        current_finding = {}
        severity_levels = {'low': 0, 'medium': 1, 'high': 2}
        min_severity_level = severity_levels.get(min_severity, 0) if min_severity else 0
        
        try:
            with open(report_path, 'r') as f:
                lines = f.readlines()
                
            # Validate report format
            if not any('# Security Review Report' in line for line in lines):
                raise ValueError("Invalid report format - missing header")
                
            for line in lines:
                line = line.strip()
                
                # Look for file paths
                if line.startswith('###'):
                    if current_finding:
                        self._validate_and_add_finding(current_finding, findings, min_severity_level, severity_levels)
                    current_file = line.replace('### ', '').strip()
                    current_finding = {'file': current_file}
                # Look for findings details
                elif line.startswith('- Type:'):
                    finding_type = line.replace('- Type:', '').strip()
                    if finding_type not in self.VALID_FINDING_TYPES:
                        logger.warning(f"Unknown finding type: {finding_type}")
                    current_finding['type'] = finding_type
                elif line.startswith('- Severity:'):
                    severity = line.replace('- Severity:', '').strip()
                    if severity not in severity_levels:
                        raise ValueError(f"Invalid severity level: {severity}")
                    current_finding['severity'] = severity
                elif line.startswith('- Details:'):
                    current_finding['details'] = line.replace('- Details:', '').strip()
                    
            # Add final finding if exists
            if current_finding:
                self._validate_and_add_finding(current_finding, findings, min_severity_level, severity_levels)
            
            # Remove duplicates and sort by severity
            unique_findings = []
            seen = set()
            for finding in findings:
                key = (finding['file'], finding['type'], finding['severity'])
                if key not in seen:
                    seen.add(key)
                    unique_findings.append(finding)
            
            return sorted(unique_findings,
                        key=lambda x: severity_levels.get(x.get('severity'), 0),
                        reverse=True)
        except Exception as e:
            logger.error(f"Failed to parse security report {report_path}: {e}")
            return []

    def _validate_and_add_finding(self, finding: Dict, findings: List, min_severity_level: int, severity_levels: Dict):
        """Validate finding and add to list if it meets criteria"""
        required_fields = {'file', 'type', 'severity', 'details'}
        if not all(field in finding for field in required_fields):
            missing = required_fields - finding.keys()
            raise ValueError(f"Finding missing required fields: {missing}")
            
        finding_severity = finding.get('severity', 'low')
        if severity_levels.get(finding_severity, 0) >= min_severity_level:
            findings.append(finding.copy())

    def _generate_fix_message(self, findings: List[Dict]) -> str:
        """Generate a fix message from findings"""
        message = "Please fix the following security issues:\n\n"
        for finding in findings:
            message += f"- {finding['type']} ({finding['severity']}): {finding['details']}\n"
        return message

    def run_fix_cycle(self, min_severity: str = None):
        """Run fix cycle using aider with direct message passing
        
        Args:
            min_severity: Minimum severity level to process ('low', 'medium', 'high')
        """
        if self.findings:
            # Group findings by file
            file_findings = {}
            for finding in self.findings:
                if finding['file'] not in file_findings:
                    file_findings[finding['file']] = []
                file_findings[finding['file']].append(finding)
            
            # Process each file's findings
            overall_success = True
            for file, findings in file_findings.items():
                message = self._generate_fix_message(findings)
                success = self._run_single_fix(file, message)
                if not success:
                    overall_success = False
            return overall_success
        else:
            return self._run_single_fix(self.files, self.message)

    def _run_single_fix(self, files: Union[str, List[str]], message: str) -> bool:
        """Run a single fix cycle for given files and message"""
        if isinstance(files, str):
            files = [files]
            
        attempts = 0
        while attempts < self.max_attempts:
            logger.info(f"\nAttempt {attempts + 1}/{self.max_attempts}")
            
            try:
                print(f"\n{DECORATORS['box_top']}")
                print(f"{DECORATORS['box_line']} {COLORS['neon_purple']}ATTEMPT {attempts + 1}/{self.max_attempts}{COLORS['reset']}")
                print(f"{DECORATORS['box_bottom']}\n")
                
                # Apply fixes using aider with direct message passing
                logger.info("Applying fixes with aider")
                
                print(f"\n{DECORATORS['box_top']}")
                print(f"{DECORATORS['box_line']} {COLORS['neon_blue']}FILES TO PROCESS:{COLORS['reset']}")
                for file in files:
                    print(f"{DECORATORS['box_line']} {COLORS['neon_green']}• {file}{COLORS['reset']}")
                print(f"{DECORATORS['box_bottom']}\n")
                
                logger.info(f"Message to aider: {message}")
                
                # Construct command with proper escaping
                cmd = ["aider", "--yes-always"]
                cmd.extend(files)
                cmd.extend(["--message", message])
                
                logger.info(f"Executing command: {' '.join(cmd)}")
                
                # Use a more secure subprocess configuration
                # Check if aider is installed
                try:
                    subprocess.run(["aider", "--version"], 
                                 capture_output=True, 
                                 check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.error("aider is not installed. Please install it with: pip install aider-chat")
                    return False

                # Run aider with real-time output display
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    env=os.environ.copy(),
                    cwd=os.getcwd()
                )

                # Display output in real-time
                while True:
                    output = process.stdout.readline()
                    if output:
                        print(output.rstrip())
                        logger.info(output.rstrip())
                    error = process.stderr.readline()
                    if error:
                        print(f"{COLORS['neon_red']}{error.rstrip()}{COLORS['reset']}")
                        logger.error(error.rstrip())
                    
                    # Check if process has finished
                    if output == '' and error == '' and process.poll() is not None:
                        break

                result = process.wait()
                
                if result == 0:
                    logger.info("Aider completed successfully")
                    
                    # Generate and log summary of changes
                    logger.info("\nSummary of changes made:")
                    for file in self.files:
                        logger.info(f"\nChanges in {file}:")
                        summary = self._generate_diff_summary(file)
                        logger.info(summary)
                    
                    self._update_changelog()
                    return True
                
                logger.error(f"Aider failed with return code {result.returncode}")
                if result.stdout:
                    logger.error(f"Output: {result.stdout}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                logger.error("Try running aider manually to debug the issue")
                
            except subprocess.TimeoutExpired:
                logger.error("Fix cycle step timed out")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error during fix cycle: {e}")
                if e.stdout:
                    logger.error(f"Stdout: {e.stdout}")
                if e.stderr:
                    logger.error(f"Stderr: {e.stderr}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            
            attempts += 1
            
        logger.error(f"Fix cycle failed after {attempts} attempts")
        return False

    def _update_changelog(self):
        """Update CHANGELOG.md with fix details"""
        files_str = ", ".join(self.files)
        
        # Generate summary of changes for changelog
        changelog_details = []
        for file in self.files:
            summary = self._generate_diff_summary(file)
            if summary and summary != "No changes made to file.":
                changelog_details.append(f"\nChanges in {file}:{summary}")
        
        changelog_entry = f"""
## Security Fix
- Applied security fixes to: {files_str}
- Changes made based on provided instructions
{"".join(changelog_details)}
"""
        changelog_path = Path("CHANGELOG.md").resolve()
        if not changelog_path.parent.samefile(Path.cwd()):
            raise ValueError("CHANGELOG.md must be in current directory")
            
        try:
            # Create with secure permissions if new
            if not changelog_path.exists():
                changelog_path.touch(mode=0o644)
            
            # Verify file permissions
            current_mode = os.stat(changelog_path).st_mode
            if current_mode & (stat.S_IWOTH | stat.S_IWGRP):
                logger.warning("Insecure CHANGELOG.md permissions detected")
                os.chmod(changelog_path, 0o644)
            
            with open(changelog_path, "a") as f:
                # Sanitize entry before writing
                safe_entry = changelog_entry.encode('utf-8', errors='replace').decode('utf-8')
                f.write(safe_entry)
        except Exception as e:
            logger.error(f"Failed to update changelog: {e}")

def _get_files_from_path(path: str, extensions: tuple = ('.py', '.js', '.ts', '.jsx', '.tsx')) -> List[str]:
    """Recursively get all files with specified extensions from a path"""
    path = Path(path).resolve()
    if path.is_file():
        return [str(path)] if path.suffix in extensions else []
    
    files = []
    try:
        for item in path.rglob('*'):
            if item.is_file() and item.suffix in extensions:
                files.append(str(item))
    except Exception as e:
        logger.error(f"Error scanning directory {path}: {e}")
    
    return files

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run fix cycle with direct message passing to aider')
    parser.add_argument('--path', nargs='+', help='Files or directories to fix')
    parser.add_argument('--message', help='Message to pass directly to aider')
    parser.add_argument('--report', help='Path to security report markdown file')
    parser.add_argument('--min-severity', choices=['low', 'medium', 'high'],
                      help='Minimum severity level to process')
    parser.add_argument('--max-attempts', type=int, default=3, help='Maximum fix attempts')
    parser.add_argument('--extensions', nargs='+', default=['.py', '.js', '.ts', '.jsx', '.tsx'],
                      help='File extensions to process (default: .py .js .ts .jsx .tsx)')
    
    args = parser.parse_args()
    
    if args.report:
        fixer = FixCycle(
            report_path=args.report,
            max_attempts=args.max_attempts
        )
        success = fixer.run_fix_cycle(min_severity=args.min_severity)
    else:
        # Default values if not provided
        if not args.path:
            args.path = ['src/agentic_security/fix_cycle.py']  # Default to self
        if not args.message:
            args.message = "Review this code for security issues and propose fixes"
        
        # Collect all files from provided paths
        all_files = []
        for path in args.path:
            files = _get_files_from_path(path, tuple(args.extensions))
            if files:
                all_files.extend(files)
            else:
                logger.warning(f"No matching files found in {path}")
        
        if not all_files:
            logger.error("No files found to process")
            return 1
        
        logger.info(f"Found {len(all_files)} files to process:")
        for file in all_files:
            logger.info(f"  - {file}")
        
        fixer = FixCycle(
            files=all_files,
            message=args.message,
            max_attempts=args.max_attempts
        )
    
    success = fixer.run_fix_cycle()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
