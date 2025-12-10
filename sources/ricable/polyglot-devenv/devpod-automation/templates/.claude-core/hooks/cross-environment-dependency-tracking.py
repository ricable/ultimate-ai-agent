#!/usr/bin/env python3
"""
Cross-Environment Dependency Tracking Hook
Monitors package files across all environments and validates compatibility.

Features:
- Monitors package.json, Cargo.toml, pyproject.toml, go.mod, devbox.json changes
- Cross-environment dependency compatibility checking
- Version conflict detection and resolution suggestions
- Security vulnerability scanning for new dependencies
- Integration with existing validation infrastructure
- Dependency graph analysis and optimization recommendations
"""

import json
import sys
import subprocess
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
import difflib
from collections import defaultdict

class CrossEnvironmentDependencyTracker:
    """Track and analyze dependencies across all polyglot environments."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_db_file = project_root / ".claude" / "dependency_tracking.json"
        self.vulnerability_cache_file = project_root / ".claude" / "vulnerability_cache.json"
        
        # Package file patterns for each environment
        self.package_files = {
            "python": ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "poetry.lock"],
            "typescript": ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "nushell": ["devbox.json"],  # Nushell uses devbox for packages
            "devbox": ["devbox.json", "devbox.lock"]  # Cross-environment devbox
        }
        
        # Security scanning tools by environment
        self.security_scanners = {
            "python": ["safety", "pip-audit"],
            "typescript": ["npm audit", "yarn audit"],
            "rust": ["cargo audit"],
            "go": ["govulncheck", "nancy"],
            "devbox": ["devbox info"]
        }
        
        # Common vulnerability patterns
        self.vulnerability_patterns = {
            "high_severity": [
                r"HIGH.*SEVERITY", r"CRITICAL.*VULNERABILITY", r"REMOTE.*CODE.*EXECUTION",
                r"SQL.*INJECTION", r"CROSS.*SITE.*SCRIPTING", r"BUFFER.*OVERFLOW"
            ],
            "dependency_confusion": [
                r"DEPENDENCY.*CONFUSION", r"TYPOSQUATTING", r"MALICIOUS.*PACKAGE"
            ],
            "outdated_critical": [
                r"OUTDATED.*CRITICAL", r"EOL.*VERSION", r"UNSUPPORTED.*VERSION"
            ]
        }
        
        # Load existing dependency state
        self.dependency_state = self._load_dependency_state()
        
        # Cross-environment compatibility rules
        self.compatibility_rules = {
            "node_versions": {
                "16": ["typescript", "javascript"],
                "18": ["typescript", "javascript"],
                "20": ["typescript", "javascript"]
            },
            "python_versions": {
                "3.11": ["python"],
                "3.12": ["python"]
            },
            "conflicting_packages": {
                "python": {
                    ("requests", "httpx"): "Consider standardizing on one HTTP client",
                    ("pytest", "unittest"): "Multiple testing frameworks detected",
                    ("black", "autopep8"): "Multiple formatters may conflict"
                },
                "typescript": {
                    ("webpack", "vite"): "Multiple bundlers detected",
                    ("jest", "vitest"): "Multiple testing frameworks detected",
                    ("eslint", "tslint"): "TSLint is deprecated, use ESLint"
                }
            }
        }
    
    def _load_dependency_state(self) -> Dict:
        """Load existing dependency tracking state."""
        if self.dependency_db_file.exists():
            try:
                with open(self.dependency_db_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "version": "1.0",
            "environments": {},
            "last_scan": {},
            "vulnerabilities": {},
            "compatibility_issues": {},
            "dependency_graph": {}
        }
    
    def _save_dependency_state(self):
        """Save dependency tracking state."""
        try:
            self.dependency_state["last_updated"] = datetime.now().isoformat()
            with open(self.dependency_db_file, 'w') as f:
                json.dump(self.dependency_state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save dependency state: {e}")
    
    def detect_environment_from_package_file(self, file_path: str) -> Optional[str]:
        """Detect environment from package file path."""
        path = Path(file_path)
        file_name = path.name
        
        # Check file name patterns
        for env, patterns in self.package_files.items():
            if file_name in patterns:
                return env
        
        # Check directory context
        path_str = str(path)
        for env in ["python", "typescript", "rust", "go", "nushell"]:
            if f"{env}-env" in path_str:
                return env
        
        return None
    
    def extract_dependencies_from_file(self, file_path: str, environment: str) -> Dict[str, Any]:
        """Extract dependency information from package file."""
        try:
            path = Path(file_path)
            dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
            
            if not path.exists():
                return dependencies
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if environment == "python" and file_path.endswith('.toml'):
                # Parse pyproject.toml
                dependencies = self._parse_pyproject_toml(content)
            elif environment == "typescript" and file_path.endswith('.json'):
                # Parse package.json
                dependencies = self._parse_package_json(content)
            elif environment == "rust" and file_path.endswith('.toml'):
                # Parse Cargo.toml
                dependencies = self._parse_cargo_toml(content)
            elif environment == "go" and file_path.endswith('.mod'):
                # Parse go.mod
                dependencies = self._parse_go_mod(content)
            elif file_path.endswith('devbox.json'):
                # Parse devbox.json
                dependencies = self._parse_devbox_json(content)
            
            dependencies["file_path"] = file_path
            dependencies["environment"] = environment
            dependencies["last_modified"] = datetime.now().isoformat()
            
            return dependencies
            
        except Exception as e:
            return {"error": str(e), "file_path": file_path, "environment": environment}
    
    def _parse_pyproject_toml(self, content: str) -> Dict[str, Any]:
        """Parse pyproject.toml for Python dependencies."""
        dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
        
        try:
            # Simple regex-based parsing (for production, use tomli/tomllib)
            dep_section = re.search(r'\[project\].*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if dep_section:
                deps_text = dep_section.group(1)
                for line in deps_text.split('\n'):
                    line = line.strip().strip(',').strip('"').strip("'")
                    if line and not line.startswith('#'):
                        # Extract package name and version
                        match = re.match(r'([a-zA-Z0-9_-]+)([>=<~!]*[0-9.]*)', line)
                        if match:
                            pkg_name, version = match.groups()
                            dependencies["dependencies"][pkg_name] = version or "latest"
            
            # Extract dev dependencies
            dev_dep_section = re.search(r'\[project\.optional-dependencies\].*?dev\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if dev_dep_section:
                dev_deps_text = dev_dep_section.group(1)
                for line in dev_deps_text.split('\n'):
                    line = line.strip().strip(',').strip('"').strip("'")
                    if line and not line.startswith('#'):
                        match = re.match(r'([a-zA-Z0-9_-]+)([>=<~!]*[0-9.]*)', line)
                        if match:
                            pkg_name, version = match.groups()
                            dependencies["dev_dependencies"][pkg_name] = version or "latest"
                            
        except Exception as e:
            dependencies["parse_error"] = str(e)
        
        return dependencies
    
    def _parse_package_json(self, content: str) -> Dict[str, Any]:
        """Parse package.json for TypeScript/JavaScript dependencies."""
        dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
        
        try:
            data = json.loads(content)
            dependencies["dependencies"] = data.get("dependencies", {})
            dependencies["dev_dependencies"] = data.get("devDependencies", {})
            dependencies["metadata"] = {
                "name": data.get("name"),
                "version": data.get("version"),
                "node_version": data.get("engines", {}).get("node")
            }
        except json.JSONDecodeError as e:
            dependencies["parse_error"] = str(e)
        
        return dependencies
    
    def _parse_cargo_toml(self, content: str) -> Dict[str, Any]:
        """Parse Cargo.toml for Rust dependencies."""
        dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
        
        try:
            # Extract [dependencies] section
            dep_section = re.search(r'\[dependencies\](.*?)(?=\[|\Z)', content, re.DOTALL)
            if dep_section:
                deps_text = dep_section.group(1)
                for line in deps_text.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            pkg_name = parts[0].strip()
                            version = parts[1].strip().strip('"').strip("'")
                            dependencies["dependencies"][pkg_name] = version
            
            # Extract [dev-dependencies] section
            dev_dep_section = re.search(r'\[dev-dependencies\](.*?)(?=\[|\Z)', content, re.DOTALL)
            if dev_dep_section:
                dev_deps_text = dev_dep_section.group(1)
                for line in dev_deps_text.split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            pkg_name = parts[0].strip()
                            version = parts[1].strip().strip('"').strip("'")
                            dependencies["dev_dependencies"][pkg_name] = version
                            
        except Exception as e:
            dependencies["parse_error"] = str(e)
        
        return dependencies
    
    def _parse_go_mod(self, content: str) -> Dict[str, Any]:
        """Parse go.mod for Go dependencies."""
        dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
        
        try:
            lines = content.split('\n')
            in_require = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('module '):
                    dependencies["metadata"]["module"] = line.split()[1]
                elif line.startswith('go '):
                    dependencies["metadata"]["go_version"] = line.split()[1]
                elif line == 'require (':
                    in_require = True
                elif line == ')' and in_require:
                    in_require = False
                elif in_require or line.startswith('require '):
                    # Extract dependency
                    parts = line.replace('require ', '').strip().split()
                    if len(parts) >= 2:
                        pkg_name = parts[0]
                        version = parts[1]
                        dependencies["dependencies"][pkg_name] = version
                        
        except Exception as e:
            dependencies["parse_error"] = str(e)
        
        return dependencies
    
    def _parse_devbox_json(self, content: str) -> Dict[str, Any]:
        """Parse devbox.json for system packages."""
        dependencies = {"dependencies": {}, "dev_dependencies": {}, "metadata": {}}
        
        try:
            data = json.loads(content)
            packages = data.get("packages", [])
            
            for package in packages:
                if isinstance(package, str):
                    # Simple package name
                    dependencies["dependencies"][package] = "latest"
                elif isinstance(package, dict):
                    # Package with version or options
                    name = package.get("name", "unknown")
                    version = package.get("version", "latest")
                    dependencies["dependencies"][name] = version
            
            dependencies["metadata"] = {
                "shell": data.get("shell"),
                "scripts": list(data.get("scripts", {}).keys())
            }
            
        except json.JSONDecodeError as e:
            dependencies["parse_error"] = str(e)
        
        return dependencies
    
    def scan_for_vulnerabilities(self, environment: str, dependencies: Dict[str, str]) -> List[Dict[str, Any]]:
        """Scan dependencies for security vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Use environment-specific security scanners
            scanners = self.security_scanners.get(environment, [])
            
            for scanner in scanners:
                if self._is_scanner_available(scanner):
                    vuln_results = self._run_vulnerability_scan(scanner, environment)
                    vulnerabilities.extend(vuln_results)
            
            # Pattern-based vulnerability detection
            for dep_name, version in dependencies.items():
                pattern_vulns = self._check_vulnerability_patterns(dep_name, version)
                vulnerabilities.extend(pattern_vulns)
            
        except Exception as e:
            vulnerabilities.append({
                "type": "scan_error",
                "message": f"Vulnerability scan failed: {e}",
                "severity": "low"
            })
        
        return vulnerabilities
    
    def _is_scanner_available(self, scanner: str) -> bool:
        """Check if security scanner is available."""
        scanner_cmd = scanner.split()[0]
        try:
            result = subprocess.run(
                ["which", scanner_cmd],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_vulnerability_scan(self, scanner: str, environment: str) -> List[Dict[str, Any]]:
        """Run vulnerability scan using specified scanner."""
        vulnerabilities = []
        
        try:
            # Change to environment directory
            env_dir = self.project_root / f"dev-env/{environment}"
            if not env_dir.exists():
                return vulnerabilities
            
            result = subprocess.run(
                scanner.split(),
                capture_output=True,
                text=True,
                timeout=60,
                cwd=env_dir
            )
            
            # Parse scanner output
            if result.returncode != 0 and result.stderr:
                # Errors often indicate vulnerabilities found
                vulns = self._parse_scanner_output(result.stderr, scanner)
                vulnerabilities.extend(vulns)
            
        except subprocess.TimeoutExpired:
            vulnerabilities.append({
                "type": "timeout",
                "scanner": scanner,
                "message": "Security scan timeout",
                "severity": "medium"
            })
        except Exception as e:
            vulnerabilities.append({
                "type": "error",
                "scanner": scanner,
                "message": str(e),
                "severity": "low"
            })
        
        return vulnerabilities
    
    def _parse_scanner_output(self, output: str, scanner: str) -> List[Dict[str, Any]]:
        """Parse vulnerability scanner output."""
        vulnerabilities = []
        
        # Look for high-severity patterns
        for severity, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, output, re.IGNORECASE):
                    vulnerabilities.append({
                        "type": "vulnerability",
                        "scanner": scanner,
                        "severity": severity,
                        "pattern": pattern,
                        "message": f"Potential {severity} issue detected"
                    })
        
        return vulnerabilities
    
    def _check_vulnerability_patterns(self, dep_name: str, version: str) -> List[Dict[str, Any]]:
        """Check for known vulnerability patterns in dependencies."""
        vulnerabilities = []
        
        # Check for suspicious package names (typosquatting)
        suspicious_patterns = [
            r".*[0O].*",  # Numbers that look like letters
            r".*[1Il].*",  # Confusing characters
            r"^[a-z]+-[a-z]+$"  # Hyphenated names (common in typosquatting)
        ]
        
        for pattern in suspicious_patterns:
            if re.match(pattern, dep_name):
                vulnerabilities.append({
                    "type": "suspicious_name",
                    "package": dep_name,
                    "version": version,
                    "severity": "medium",
                    "message": f"Suspicious package name pattern: {dep_name}"
                })
        
        # Check for version patterns that might indicate issues
        if version in ["latest", "*", "^0.0.0"]:
            vulnerabilities.append({
                "type": "unsafe_version",
                "package": dep_name,
                "version": version,
                "severity": "low",
                "message": f"Unsafe version specification: {version}"
            })
        
        return vulnerabilities
    
    def analyze_cross_environment_compatibility(self) -> List[Dict[str, Any]]:
        """Analyze compatibility issues across environments."""
        issues = []
        
        try:
            # Check for conflicting package patterns
            all_deps = {}
            for env, state in self.dependency_state.get("environments", {}).items():
                deps = state.get("dependencies", {})
                all_deps[env] = deps
            
            # Look for conflicting packages within environments
            for env, deps in all_deps.items():
                conflicts = self.compatibility_rules.get("conflicting_packages", {}).get(env, {})
                for conflict_pair, message in conflicts.items():
                    if all(pkg in deps for pkg in conflict_pair):
                        issues.append({
                            "type": "package_conflict",
                            "environment": env,
                            "packages": list(conflict_pair),
                            "message": message,
                            "severity": "medium"
                        })
            
            # Check for version mismatches across environments
            package_versions = defaultdict(dict)
            for env, deps in all_deps.items():
                for pkg, version in deps.items():
                    package_versions[pkg][env] = version
            
            for pkg, env_versions in package_versions.items():
                if len(env_versions) > 1:
                    versions = set(env_versions.values())
                    if len(versions) > 1:
                        issues.append({
                            "type": "version_mismatch",
                            "package": pkg,
                            "environments": env_versions,
                            "severity": "low",
                            "message": f"Different versions of {pkg} across environments"
                        })
            
        except Exception as e:
            issues.append({
                "type": "analysis_error",
                "message": str(e),
                "severity": "low"
            })
        
        return issues
    
    def generate_dependency_recommendations(self, environment: str, dependencies: Dict[str, str]) -> List[str]:
        """Generate recommendations for dependency management."""
        recommendations = []
        
        try:
            # Environment-specific recommendations
            if environment == "python":
                if len(dependencies) > 20:
                    recommendations.append("Consider using dependency groups for large Python projects")
                if "requests" in dependencies and "httpx" in dependencies:
                    recommendations.append("Consider standardizing on httpx for async HTTP requests")
            
            elif environment == "typescript":
                if len(dependencies) > 30:
                    recommendations.append("Consider code splitting to reduce bundle size")
                if "@types/" not in str(dependencies):
                    recommendations.append("Add TypeScript type definitions for better development experience")
            
            elif environment == "rust":
                if len(dependencies) > 15:
                    recommendations.append("Consider using features to reduce compilation time")
                if "tokio" in dependencies:
                    recommendations.append("Ensure async runtime is properly configured")
            
            elif environment == "go":
                if len(dependencies) > 10:
                    recommendations.append("Go encourages minimal dependencies - review if all are necessary")
            
            # General recommendations
            outdated_count = sum(1 for v in dependencies.values() if v in ["*", "latest"])
            if outdated_count > 3:
                recommendations.append(f"Pin {outdated_count} wildcard versions for reproducible builds")
            
            if len(dependencies) > 50:
                recommendations.append("Large dependency count - consider dependency audit and cleanup")
            
        except Exception as e:
            recommendations.append(f"Recommendation analysis failed: {e}")
        
        return recommendations
    
    def process_package_file_change(self, file_path: str) -> Dict[str, Any]:
        """Process package file change and perform comprehensive analysis."""
        result = {
            "processed": False,
            "environment": None,
            "changes_detected": [],
            "vulnerabilities": [],
            "compatibility_issues": [],
            "recommendations": []
        }
        
        try:
            # Detect environment
            environment = self.detect_environment_from_package_file(file_path)
            if not environment:
                return result
            
            result.update({
                "processed": True,
                "environment": environment,
                "file_path": file_path
            })
            
            # Extract current dependencies
            current_deps = self.extract_dependencies_from_file(file_path, environment)
            
            # Compare with previous state
            prev_deps = self.dependency_state.get("environments", {}).get(environment, {})
            changes = self._detect_dependency_changes(prev_deps, current_deps)
            result["changes_detected"] = changes
            
            # Update state
            self.dependency_state.setdefault("environments", {})[environment] = current_deps
            
            # Scan for vulnerabilities
            deps_dict = current_deps.get("dependencies", {})
            deps_dict.update(current_deps.get("dev_dependencies", {}))
            vulnerabilities = self.scan_for_vulnerabilities(environment, deps_dict)
            result["vulnerabilities"] = vulnerabilities
            
            # Check cross-environment compatibility
            compatibility_issues = self.analyze_cross_environment_compatibility()
            result["compatibility_issues"] = compatibility_issues
            
            # Generate recommendations
            recommendations = self.generate_dependency_recommendations(environment, deps_dict)
            result["recommendations"] = recommendations
            
            # Save updated state
            self._save_dependency_state()
            
            # Print analysis results
            if changes:
                print(f"📦 Dependency Changes in {environment}:")
                for change in changes[:3]:  # Show first 3 changes
                    print(f"   {change['type']}: {change['package']}")
            
            if vulnerabilities:
                high_severity = [v for v in vulnerabilities if v.get("severity") == "high"]
                if high_severity:
                    print(f"🚨 High severity vulnerabilities found: {len(high_severity)}")
            
            if compatibility_issues:
                print(f"⚠️ Cross-environment issues detected: {len(compatibility_issues)}")
            
            if recommendations:
                print(f"💡 Optimization recommendations:")
                print(f"   {recommendations[0]}")
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _detect_dependency_changes(self, prev_deps: Dict, current_deps: Dict) -> List[Dict[str, Any]]:
        """Detect changes between dependency states."""
        changes = []
        
        try:
            prev_all = {}
            prev_all.update(prev_deps.get("dependencies", {}))
            prev_all.update(prev_deps.get("dev_dependencies", {}))
            
            current_all = {}
            current_all.update(current_deps.get("dependencies", {}))
            current_all.update(current_deps.get("dev_dependencies", {}))
            
            # Detect additions
            for pkg in current_all:
                if pkg not in prev_all:
                    changes.append({
                        "type": "added",
                        "package": pkg,
                        "version": current_all[pkg]
                    })
            
            # Detect removals
            for pkg in prev_all:
                if pkg not in current_all:
                    changes.append({
                        "type": "removed",
                        "package": pkg,
                        "version": prev_all[pkg]
                    })
            
            # Detect version changes
            for pkg in prev_all:
                if pkg in current_all and prev_all[pkg] != current_all[pkg]:
                    changes.append({
                        "type": "updated",
                        "package": pkg,
                        "old_version": prev_all[pkg],
                        "new_version": current_all[pkg]
                    })
            
        except Exception as e:
            changes.append({"type": "error", "message": str(e)})
        
        return changes
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for dependency tracking."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            if tool_name in ["Edit", "MultiEdit", "Write"]:
                file_path = tool_input.get("file_path", "")
                
                if file_path:
                    # Check if it's a package file
                    environment = self.detect_environment_from_package_file(file_path)
                    
                    if environment:
                        result = self.process_package_file_change(file_path)
                        
                        if result.get("processed"):
                            changes = result.get("changes_detected", [])
                            vulnerabilities = result.get("vulnerabilities", [])
                            
                            if changes:
                                print(f"📦 Cross-Environment Dependency Tracking: {len(changes)} changes in {environment}")
                            
                            if vulnerabilities:
                                high_severity = [v for v in vulnerabilities if v.get("severity") in ["high", "critical"]]
                                if high_severity:
                                    print(f"🚨 Security Alert: {len(high_severity)} high-severity issues detected")
            
            print(f"📦 Cross-Environment Dependency Tracking: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Dependency tracking processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root - adapt for DevPod container
        if Path('/workspace').exists():
            project_root = Path('/workspace')
        else:
            project_root = Path.cwd()
        
        # Initialize dependency tracker
        tracker = CrossEnvironmentDependencyTracker(project_root)
        
        # Process the event
        tracker.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cross-Environment Dependency Tracking Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()