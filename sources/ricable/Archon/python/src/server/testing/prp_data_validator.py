#!/usr/bin/env python3
"""
PRP Data Validator

This script validates PRP document structure and content directly from the database
without needing to render the UI. It identifies potential rendering issues by analyzing
the document data structure.

Usage:
    docker exec Archon-Server python /app/src/server/testing/prp_data_validator.py --project-id <PROJECT_UUID>
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables
if os.path.exists("/.dockerenv") and os.path.exists("/app/.env"):
    load_dotenv("/app/.env")
else:
    load_dotenv()


class PRPDataValidator:
    """Validates PRP document data structure"""

    def __init__(self, project_id: str, output_dir: str = "./test_results"):
        self.project_id = project_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials in environment")

        self.supabase: Client = create_client(supabase_url, supabase_key)

        # Results storage
        self.results = {
            "project_id": project_id,
            "validation_date": datetime.now().isoformat(),
            "documents": [],
            "summary": {"total_documents": 0, "documents_with_issues": 0, "common_issues": []},
        }

    def fetch_project_data(self) -> dict[str, Any]:
        """Fetch project and its documents from database"""
        try:
            # Fetch project
            project_response = (
                self.supabase.table("archon_projects")
                .select("*")
                .eq("id", self.project_id)
                .execute()
            )
            if not project_response.data:
                raise ValueError(f"Project {self.project_id} not found")

            project = project_response.data[0]

            # Fetch all document types from project
            documents = []

            # Check if project has docs array
            if project.get("docs"):
                for doc in project["docs"]:
                    documents.append({
                        "id": doc.get("id", f"doc_{len(documents)}"),
                        "title": doc.get("title", "Untitled"),
                        "type": doc.get("document_type", doc.get("type", "unknown")),
                        "content": doc.get("content", doc),
                        "source": "project.docs",
                        "raw_data": doc,
                    })

            # Check if project has prd field
            if project.get("prd"):
                documents.append({
                    "id": "prd_main",
                    "title": project.get("prd", {}).get("title", "Main PRD"),
                    "type": "prd",
                    "content": project["prd"],
                    "source": "project.prd",
                    "raw_data": project["prd"],
                })

            return {"project": project, "documents": documents}

        except Exception as e:
            print(f"Error fetching project data: {e}")
            raise

    def validate_document_structure(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate a single document's structure and identify issues"""
        issues = []
        content = doc.get("content", doc.get("raw_data", {}))

        # Check if content is a string or object
        if isinstance(content, str):
            # Raw markdown string
            issues.append({
                "type": "raw_markdown_string",
                "description": "Document stored as raw markdown string instead of structured object",
                "impact": "May not render properly in PRPViewer",
                "recommendation": "Convert to structured PRP object format",
            })

            # Check for image placeholders
            if "[Image #" in content:
                import re

                placeholders = re.findall(r"\[Image #(\d+)\]", content)
                issues.append({
                    "type": "image_placeholders",
                    "count": len(placeholders),
                    "placeholders": placeholders,
                    "description": f"Found {len(placeholders)} image placeholder(s)",
                    "impact": "Images will show as text placeholders",
                })

        elif isinstance(content, dict):
            # Structured object

            # Check for nested content field
            if "content" in content and isinstance(content["content"], (str, dict)):
                issues.append({
                    "type": "nested_content_field",
                    "description": "Document has nested 'content' field",
                    "impact": "May cause double-wrapping in rendering",
                })

            # Check for mixed content types
            string_fields = []
            object_fields = []
            array_fields = []

            for key, value in content.items():
                if isinstance(value, str):
                    string_fields.append(key)
                    # Check for JSON strings
                    if value.strip().startswith("{") or value.strip().startswith("["):
                        try:
                            json.loads(value)
                            issues.append({
                                "type": "json_string_field",
                                "field": key,
                                "description": f"Field '{key}' contains JSON as string",
                                "impact": "Will render as raw JSON text instead of formatted content",
                            })
                        except:
                            pass

                    # Check for image placeholders in strings
                    if "[Image #" in value:
                        import re

                        placeholders = re.findall(r"\[Image #(\d+)\]", value)
                        if placeholders:
                            issues.append({
                                "type": "image_placeholders_in_field",
                                "field": key,
                                "count": len(placeholders),
                                "description": f"Field '{key}' contains {len(placeholders)} image placeholder(s)",
                            })

                elif isinstance(value, dict):
                    object_fields.append(key)
                elif isinstance(value, list):
                    array_fields.append(key)

            # Check for missing expected PRP sections
            expected_sections = [
                "goal",
                "why",
                "what",
                "context",
                "user_personas",
                "user_flows",
                "success_metrics",
                "implementation_plan",
                "technical_implementation",
                "validation_gates",
            ]

            missing_sections = [s for s in expected_sections if s not in content]
            if missing_sections:
                issues.append({
                    "type": "missing_sections",
                    "sections": missing_sections,
                    "description": f"Missing {len(missing_sections)} expected PRP sections",
                    "impact": "Incomplete PRP structure",
                })

            # Check for sections that might not render
            metadata_fields = ["title", "version", "author", "date", "status", "document_type"]
            renderable_sections = [k for k in content.keys() if k not in metadata_fields]

            if len(renderable_sections) == 0:
                issues.append({
                    "type": "no_renderable_content",
                    "description": "Document has no renderable sections (only metadata)",
                    "impact": "Nothing will display in the viewer",
                })

        else:
            issues.append({
                "type": "invalid_content_type",
                "content_type": type(content).__name__,
                "description": f"Content is of type {type(content).__name__}, expected string or dict",
                "impact": "Cannot render this content type",
            })

        return issues

    def analyze_milkdown_compatibility(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze if document will convert properly to markdown for Milkdown editor"""
        issues = []
        content = doc.get("content", doc.get("raw_data", {}))

        if isinstance(content, dict):
            # Check convertPRPToMarkdown compatibility
            # Based on the function in MilkdownEditor.tsx

            # Check for complex nested structures
            for key, value in content.items():
                if isinstance(value, dict) and any(
                    isinstance(v, (dict, list)) for v in value.values()
                ):
                    issues.append({
                        "type": "complex_nesting",
                        "field": key,
                        "description": f"Field '{key}' has complex nested structure",
                        "impact": "May not convert properly to markdown",
                    })

                # Check for non-standard field names
                if not key.replace("_", "").isalnum():
                    issues.append({
                        "type": "non_standard_field_name",
                        "field": key,
                        "description": f"Field '{key}' has non-standard characters",
                        "impact": "May not display properly as section title",
                    })

        return issues

    def run_validation(self):
        """Run all validations"""
        print(f"Starting PRP Data Validation for project {self.project_id}")

        # Fetch project data
        print("Fetching project data...")
        project_data = self.fetch_project_data()
        documents = project_data["documents"]

        if not documents:
            print("No documents found in project")
            return

        print(f"Found {len(documents)} documents to validate")
        self.results["summary"]["total_documents"] = len(documents)

        # Validate each document
        for i, doc in enumerate(documents):
            print(f"\nValidating document {i + 1}/{len(documents)}: {doc['title']} ({doc['type']})")

            # Structure validation
            structure_issues = self.validate_document_structure(doc)

            # Milkdown compatibility
            milkdown_issues = self.analyze_milkdown_compatibility(doc)

            all_issues = structure_issues + milkdown_issues

            result = {
                "doc_id": doc["id"],
                "title": doc["title"],
                "type": doc["type"],
                "source": doc["source"],
                "issues": all_issues,
                "issue_count": len(all_issues),
            }

            self.results["documents"].append(result)

            if all_issues:
                self.results["summary"]["documents_with_issues"] += 1
                print(f"  Found {len(all_issues)} issues")
            else:
                print("  ✓ No issues found")

        # Analyze common issues
        self.analyze_common_issues()

        # Save results
        self.save_results()

        print(f"\nValidation completed. Results saved to {self.output_dir}")
        print(
            f"Summary: {self.results['summary']['documents_with_issues']} out of {self.results['summary']['total_documents']} documents have issues"
        )

    def analyze_common_issues(self):
        """Analyze and summarize common issues across all documents"""
        issue_counts = {}

        for doc in self.results["documents"]:
            for issue in doc["issues"]:
                issue_type = issue["type"]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Sort by frequency
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        self.results["summary"]["common_issues"] = [issue[0] for issue in common_issues[:5]]
        self.results["summary"]["issue_breakdown"] = dict(common_issues)

    def save_results(self):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_filename = f"DataValidation_{self.project_id}_{timestamp}.json"
        json_filepath = self.output_dir / json_filename

        with open(json_filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save human-readable report
        txt_filename = f"DataValidationSummary_{self.project_id}_{timestamp}.txt"
        txt_filepath = self.output_dir / txt_filename

        with open(txt_filepath, "w") as f:
            f.write("PRP Data Validation Summary\n")
            f.write("===========================\n\n")
            f.write(f"Project ID: {self.project_id}\n")
            f.write(f"Validation Date: {self.results['validation_date']}\n")
            f.write(f"Total Documents: {self.results['summary']['total_documents']}\n")
            f.write(
                f"Documents with Issues: {self.results['summary']['documents_with_issues']}\n\n"
            )

            f.write("Common Issues:\n")
            for issue_type, count in self.results["summary"].get("issue_breakdown", {}).items():
                f.write(f"  - {issue_type}: {count} occurrences\n")

            f.write("\nDetailed Issues by Document:\n")
            f.write("----------------------------\n")
            for doc in self.results["documents"]:
                if doc["issues"]:
                    f.write(f"\n{doc['title']} ({doc['type']}):\n")
                    for issue in doc["issues"]:
                        f.write(f"  - [{issue['type']}] {issue['description']}\n")
                        if "impact" in issue:
                            f.write(f"    Impact: {issue['impact']}\n")
                        if "recommendation" in issue:
                            f.write(f"    Fix: {issue['recommendation']}\n")

        print("\nResults saved:")
        print(f"  - JSON report: {json_filepath}")
        print(f"  - Summary: {txt_filepath}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate PRP document data structure")
    parser.add_argument("--project-id", required=True, help="UUID of the project to validate")
    parser.add_argument(
        "--output-dir", default="./test_results", help="Directory to save validation results"
    )

    args = parser.parse_args()

    # Run validation
    validator = PRPDataValidator(args.project_id, args.output_dir)
    validator.run_validation()


if __name__ == "__main__":
    main()
