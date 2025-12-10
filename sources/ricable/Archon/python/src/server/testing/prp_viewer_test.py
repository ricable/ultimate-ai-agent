#!/usr/bin/env python3
"""
PRP Viewer Test Tool

This script tests the rendering consistency between the Milkdown editor view
and the PRPViewer (beautiful view) in the Archon UI.

Usage:
    python prp_viewer_test.py --project-id <PROJECT_UUID> [--output-dir <DIR>]

Requirements:
    - Archon UI server running on port 3737
    - Database connection configured via environment variables
    - Playwright installed (via crawl4ai dependency)
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from playwright.async_api import Browser, Page, async_playwright
from supabase import Client, create_client

# Load environment variables
# When in Docker, load from the mounted .env file
if os.path.exists("/.dockerenv") and os.path.exists("/app/.env"):
    load_dotenv("/app/.env")
else:
    load_dotenv()


class PRPViewerTester:
    """Tests PRP Viewer rendering consistency"""

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
        # When running inside Docker, use host.docker.internal
        if os.path.exists("/.dockerenv"):
            ui_port = os.getenv("ARCHON_UI_PORT", "3737")
            self.base_url = f"http://host.docker.internal:{ui_port}"
        else:
            # When running on host, use localhost
            ui_port = os.getenv("ARCHON_UI_PORT", "3737")
            self.base_url = f"http://localhost:{ui_port}"

        # Results storage
        self.results = {
            "project_id": project_id,
            "test_date": datetime.now().isoformat(),
            "documents": [],
            "summary": {"total_documents": 0, "documents_with_issues": 0, "common_issues": []},
        }

    async def fetch_project_data(self) -> dict[str, Any]:
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
                    })

            # Check if project has prd field
            if project.get("prd"):
                documents.append({
                    "id": "prd_main",
                    "title": project.get("prd", {}).get("title", "Main PRD"),
                    "type": "prd",
                    "content": project["prd"],
                    "source": "project.prd",
                })

            return {"project": project, "documents": documents}

        except Exception as e:
            print(f"Error fetching project data: {e}")
            raise

    async def capture_view_content(self, page: Page, view_type: str) -> dict[str, Any]:
        """Capture content from a specific view"""
        try:
            # Wait for view to load
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)  # Additional wait for React rendering

            if view_type == "markdown":
                # Capture Milkdown editor content
                selector = ".milkdown-editor"
                await page.wait_for_selector(selector, timeout=10000)

                # Get raw markdown content
                markdown_content = await page.evaluate("""
                    () => {
                        const editor = document.querySelector('.milkdown-editor');
                        if (!editor) return null;

                        // Try to get content from various possible sources
                        const prosemirror = editor.querySelector('.ProseMirror');
                        if (prosemirror) {
                            return {
                                text: prosemirror.innerText,
                                html: prosemirror.innerHTML,
                                sections: Array.from(prosemirror.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                                    level: h.tagName,
                                    text: h.innerText
                                }))
                            };
                        }

                        return {
                            text: editor.innerText,
                            html: editor.innerHTML,
                            sections: []
                        };
                    }
                """)

                # Take screenshot
                screenshot_path = self.output_dir / f"{view_type}_{datetime.now().timestamp()}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)

                return {
                    "type": view_type,
                    "content": markdown_content,
                    "screenshot": str(screenshot_path),
                }

            elif view_type == "beautiful":
                # Capture PRPViewer content
                selector = ".prp-viewer"
                await page.wait_for_selector(selector, timeout=10000)

                # Get rendered content
                viewer_content = await page.evaluate("""
                    () => {
                        const viewer = document.querySelector('.prp-viewer');
                        if (!viewer) return null;

                        return {
                            text: viewer.innerText,
                            html: viewer.innerHTML,
                            sections: Array.from(viewer.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                                level: h.tagName,
                                text: h.innerText,
                                parent: h.parentElement?.className || ''
                            })),
                            images: Array.from(viewer.querySelectorAll('img')).map(img => ({
                                src: img.src,
                                alt: img.alt,
                                displayed: img.naturalWidth > 0
                            })),
                            jsonArtifacts: Array.from(viewer.querySelectorAll('pre')).map(pre => ({
                                content: pre.innerText,
                                isJson: (() => {
                                    try {
                                        JSON.parse(pre.innerText);
                                        return true;
                                    } catch {
                                        return false;
                                    }
                                })()
                            }))
                        };
                    }
                """)

                # Take screenshot
                screenshot_path = self.output_dir / f"{view_type}_{datetime.now().timestamp()}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)

                return {
                    "type": view_type,
                    "content": viewer_content,
                    "screenshot": str(screenshot_path),
                }

        except Exception as e:
            print(f"Error capturing {view_type} view: {e}")
            return {"type": view_type, "error": str(e), "content": None}

    async def compare_views(
        self, doc: dict[str, Any], markdown_view: dict[str, Any], beautiful_view: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compare the two views and identify issues"""
        issues = []

        # Check if both views loaded successfully
        if not markdown_view.get("content") or not beautiful_view.get("content"):
            issues.append({
                "type": "render_failure",
                "description": "One or both views failed to render",
                "markdown_loaded": bool(markdown_view.get("content")),
                "beautiful_loaded": bool(beautiful_view.get("content")),
            })
            return issues

        markdown_content = markdown_view["content"]
        beautiful_content = beautiful_view["content"]

        # Compare sections
        markdown_sections = {s["text"].lower() for s in markdown_content.get("sections", [])}
        beautiful_sections = {s["text"].lower() for s in beautiful_content.get("sections", [])}

        # Find missing sections
        missing_in_beautiful = markdown_sections - beautiful_sections
        missing_in_markdown = beautiful_sections - markdown_sections

        for section in missing_in_beautiful:
            issues.append({
                "type": "missing_section",
                "section": section,
                "visible_in": ["markdown"],
                "missing_from": ["beautiful_view"],
            })

        for section in missing_in_markdown:
            issues.append({
                "type": "missing_section",
                "section": section,
                "visible_in": ["beautiful_view"],
                "missing_from": ["markdown"],
            })

        # Check for image placeholder issues
        if beautiful_content.get("images"):
            for img in beautiful_content["images"]:
                if "placeholder-image-" in img["src"] or not img["displayed"]:
                    issues.append({
                        "type": "image_placeholder",
                        "src": img["src"],
                        "alt": img["alt"],
                        "displayed": img["displayed"],
                    })

        # Check for JSON artifacts (raw JSON visible in the view)
        if beautiful_content.get("jsonArtifacts"):
            for artifact in beautiful_content["jsonArtifacts"]:
                if artifact["isJson"]:
                    issues.append({
                        "type": "json_artifact",
                        "description": "Raw JSON visible instead of formatted content",
                        "preview": artifact["content"][:100] + "..."
                        if len(artifact["content"]) > 100
                        else artifact["content"],
                    })

        # Check for significant content length differences
        markdown_length = len(markdown_content.get("text", ""))
        beautiful_length = len(beautiful_content.get("text", ""))

        if markdown_length > 0 and beautiful_length > 0:
            length_ratio = beautiful_length / markdown_length
            if length_ratio < 0.5 or length_ratio > 2.0:
                issues.append({
                    "type": "content_length_mismatch",
                    "markdown_length": markdown_length,
                    "beautiful_length": beautiful_length,
                    "ratio": length_ratio,
                })

        return issues

    async def test_document(self, browser: Browser, doc: dict[str, Any]) -> dict[str, Any]:
        """Test a single document's rendering"""
        result = {
            "doc_id": doc["id"],
            "title": doc["title"],
            "type": doc["type"],
            "source": doc["source"],
            "issues": [],
        }

        try:
            # Create a new page for testing
            page = await browser.new_page()

            # Navigate to the project's docs tab
            url = f"{self.base_url}/projects/{self.project_id}"
            print(f"Navigating to: {url}")

            # Vite dev server might block direct navigation, so use browser context
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                print(f"Initial navigation failed: {e}")
                # Try without waiting for full load
                await page.goto(url, wait_until="commit", timeout=30000)

            # Click on the Docs tab first
            try:
                # Look for the Docs tab button and click it
                await page.wait_for_selector('button:has-text("Docs")', timeout=5000)
                await page.click('button:has-text("Docs")')
                await asyncio.sleep(1)  # Wait for tab to switch
            except:
                print("Could not find Docs tab button, it might already be selected")

            # Wait for any sign of the page being loaded
            try:
                # First wait for React app to be ready
                await page.wait_for_selector("#root", timeout=10000)

                # Then wait for either docs content or project content
                await page.wait_for_selector(
                    'h2:has-text("Project Docs"), .prp-viewer, .milkdown-editor', timeout=15000
                )
                print("Page loaded successfully")
            except Exception as e:
                print(f"Warning: Page might not have loaded fully: {e}")
                # Take a screenshot for debugging
                await page.screenshot(path=f"/app/test_results/debug_{doc['id']}.png")

            # Select the document (if multiple docs exist)
            # Look for document cards in the horizontal scroll area
            try:
                # Wait for document cards to be visible
                await page.wait_for_selector(".flex.gap-4 .cursor-pointer", timeout=5000)

                # Try to find and click on the document by title
                doc_cards = await page.query_selector_all(".flex.gap-4 .cursor-pointer")
                for card in doc_cards:
                    card_text = await card.inner_text()
                    if doc["title"] in card_text:
                        await card.click()
                        print(f"Selected document: {doc['title']}")
                        await asyncio.sleep(1)  # Wait for selection
                        break
            except Exception as e:
                print(f"Could not select document: {e}")
                # Document might already be selected or is the only one

            # Test markdown view
            await page.click('button:has-text("Markdown"), [data-view="markdown"]')
            markdown_view = await self.capture_view_content(page, "markdown")

            # Test beautiful view
            await page.click(
                'button:has-text("Beautiful"), button:has-text("View"), [data-view="beautiful"]'
            )
            beautiful_view = await self.capture_view_content(page, "beautiful")

            # Compare views
            issues = await self.compare_views(doc, markdown_view, beautiful_view)
            result["issues"] = issues

            # Store view data for debugging
            result["views"] = {
                "markdown": {
                    "screenshot": markdown_view.get("screenshot"),
                    "sections_found": len(markdown_view.get("content", {}).get("sections", []))
                    if markdown_view.get("content")
                    else 0,
                },
                "beautiful": {
                    "screenshot": beautiful_view.get("screenshot"),
                    "sections_found": len(beautiful_view.get("content", {}).get("sections", []))
                    if beautiful_view.get("content")
                    else 0,
                },
            }

            await page.close()

        except Exception as e:
            result["issues"].append({"type": "test_error", "error": str(e)})

        return result

    async def run_tests(self):
        """Run all tests"""
        print(f"Starting PRP Viewer tests for project {self.project_id}")

        # Fetch project data
        print("Fetching project data...")
        project_data = await self.fetch_project_data()
        documents = project_data["documents"]

        if not documents:
            print("No documents found in project")
            return

        print(f"Found {len(documents)} documents to test")
        self.results["summary"]["total_documents"] = len(documents)

        # Launch browser
        async with async_playwright() as p:
            print("Launching browser...")
            # Always use headless mode in Docker
            headless = os.path.exists("/.dockerenv")
            browser = await p.chromium.launch(headless=headless)

            try:
                # Test each document
                for i, doc in enumerate(documents):
                    print(
                        f"\nTesting document {i + 1}/{len(documents)}: {doc['title']} ({doc['type']})"
                    )

                    result = await self.test_document(browser, doc)
                    self.results["documents"].append(result)

                    if result["issues"]:
                        self.results["summary"]["documents_with_issues"] += 1

                    # Small delay between documents
                    await asyncio.sleep(2)

            finally:
                await browser.close()

        # Analyze common issues
        self.analyze_common_issues()

        # Save results
        self.save_results()

        print(f"\nTest completed. Results saved to {self.output_dir}")
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
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ViewInconsistencies_{self.project_id}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)

        # Also save a summary report
        summary_file = self.output_dir / f"Summary_{self.project_id}_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("PRP Viewer Test Summary\n")
            f.write("======================\n\n")
            f.write(f"Project ID: {self.project_id}\n")
            f.write(f"Test Date: {self.results['test_date']}\n")
            f.write(f"Total Documents: {self.results['summary']['total_documents']}\n")
            f.write(
                f"Documents with Issues: {self.results['summary']['documents_with_issues']}\n\n"
            )

            f.write("Common Issues:\n")
            for issue_type, count in self.results["summary"].get("issue_breakdown", {}).items():
                f.write(f"  - {issue_type}: {count} occurrences\n")

            f.write("\nDetailed Issues by Document:\n")
            f.write("---------------------------\n")
            for doc in self.results["documents"]:
                if doc["issues"]:
                    f.write(f"\n{doc['title']} ({doc['type']}):\n")
                    for issue in doc["issues"]:
                        f.write(f"  - {issue['type']}: {issue.get('description', issue)}\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test PRP Viewer rendering consistency")
    parser.add_argument("--project-id", required=True, help="UUID of the project to test")
    parser.add_argument(
        "--output-dir", default="./test_results", help="Directory to save test results"
    )

    args = parser.parse_args()

    # Check if UI server is running

    # Determine UI URL based on environment
    if os.path.exists("/.dockerenv"):
        ui_port = os.getenv("ARCHON_UI_PORT", "3737")
        ui_url = f"http://host.docker.internal:{ui_port}"
    else:
        ui_port = os.getenv("ARCHON_UI_PORT", "3737")
        ui_url = f"http://localhost:{ui_port}"

    # Skip UI connectivity check for now - Vite dev server may block direct requests
    print(f"Using UI server at {ui_url}")
    print("Note: Skipping connectivity check as Vite dev server may block direct HTTP requests")

    # Run tests
    tester = PRPViewerTester(args.project_id, args.output_dir)
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
