"""
Project Module for Archon MCP Server - PRP-Driven Development Platform

🛡️ AUTOMATIC VERSION CONTROL & DATA PROTECTION:
This module provides comprehensive project management with BUILT-IN VERSION CONTROL
that prevents documentation erasure and enables complete audit trails.

🔄 Version Control Features:
- AUTOMATIC SNAPSHOTS: Every document update creates immutable version backup
- COMPLETE ROLLBACK: Any version can be restored without data loss
- AUDIT COMPLIANCE: Full change history with timestamps and creator attribution
- DISASTER RECOVERY: All operations preserve historical data permanently

📋 PRP (Product Requirement Prompt) Integration:
- Structured JSON format for proper PRPViewer compatibility
- Complete PRP templates with all required sections
- Validation gates and implementation blueprints
- Task generation from PRP implementation plans

🏗️ Consolidated MCP Tools:
- manage_project: Project lifecycle with automatic version control
- manage_task: PRP-driven task management with status workflows
- manage_document: Document management with version snapshots
- manage_versions: Complete version history and rollback capabilities
- get_project_features: Feature query operations

⚠️ CRITICAL SAFETY: All operations preserve data through automatic versioning.
No content can be permanently lost - use manage_versions for recovery.
"""

import json
import logging
from typing import Any
from urllib.parse import urljoin

# Import HTTP client and service discovery
import httpx

from mcp.server.fastmcp import Context, FastMCP

# Import service discovery for HTTP calls
from src.server.config.service_discovery import get_api_url

logger = logging.getLogger(__name__)


def register_project_tools(mcp: FastMCP):
    """Register consolidated project and task management tools with the MCP server."""

    @mcp.tool()
    async def manage_project(
        ctx: Context,
        action: str,
        project_id: str = None,
        title: str = None,
        prd: dict[str, Any] = None,
        github_repo: str = None,
    ) -> str:
        """
        Unified tool for Archon project lifecycle management with integrated PRP support.

        🚀 PRP-DRIVEN PROJECT ARCHITECTURE:
        Archon projects are designed around the PRP (Product Requirement Prompt) methodology:
        - Projects contain structured documents (PRPs, specs, designs)
        - Each project has automatic version control for all content
        - Tasks are generated from PRP implementation blueprints
        - Progress is tracked through task status workflows
        - All changes are auditable with complete history

        ⚠️ DATA SAFETY FEATURES:
        - ALL project data is versioned automatically (docs, tasks, features)
        - DELETE operations preserve version history for audit compliance
        - Documents and tasks remain recoverable through version management
        - Use manage_versions tool to restore accidentally deleted content

        Args:
            action: Project operation - "create" | "list" | "get" | "delete"
                   - create: Initialize new PRP-driven project with version control
                   - list: Retrieve all projects with metadata summary
                   - get: Fetch complete project details including documents and tasks
                   - delete: Archive project (preserves all version history)

            project_id: UUID of the project (required for get/delete operations)
                       Obtained from list operation or project creation response

            title: Human-readable project title (required for create)
                  Should be descriptive and specific (e.g., "OAuth2 Authentication System", "E-commerce API v3.0")

            prd: Product Requirements Document as structured JSON (optional for create)
                Format: {
                    "product_vision": "Clear vision statement",
                    "target_users": ["User persona 1", "User persona 2"],
                    "key_features": ["Feature 1", "Feature 2"],
                    "success_metrics": ["Metric 1", "Metric 2"],
                    "constraints": ["Technical constraints", "Business constraints"]
                }

            github_repo: GitHub repository URL (optional for create)
                        Format: "https://github.com/username/repository"
                        Used for linking project to source code repository

        Returns:
            JSON string with project operation results:
            - success: Boolean indicating operation success
            - project: Complete project object (for create/get actions)
            - projects: Array of projects (for list action)
            - message: Human-readable status message
            - error: Error description (if success=false)

        🏗️ COMPREHENSIVE EXAMPLES:

        Create New PRP-Driven Project:
            manage_project(
                action="create",
                title="OAuth2 Authentication System",
                prd={
                    "product_vision": "Secure, user-friendly authentication system supporting multiple OAuth2 providers",
                    "target_users": ["Web application users", "Mobile app users", "API developers"],
                    "key_features": [
                        "Google OAuth2 integration",
                        "GitHub OAuth2 integration",
                        "Automatic token refresh",
                        "User profile synchronization",
                        "Security compliance (PKCE, CSRF protection)"
                    ],
                    "success_metrics": [
                        "< 3 clicks for user authentication",
                        "99.9% authentication success rate",
                        "< 2 second authentication flow completion",
                        "Zero security incidents in production"
                    ],
                    "constraints": [
                        "Must work with existing user database schema",
                        "GDPR compliance required for EU users",
                        "Maximum 2MB additional bundle size"
                    ]
                },
                github_repo="https://github.com/company/auth-service"
            )

        List All Projects:
            manage_project(action="list")
            # Returns: Array of all projects with basic metadata

        Get Project with Full Details:
            manage_project(
                action="get",
                project_id="550e8400-e29b-41d4-a716-446655440000"
            )
            # Returns: Complete project object with documents, tasks, features, version history

        Archive Project (Preserves Version History):
            manage_project(
                action="delete",
                project_id="550e8400-e29b-41d4-a716-446655440000"
            )
            # Note: All project data remains recoverable through version management

        Create Minimal Project:
            manage_project(
                action="create",
                title="Quick Prototype - User Dashboard"
            )
            # Creates project with basic structure, PRD and GitHub repo can be added later
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            if action == "create":
                if not title:
                    return json.dumps({
                        "success": False,
                        "error": "Title is required for create action",
                    })

                # Call Server API to create project
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        urljoin(api_url, "/api/projects"),
                        json={"title": title, "prd": prd, "github_repo": github_repo},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, "project": result})
                    else:
                        error_detail = (
                            response.json().get("detail", {}).get("error", "Unknown error")
                        )
                        return json.dumps({"success": False, "error": error_detail})

            elif action == "list":
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(urljoin(api_url, "/api/projects"))

                    if response.status_code == 200:
                        projects = response.json()
                        return json.dumps({"success": True, "projects": projects})
                    else:
                        return json.dumps({"success": False, "error": "Failed to list projects"})

            elif action == "get":
                if not project_id:
                    return json.dumps({
                        "success": False,
                        "error": "project_id is required for get action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(urljoin(api_url, f"/api/projects/{project_id}"))

                    if response.status_code == 200:
                        project = response.json()
                        return json.dumps({"success": True, "project": project})
                    elif response.status_code == 404:
                        return json.dumps({
                            "success": False,
                            "error": f"Project {project_id} not found",
                        })
                    else:
                        return json.dumps({"success": False, "error": "Failed to get project"})

            elif action == "delete":
                if not project_id:
                    return json.dumps({
                        "success": False,
                        "error": "project_id is required for delete action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.delete(urljoin(api_url, f"/api/projects/{project_id}"))

                    if response.status_code == 200:
                        return json.dumps({
                            "success": True,
                            "message": "Project deleted successfully",
                        })
                    else:
                        return json.dumps({"success": False, "error": "Failed to delete project"})

            else:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action '{action}'. Must be one of: create, list, get, delete",
                })

        except Exception as e:
            logger.error(f"Error in manage_project: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @mcp.tool()
    async def manage_task(
        ctx: Context,
        action: str,
        task_id: str = None,
        project_id: str = None,
        filter_by: str = None,
        filter_value: str = None,
        title: str = None,
        description: str = "",
        assignee: str = "User",
        task_order: int = 0,
        feature: str = None,
        sources: list[dict[str, Any]] = None,
        code_examples: list[dict[str, Any]] = None,
        update_fields: dict[str, Any] = None,
        include_closed: bool = False,
        page: int = 1,
        per_page: int = 50,
    ) -> str:
        """
        Unified tool for task management operations within PRP-driven projects.

        🎯 PRP TASK LIFECYCLE MANAGEMENT:
        Tasks follow the PRP methodology lifecycle: todo → doing → review → done
        - todo: Task is ready to be started, all dependencies resolved
        - doing: Task is actively being worked on (limit: 1 task per agent)
        - review: Task implementation complete, awaiting validation
        - done: Task validated and integrated, no further work needed

        📋 TASK STATUS WORKFLOW:
        1. PRP breaks down into implementation tasks (created as 'todo')
        2. Agent moves task to 'doing' before starting work
        3. Agent completes implementation and moves to 'review'
        4. prp-validator runs validation gates and moves to 'done' or back to 'doing'
        5. Completed tasks feed back into PRP progress tracking

        👥 AGENT ASSIGNMENTS:
        - 'User': Manual tasks requiring human intervention
        - 'Archon': AI-driven development tasks
        - 'AI IDE Agent': Direct code implementation tasks
        - 'prp-executor': PRP implementation coordination
        - 'prp-validator': Quality assurance and testing
        - 'archon-task-manager': Workflow orchestration

        Args:
            action: Task operation - "create" | "list" | "get" | "update" | "delete" | "archive"
                   - create: Generate new task from PRP implementation plan
                   - list: Retrieve tasks with filtering (by status, project, assignee)
                   - get: Fetch complete task details including sources and code examples
                   - update: Modify task properties (primarily status transitions)
                   - delete/archive: Remove completed or obsolete tasks

            task_id: UUID of the task (required for get/update/delete/archive)

            project_id: UUID of the project (required for create, optional for list filtering)

            filter_by: List filtering type - "status" | "project" | "assignee"
            filter_value: Value for the filter (e.g., "todo", "doing", "review", "done")

            title: Task title (required for create) - should be specific and actionable
                  ✅ Good: "Implement OAuth2 Google provider configuration"
                  ✅ Good: "Add unit tests for token refresh mechanism"
                  ❌ Bad: "Work on auth", "Fix OAuth stuff"

            description: Detailed task description (for create) - include context and acceptance criteria

            assignee: Agent responsible for task execution
                     - 'User': Tasks requiring manual review or configuration
                     - 'Archon': General AI implementation tasks
                     - 'AI IDE Agent': Direct code modification tasks
                     - 'prp-executor': PRP coordination and orchestration
                     - 'prp-validator': Testing and quality validation

            task_order: Priority within status (0-100, higher = more priority)
                       Use to sequence dependent tasks within each status

            feature: Feature label for grouping related tasks (e.g., "authentication", "oauth2")

            sources: List of source metadata for task context
                    [{"url": "docs/oauth.md", "type": "documentation", "relevance": "OAuth2 implementation guide"}]

            code_examples: List of relevant code examples for implementation
                          [{"file": "src/auth/base.py", "function": "authenticate_user", "purpose": "Base auth pattern"}]

            update_fields: Dict of fields to update (for update action)
                          Common updates: {"status": "doing"}, {"assignee": "prp-validator"},
                          {"description": "Updated requirements based on testing feedback"}

            include_closed: Include 'done' tasks in list results (default: False)
                           Set to True for progress reporting and audit trails

            page: Page number for pagination (default: 1, for large task lists)
            per_page: Items per page (default: 50, max: 100)

        Returns:
            JSON string with task operation results:
            - success: Boolean indicating operation success
            - task/tasks: Task object(s) with complete metadata
            - pagination: Pagination info for list operations
            - message: Human-readable status message
            - error: Error description (if success=false)

        📚 COMPREHENSIVE EXAMPLES:

        Create PRP Implementation Task:
            manage_task(
                action="create",
                project_id="550e8400-e29b-41d4-a716-446655440000",
                title="Implement OAuth2 Google provider configuration",
                description="Create GoogleOAuthProvider class with proper endpoints, scopes, and client configuration. Must handle authorization URL generation with PKCE security.",
                assignee="AI IDE Agent",
                task_order=10,
                feature="authentication",
                sources=[
                    {"url": "https://developers.google.com/identity/protocols/oauth2", "type": "documentation", "relevance": "Official OAuth2 spec"},
                    {"file": "docs/auth/README.md", "type": "internal_docs", "relevance": "Current auth architecture"}
                ],
                code_examples=[
                    {"file": "src/auth/base.py", "class": "BaseAuthProvider", "purpose": "Provider interface pattern"},
                    {"file": "examples/oauth-flow.py", "function": "generate_auth_url", "purpose": "URL generation example"}
                ]
            )

        Update Task Status (todo → doing):
            manage_task(
                action="update",
                task_id="task-123e4567-e89b-12d3-a456-426614174000",
                update_fields={"status": "doing", "assignee": "prp-executor"}
            )

        List Tasks by Status for Progress Tracking:
            manage_task(
                action="list",
                filter_by="status",
                filter_value="review",
                project_id="550e8400-e29b-41d4-a716-446655440000",
                per_page=25
            )

        List All Tasks for Project Audit:
            manage_task(
                action="list",
                filter_by="project",
                filter_value="550e8400-e29b-41d4-a716-446655440000",
                include_closed=True,
                per_page=100
            )

        Get Task with Full Context:
            manage_task(
                action="get",
                task_id="task-123e4567-e89b-12d3-a456-426614174000"
            )
            # Returns: Complete task object with sources, code_examples, and metadata

        Archive Completed Task:
            manage_task(
                action="archive",
                task_id="task-123e4567-e89b-12d3-a456-426614174000"
            )
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            if action == "create":
                if not project_id:
                    return json.dumps({
                        "success": False,
                        "error": "project_id is required for create action",
                    })
                if not title:
                    return json.dumps({
                        "success": False,
                        "error": "title is required for create action",
                    })

                # Call Server API to create task
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        urljoin(api_url, "/api/tasks"),
                        json={
                            "project_id": project_id,
                            "title": title,
                            "description": description,
                            "assignee": assignee,
                            "task_order": task_order,
                            "feature": feature,
                            "sources": sources,
                            "code_examples": code_examples,
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "task": result.get("task"),
                            "message": result.get("message"),
                        })
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            elif action == "list":
                # Build URL with query parameters based on filter type
                params = {
                    "page": page,
                    "per_page": per_page,
                    "exclude_large_fields": True,  # Always exclude large fields in MCP responses
                }

                # Use different endpoints based on filter type for proper parameter handling
                if filter_by == "project" and filter_value:
                    # Use project-specific endpoint for project filtering
                    url = urljoin(api_url, f"/api/projects/{filter_value}/tasks")
                    params["include_archived"] = False  # For backward compatibility

                    # Only add include_closed logic for project filtering
                    if not include_closed:
                        # This endpoint handles done task filtering differently
                        pass  # Let the endpoint handle it
                elif filter_by == "status" and filter_value:
                    # Use generic tasks endpoint for status filtering
                    url = urljoin(api_url, "/api/tasks")
                    params["status"] = filter_value
                    params["include_closed"] = include_closed
                    # Add project_id if provided
                    if project_id:
                        params["project_id"] = project_id
                else:
                    # Default to generic tasks endpoint
                    url = urljoin(api_url, "/api/tasks")
                    params["include_closed"] = include_closed

                # Make the API call
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()

                    result = response.json()

                    # Handle both direct array and paginated response formats
                    if isinstance(result, list):
                        # Direct array response
                        tasks = result
                        pagination_info = None
                    else:
                        # Paginated response or object with tasks property
                        if "tasks" in result:
                            tasks = result.get("tasks", [])
                            pagination_info = result.get("pagination", {})
                        else:
                            # Direct array in object form
                            tasks = result if isinstance(result, list) else []
                            pagination_info = None

                    return json.dumps({
                        "success": True,
                        "tasks": tasks,
                        "pagination": pagination_info,
                        "total_count": len(tasks)
                        if pagination_info is None
                        else pagination_info.get("total", len(tasks)),
                    })

            elif action == "get":
                if not task_id:
                    return json.dumps({
                        "success": False,
                        "error": "task_id is required for get action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(urljoin(api_url, f"/api/tasks/{task_id}"))

                    if response.status_code == 200:
                        task = response.json()
                        return json.dumps({"success": True, "task": task})
                    elif response.status_code == 404:
                        return json.dumps({"success": False, "error": f"Task {task_id} not found"})
                    else:
                        return json.dumps({"success": False, "error": "Failed to get task"})

            elif action == "update":
                if not task_id:
                    return json.dumps({
                        "success": False,
                        "error": "task_id is required for update action",
                    })
                if not update_fields:
                    return json.dumps({
                        "success": False,
                        "error": "update_fields is required for update action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.put(
                        urljoin(api_url, f"/api/tasks/{task_id}"), json=update_fields
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "task": result.get("task"),
                            "message": result.get("message"),
                        })
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            elif action in ["delete", "archive"]:
                if not task_id:
                    return json.dumps({
                        "success": False,
                        "error": "task_id is required for delete/archive action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.delete(urljoin(api_url, f"/api/tasks/{task_id}"))

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "message": result.get("message"),
                            "subtasks_archived": result.get("subtasks_archived", 0),
                        })
                    else:
                        return json.dumps({"success": False, "error": "Failed to archive task"})

            else:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action '{action}'. Must be one of: create, list, get, update, delete, archive",
                })

        except Exception as e:
            logger.error(f"Error in manage_task: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @mcp.tool()
    async def manage_document(
        ctx: Context,
        action: str,
        project_id: str,
        doc_id: str = None,
        document_type: str = None,
        title: str = None,
        content: dict[str, Any] = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """
        Unified tool for document management within projects with AUTOMATIC VERSION CONTROL.

        🔒 CRITICAL SAFETY FEATURES:
        - AUTOMATIC VERSION SNAPSHOTS: Every update creates immutable backup before changes
        - PREVENTS DOCUMENTATION ERASURE: Complete version history preserved permanently
        - ROLLBACK CAPABILITY: Use manage_versions to restore any previous version
        - AUDIT TRAIL: All changes tracked with timestamps, authors, and change summaries

        📋 PRP (Product Requirement Prompt) FORMAT REQUIREMENTS:
        For PRP documents (document_type="prp"), content MUST be structured JSON compatible
        with PRPViewer component, NOT raw markdown. This ensures proper rendering and validation.

        Required PRP Metadata Fields:
        - title: Clear, descriptive document title
        - version: Semantic version (e.g., "1.0", "2.1", "3.0-beta")
        - author: Agent identifier ("prp-creator", "prp-executor", "prp-validator", "AI IDE Agent")
        - date: ISO date format (YYYY-MM-DD)
        - status: Lifecycle status ("draft", "review", "approved", "deprecated")
        - document_type: Always "prp" for PRP documents

        📊 COMPLETE PRP Structure Template:
        {
            "document_type": "prp",
            "title": "OAuth2 Authentication Implementation",
            "version": "1.0",
            "author": "prp-creator",
            "date": "2025-07-30",
            "status": "draft",

            "goal": "Implement secure OAuth2 authentication with Google and GitHub providers",

            "why": [
                "Enable secure user authentication without password management",
                "Reduce registration friction and improve user conversion rates",
                "Comply with enterprise security requirements for SSO integration"
            ],

            "what": {
                "description": "Complete OAuth2 flow with provider selection, token management, and user profile integration",
                "success_criteria": [
                    "Users can authenticate with Google/GitHub in <3 clicks",
                    "Secure token storage with automatic refresh handling",
                    "Profile data synchronization with local user accounts",
                    "Graceful error handling for failed authentication attempts"
                ],
                "user_stories": [
                    "As a new user, I want to sign up with my Google account to avoid creating another password",
                    "As a developer, I want to use GitHub auth to leverage my existing developer identity"
                ]
            },

            "context": {
                "documentation": [
                    {"source": "https://developers.google.com/identity/protocols/oauth2", "why": "Official OAuth2 implementation guide"},
                    {"source": "docs/auth/README.md", "why": "Current authentication architecture"},
                    {"source": "examples/oauth-flow.py", "why": "Reference implementation pattern"}
                ],
                "existing_code": [
                    {"file": "src/auth/base.py", "purpose": "Base authentication classes and interfaces"},
                    {"file": "src/auth/session.py", "purpose": "Session management and token storage"}
                ],
                "gotchas": [
                    "OAuth2 state parameter MUST be validated to prevent CSRF attacks",
                    "Token refresh must happen before expiration to avoid user session loss",
                    "Provider-specific scopes vary - Google uses 'openid profile email', GitHub uses 'user:email'",
                    "PKCE (Proof Key for Code Exchange) required for mobile/SPA applications"
                ],
                "current_state": "Basic username/password authentication exists. Session management handles JWT tokens. Need to integrate OAuth2 providers alongside existing system.",
                "dependencies": [
                    "requests-oauthlib", "cryptography", "python-jose[cryptography]"
                ],
                "environment_variables": [
                    "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                    "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET",
                    "OAUTH_REDIRECT_URI"
                ]
            },

            "implementation_blueprint": {
                "phase_1_provider_setup": {
                    "description": "Configure OAuth2 providers and basic flow",
                    "tasks": [
                        {
                            "title": "Create OAuth2 provider configurations",
                            "files": ["src/auth/oauth/providers.py"],
                            "details": "Define GoogleOAuthProvider and GitHubOAuthProvider classes with endpoints, scopes, and client configuration"
                        },
                        {
                            "title": "Implement authorization URL generation",
                            "files": ["src/auth/oauth/flow.py"],
                            "details": "Generate secure authorization URLs with state parameter and PKCE for enhanced security"
                        },
                        {
                            "title": "Add OAuth2 routes to FastAPI",
                            "files": ["src/api/auth.py"],
                            "details": "Add /auth/oauth/{provider} and /auth/oauth/{provider}/callback endpoints"
                        }
                    ]
                },
                "phase_2_token_handling": {
                    "description": "Implement secure token exchange and storage",
                    "tasks": [
                        {
                            "title": "Implement authorization code exchange",
                            "files": ["src/auth/oauth/token_handler.py"],
                            "details": "Exchange authorization code for access/refresh tokens with proper error handling"
                        },
                        {
                            "title": "Add token storage to database",
                            "files": ["src/models/oauth_token.py", "migrations/add_oauth_tokens.py"],
                            "details": "Create OAuthToken model with encrypted storage and automatic cleanup of expired tokens"
                        }
                    ]
                },
                "phase_3_user_integration": {
                    "description": "Link OAuth2 accounts with user profiles",
                    "tasks": [
                        {
                            "title": "Fetch and normalize user profiles",
                            "files": ["src/auth/oauth/profile.py"],
                            "details": "Retrieve user profile data from providers and normalize to common User model fields"
                        },
                        {
                            "title": "Implement account linking logic",
                            "files": ["src/auth/oauth/account_linking.py"],
                            "details": "Link OAuth2 accounts to existing users or create new accounts with proper conflict resolution"
                        }
                    ]
                }
            },

            "validation": {
                "level_1_syntax": [
                    "ruff check --fix src/auth/oauth/",
                    "mypy src/auth/oauth/",
                    "black src/auth/oauth/"
                ],
                "level_2_unit_tests": [
                    "pytest tests/auth/test_oauth_providers.py -v",
                    "pytest tests/auth/test_oauth_flow.py -v",
                    "pytest tests/auth/test_token_handler.py -v"
                ],
                "level_3_integration": [
                    "pytest tests/integration/test_oauth_flow_complete.py -v",
                    "curl -X GET http://localhost:8181/auth/oauth/google",
                    "curl -X POST http://localhost:8181/auth/oauth/google/callback -d 'code=test&state=valid_state'"
                ],
                "level_4_end_to_end": [
                    "Start development server: uvicorn main:app --reload",
                    "Navigate to /auth/oauth/google in browser",
                    "Complete OAuth2 flow and verify user profile creation",
                    "Test token refresh mechanism with expired tokens",
                    "Verify secure logout clears OAuth2 tokens"
                ]
            },

            "additional_context": {
                "security_considerations": [
                    "Always validate OAuth2 state parameter to prevent CSRF",
                    "Use HTTPS for all OAuth2 redirects in production",
                    "Implement rate limiting on OAuth2 endpoints",
                    "Store refresh tokens encrypted in database",
                    "Set appropriate token expiration times"
                ],
                "testing_strategies": [
                    "Mock OAuth2 provider responses for unit tests",
                    "Use test OAuth2 applications for integration testing",
                    "Test error scenarios: network failures, invalid codes, expired tokens",
                    "Verify proper cleanup of test data between test runs"
                ],
                "monitoring_and_logging": [
                    "Log OAuth2 authentication attempts with success/failure metrics",
                    "Monitor token refresh rates and failures",
                    "Alert on unusual OAuth2 error patterns",
                    "Track user adoption of OAuth2 vs traditional auth"
                ]
            }
        }

        🔄 Version Control Behavior:
        - AUTOMATIC SNAPSHOTS: Every update creates immutable version before applying changes
        - COMPLETE STATE PRESERVATION: Full document content, metadata, and structure saved
        - CHRONOLOGICAL HISTORY: All versions timestamped with change summaries
        - INSTANT ROLLBACK: Use manage_versions(action="restore") to revert to any previous version
        - AUDIT COMPLIANCE: Permanent record of who changed what and when

        Args:
            action: Operation - "add" | "list" | "get" | "update" | "delete"
            project_id: UUID of the project (always required)
            doc_id: UUID of the document (required for get/update/delete)
            document_type: Type of document (required for add) - use "prp" for PRP documents
            title: Document title (required for add, optional for update)
            content: Document content as structured JSON (for add/update)
                     For PRPs: Use structured JSON format above, NOT markdown
            metadata: Dict with optional fields: tags, status, version, author
                      For PRPs: Include required fields (title, version, author, date, status, document_type)

        Returns:
            JSON string with operation results

        Examples:
            Add PRP: manage_document(action="add", project_id="uuid", document_type="prp",
                                   title="OAuth Implementation", content={PRP_JSON_STRUCTURE})
            Add Document: manage_document(action="add", project_id="uuid", document_type="spec",
                                        title="API Spec", content={"sections": {...}})
            List: manage_document(action="list", project_id="uuid")
            Get: manage_document(action="get", project_id="uuid", doc_id="doc-uuid")
            Update PRP: manage_document(action="update", project_id="uuid", doc_id="doc-uuid",
                                      content={UPDATED_PRP_JSON})
            Delete: manage_document(action="delete", project_id="uuid", doc_id="doc-uuid")
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            if action == "add":
                if not document_type:
                    return json.dumps({
                        "success": False,
                        "error": "document_type is required for add action",
                    })
                if not title:
                    return json.dumps({
                        "success": False,
                        "error": "title is required for add action",
                    })

                # CRITICAL VALIDATION: PRP documents must use structured JSON format
                if document_type == "prp":
                    if not isinstance(content, dict):
                        return json.dumps({
                            "success": False,
                            "error": "PRP documents (document_type='prp') require structured JSON content, not markdown strings. Content must be a dictionary with sections like 'goal', 'why', 'what', 'context', 'implementation_blueprint', 'validation'. See MCP documentation for required PRP structure.",
                        })

                    # Validate required PRP structure fields
                    required_fields = [
                        "goal",
                        "why",
                        "what",
                        "context",
                        "implementation_blueprint",
                        "validation",
                    ]
                    missing_fields = [field for field in required_fields if field not in content]
                    if missing_fields:
                        return json.dumps({
                            "success": False,
                            "error": f"PRP content missing required fields: {missing_fields}. PRP documents must include: goal, why, what, context, implementation_blueprint, validation. See MCP documentation for complete PRP structure template.",
                        })

                    # Ensure document_type is set in content for PRPViewer compatibility
                    if "document_type" not in content:
                        content["document_type"] = "prp"

                # Call Server API to create document
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        urljoin(api_url, f"/api/projects/{project_id}/docs"),
                        json={
                            "document_type": document_type,
                            "title": title,
                            "content": content,
                            "tags": metadata.get("tags") if metadata else None,
                            "author": metadata.get("author") if metadata else None,
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "document": result.get("document"),
                            "message": result.get("message"),
                        })
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            elif action == "list":
                async with httpx.AsyncClient(timeout=timeout) as client:
                    url = urljoin(api_url, f"/api/projects/{project_id}/docs")
                    logger.info(f"Calling document list API: {url}")
                    response = await client.get(url)

                    logger.info(f"Document list API response: {response.status_code}")
                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, **result})
                    else:
                        error_text = response.text
                        logger.error(
                            f"Document list API error: {response.status_code} - {error_text}"
                        )
                        return json.dumps({
                            "success": False,
                            "error": f"HTTP {response.status_code}: {error_text}",
                        })

            elif action == "get":
                if not doc_id:
                    return json.dumps({
                        "success": False,
                        "error": "doc_id is required for get action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(
                        urljoin(api_url, f"/api/projects/{project_id}/docs/{doc_id}")
                    )

                    if response.status_code == 200:
                        document = response.json()
                        return json.dumps({"success": True, "document": document})
                    elif response.status_code == 404:
                        return json.dumps({
                            "success": False,
                            "error": f"Document {doc_id} not found",
                        })
                    else:
                        return json.dumps({"success": False, "error": "Failed to get document"})

            elif action == "update":
                if not doc_id:
                    return json.dumps({
                        "success": False,
                        "error": "doc_id is required for update action",
                    })

                # CRITICAL VALIDATION: PRP documents must use structured JSON format
                if content is not None:
                    # First get the existing document to check its type
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        get_response = await client.get(
                            urljoin(api_url, f"/api/projects/{project_id}/docs/{doc_id}")
                        )
                        if get_response.status_code == 200:
                            existing_doc = get_response.json().get("document", {})
                            existing_type = existing_doc.get(
                                "document_type", existing_doc.get("type")
                            )

                            if existing_type == "prp":
                                if not isinstance(content, dict):
                                    return json.dumps({
                                        "success": False,
                                        "error": "PRP documents (document_type='prp') require structured JSON content, not markdown strings. "
                                        "Content must be a dictionary with required fields: goal, why, what, context, implementation_blueprint, validation. "
                                        "See project_module.py lines 570-756 for the complete PRP structure specification.",
                                    })

                                # Validate required PRP fields
                                required_fields = [
                                    "goal",
                                    "why",
                                    "what",
                                    "context",
                                    "implementation_blueprint",
                                    "validation",
                                ]
                                missing_fields = [
                                    field for field in required_fields if field not in content
                                ]

                                if missing_fields:
                                    return json.dumps({
                                        "success": False,
                                        "error": f"PRP content missing required fields: {', '.join(missing_fields)}. "
                                        f"Required fields: {', '.join(required_fields)}",
                                    })

                                # Ensure document_type is set for PRPViewer compatibility
                                if "document_type" not in content:
                                    content["document_type"] = "prp"

                # Build update fields
                update_fields = {}
                if title is not None:
                    update_fields["title"] = title
                if content is not None:
                    update_fields["content"] = content
                if metadata:
                    if "tags" in metadata:
                        update_fields["tags"] = metadata["tags"]
                    if "author" in metadata:
                        update_fields["author"] = metadata["author"]

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.put(
                        urljoin(api_url, f"/api/projects/{project_id}/docs/{doc_id}"),
                        json=update_fields,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "document": result.get("document"),
                            "message": result.get("message"),
                        })
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            elif action == "delete":
                if not doc_id:
                    return json.dumps({
                        "success": False,
                        "error": "doc_id is required for delete action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.delete(
                        urljoin(api_url, f"/api/projects/{project_id}/docs/{doc_id}")
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, "message": result.get("message")})
                    else:
                        return json.dumps({"success": False, "error": "Failed to delete document"})

            else:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action '{action}'. Must be one of: add, list, get, update, delete",
                })

        except Exception as e:
            logger.error(f"Error in manage_document: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @mcp.tool()
    async def manage_versions(
        ctx: Context,
        action: str,
        project_id: str,
        field_name: str,
        version_number: int = None,
        content: dict[str, Any] = None,
        change_summary: str = None,
        document_id: str = None,
        created_by: str = "system",
    ) -> str:
        """
        Unified tool for IMMUTABLE document version management and complete change history.

        🛡️ AUTOMATIC VERSION PROTECTION:
        - EVERY UPDATE to manage_document triggers automatic version snapshot BEFORE applying changes
        - PREVENTS DATA LOSS: Complete document state preserved before any modification
        - NO MANUAL ACTION REQUIRED: Version control is transparent and automatic
        - ROLLBACK SAFETY NET: Any change can be instantly reverted using restore action

        📈 VERSION MANAGEMENT BEST PRACTICES:

        Change Summary Guidelines (be specific and actionable):
        ✅ GOOD: "Added OAuth2 validation gates and security considerations to implementation blueprint"
        ✅ GOOD: "Fixed task dependencies in phase_2_token_handling, added missing error handling"
        ✅ GOOD: "Updated success criteria to include performance benchmarks and user experience metrics"
        ❌ BAD: "Updated document", "Fixed stuff", "Changes made"

        Created By Identifiers (use consistent agent names):
        - "prp-creator": Initial PRP creation and major structural changes
        - "prp-executor": Implementation progress updates and task completion
        - "prp-validator": Quality assurance, testing validation, and approval workflows
        - "AI IDE Agent": Direct user-driven modifications and quick fixes
        - "archon-task-manager": Automated task lifecycle updates
        - "archon-project-orchestrator": Project-wide coordination changes

        🔍 PRP VERSION TRACKING:
        For PRP documents, versions capture COMPLETE structured content:
        - Full metadata: title, version, author, date, status, document_type
        - Complete goal and business justification sections
        - Entire context: documentation links, gotchas, current state, dependencies
        - Full implementation blueprint with all phases and tasks
        - Complete validation gates at all levels
        - Additional context: security considerations, testing strategies, monitoring

        📚 IMMUTABLE AUDIT TRAIL:
        - PERMANENT RECORD: Versions cannot be modified once created
        - CHRONOLOGICAL EVOLUTION: List action shows document development over time
        - METADATA TRACKING: Each version includes timestamp, creator, change summary
        - RESTORE WITHOUT LOSS: Restoration creates new version while preserving all history
        - COMPLIANCE READY: Complete audit trail for regulatory and process compliance

        🔄 VERSION LIFECYCLE:
        1. Document updated via manage_document → Automatic version snapshot created
        2. Changes applied to current document state
        3. New version number assigned (auto-incremented)
        4. Historical versions remain permanently accessible
        5. Any version can be restored, creating new current version

        🚨 DISASTER RECOVERY:
        - If document corruption occurs: Use list action to find last good version
        - If incorrect changes applied: Use restore action with specific version_number
        - If need to compare versions: Use get action to examine specific historical states
        - If need change audit: Version list shows complete modification history

        Args:
            action: Version control operation - "create" | "list" | "get" | "restore"
                   - "create": Make manual version snapshot (automatic versions created by manage_document)
                   - "list": Show chronological version history with metadata
                   - "get": Retrieve complete content of specific historical version
                   - "restore": Rollback to previous version (creates new version, preserves history)

            project_id: UUID of the project (ALWAYS required for all actions)

            field_name: JSONB field name for version tracking
                       - "docs": Document versions (PRPs, specs, designs, notes)
                       - "features": Feature development snapshots
                       - "data": Project data and configuration snapshots
                       - "prd": Product Requirements Document versions

            version_number: Specific version number (required for get/restore actions)
                          Obtained from list action results (auto-incremented integers)

            content: Complete content to snapshot (required for create action)
                    ⚠️ For PRPs: Must include ALL sections - goal, why, what, context, implementation_blueprint, validation
                    ⚠️ For Features: Complete feature definitions with status and components
                    ⚠️ Use structured JSON, not strings or markdown

            change_summary: Descriptive summary of what changed (for create action)
                          ✅ Be specific: "Added OAuth2 validation section with security checklist"
                          ✅ Include impact: "Updated implementation blueprint to fix dependency ordering"
                          ✅ Reference context: "Milestone checkpoint before major refactoring"
                          ❌ Avoid generic: "Updated document", "Made changes"

            document_id: Specific document UUID within docs array (for create action with docs field)
                        Used to associate version with specific document

            created_by: Agent or user identifier who created this version
                       Standard identifiers: "prp-creator", "prp-executor", "prp-validator",
                       "AI IDE Agent", "archon-task-manager", "archon-project-orchestrator"

        Returns:
            JSON string with version operation results:
            - success: Boolean indicating operation success
            - version: Version object with metadata (for create/get actions)
            - versions: Array of version history (for list action)
            - message: Human-readable status message
            - content: Full versioned content (for get action)
            - error: Error description (if success=false)

            Version Object Structure:
            {
                "id": "version-uuid",
                "version_number": 3,
                "field_name": "docs",
                "change_summary": "Added comprehensive validation gates",
                "change_type": "manual",  # or "automatic"
                "created_by": "prp-creator",
                "created_at": "2025-07-30T10:30:00Z",
                "document_id": "doc-uuid",  # if applicable
                "content_preview": "First 200 chars of content..."
            }

        Examples:
            Manual PRP Checkpoint:
                manage_versions(action="create", project_id="uuid", field_name="docs",
                              content={COMPLETE_PRP_JSON}, change_summary="Added validation gates for OAuth implementation",
                              document_id="doc-uuid", created_by="prp-creator")

            List PRP History:
                manage_versions(action="list", project_id="uuid", field_name="docs")

            View Specific Version:
                manage_versions(action="get", project_id="uuid", field_name="docs", version_number=3)

            Restore Previous PRP:
                manage_versions(action="restore", project_id="uuid", field_name="docs",
                              version_number=2, created_by="prp-validator")

            Create Feature Snapshot:
                manage_versions(action="create", project_id="uuid", field_name="features",
                              content={...}, change_summary="Added user authentication feature set")
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            if action == "create":
                if not content:
                    return json.dumps({
                        "success": False,
                        "error": "content is required for create action",
                    })

                # Call Server API to create version
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        urljoin(api_url, f"/api/projects/{project_id}/versions"),
                        json={
                            "field_name": field_name,
                            "content": content,
                            "change_summary": change_summary,
                            "change_type": "manual",
                            "document_id": document_id,
                            "created_by": created_by,
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({
                            "success": True,
                            "version": result.get("version"),
                            "message": result.get("message"),
                        })
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            elif action == "list":
                # Build URL with optional field_name parameter
                params = {}
                if field_name:
                    params["field_name"] = field_name

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(
                        urljoin(api_url, f"/api/projects/{project_id}/versions"), params=params
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, **result})
                    else:
                        return json.dumps({"success": False, "error": "Failed to list versions"})

            elif action == "get":
                if not version_number:
                    return json.dumps({
                        "success": False,
                        "error": "version_number is required for get action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(
                        urljoin(
                            api_url,
                            f"/api/projects/{project_id}/versions/{field_name}/{version_number}",
                        )
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, **result})
                    elif response.status_code == 404:
                        return json.dumps({
                            "success": False,
                            "error": f"Version {version_number} not found",
                        })
                    else:
                        return json.dumps({"success": False, "error": "Failed to get version"})

            elif action == "restore":
                if not version_number:
                    return json.dumps({
                        "success": False,
                        "error": "version_number is required for restore action",
                    })

                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        urljoin(
                            api_url,
                            f"/api/projects/{project_id}/versions/{field_name}/{version_number}/restore",
                        ),
                        json={"restored_by": created_by},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        return json.dumps({"success": True, "message": result.get("message")})
                    else:
                        error_detail = response.text
                        return json.dumps({"success": False, "error": error_detail})

            else:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid action '{action}'. Must be one of: create, list, get, restore",
                })

        except Exception as e:
            logger.error(f"Error in manage_versions: {e}")
            return json.dumps({"success": False, "error": str(e)})

    @mcp.tool()
    async def get_project_features(ctx: Context, project_id: str) -> str:
        """
        Get features from a project's features JSONB field.

        This remains a standalone tool as it's a specific query operation
        that doesn't fit the CRUD pattern of the other tools.

        Args:
            project_id: UUID of the project

        Returns:
            JSON string with list of features
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    urljoin(api_url, f"/api/projects/{project_id}/features")
                )

                if response.status_code == 200:
                    result = response.json()
                    return json.dumps({"success": True, **result})
                elif response.status_code == 404:
                    return json.dumps({"success": False, "error": "Project not found"})
                else:
                    return json.dumps({"success": False, "error": "Failed to get project features"})

        except Exception as e:
            logger.error(f"Error getting project features: {e}")
            return json.dumps({"success": False, "error": str(e)})

    logger.info("✓ Project Module registered with 5 consolidated tools")
