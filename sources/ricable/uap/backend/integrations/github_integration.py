# File: backend/integrations/github_integration.py
"""
GitHub integration implementation for repository and development workflow management.
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from urllib.parse import urlencode

from .base import (
    IntegrationBase, IntegrationConfig, IntegrationResponse, 
    IntegrationEvent, IntegrationError, WebhookIntegration
)
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class GitHubIntegration(WebhookIntegration):
    """
    GitHub integration with full API support for repositories, issues, and pull requests.
    
    Supports OAuth2 authentication, repository management, issue creation,
    and webhook event processing.
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.user_login: Optional[str] = None
        
        # GitHub API endpoints
        self.endpoints = {
            "user": "/user",
            "repos": "/user/repos",
            "repo": "/repos/{owner}/{repo}",
            "issues": "/repos/{owner}/{repo}/issues",
            "issue": "/repos/{owner}/{repo}/issues/{issue_number}",
            "pulls": "/repos/{owner}/{repo}/pulls",
            "pull": "/repos/{owner}/{repo}/pulls/{pull_number}",
            "commits": "/repos/{owner}/{repo}/commits",
            "releases": "/repos/{owner}/{repo}/releases",
            "webhooks": "/repos/{owner}/{repo}/hooks",
            "oauth_access": "https://github.com/login/oauth/access_token"
        }
    
    async def initialize(self) -> IntegrationResponse:
        """Initialize GitHub integration with HTTP session."""
        try:
            # Initialize HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "UAP-GitHub-Integration/1.0",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
            )
            
            # Call parent initialization
            parent_response = await super().initialize()
            if not parent_response.success:
                return parent_response
            
            uap_logger.log_event(
                LogLevel.INFO,
                "GitHub integration initialized successfully",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "github_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={"status": "initialized", "api_version": "2022-11-28"}
            )
            
        except Exception as e:
            return self._format_error_response(e, "github_init_failed")
    
    async def authenticate(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate with GitHub using OAuth2 or personal access token.
        
        Args:
            credentials: Dictionary containing authentication data
                - For OAuth2: {"code": "auth_code"}
                - For PAT: {"access_token": "ghp_..."}
        
        Returns:
            Authentication response
        """
        try:
            if "access_token" in credentials:
                # Direct personal access token authentication
                return await self._authenticate_token(credentials["access_token"])
            
            elif "code" in credentials:
                # OAuth2 code exchange
                return await self._authenticate_oauth2(credentials)
            
            else:
                raise IntegrationError(
                    "Invalid credentials format. Provide either 'access_token' or OAuth2 'code'",
                    self.integration_id
                )
            
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "github_auth_failed")
    
    async def _authenticate_token(self, access_token: str) -> IntegrationResponse:
        """Authenticate using personal access token."""
        self.access_token = access_token
        
        # Test authentication
        test_response = await self.test_connection()
        if test_response.success:
            uap_logger.log_event(
                LogLevel.INFO,
                "GitHub token authentication successful",
                EventType.AUTHENTICATION,
                {"integration_id": self.integration_id},
                "github_integration"
            )
            return IntegrationResponse(
                success=True,
                data=test_response.data,
                metadata={"auth_method": "personal_access_token"}
            )
        else:
            return test_response
    
    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """Authenticate using OAuth2 code exchange."""
        try:
            token_data = {
                "client_id": self.config.auth_config.get("client_id"),
                "client_secret": self.config.auth_config.get("client_secret"),
                "code": credentials["code"]
            }
            
            if not all([token_data["client_id"], token_data["client_secret"], token_data["code"]]):
                raise IntegrationError("Missing OAuth2 configuration", self.integration_id)
            
            headers = {
                "Accept": "application/json",
                "User-Agent": "UAP-GitHub-Integration/1.0"
            }
            
            # Exchange code for tokens
            async with self.session.post(
                self.endpoints["oauth_access"],
                headers=headers,
                data=token_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise IntegrationError(f"OAuth2 exchange failed: {response.status} - {error_text}", self.integration_id)
                
                result = await response.json()
                
                if "error" in result:
                    raise IntegrationError(f"GitHub OAuth2 error: {result.get('error_description', result.get('error'))}", self.integration_id)
                
                # Store tokens
                self.access_token = result.get("access_token")
                
                # Get user info to validate token
                user_response = await self.get_user_info()
                if not user_response.success:
                    return user_response
                
                self.user_login = user_response.data.get("login")
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    "GitHub OAuth2 authentication successful",
                    EventType.AUTHENTICATION,
                    {
                        "integration_id": self.integration_id,
                        "user_login": self.user_login
                    },
                    "github_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "user_info": user_response.data,
                        "scope": result.get("scope"),
                        "token_type": result.get("token_type")
                    },
                    metadata={"auth_method": "oauth2"}
                )
                
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "github_oauth2_failed")
    
    async def test_connection(self) -> IntegrationResponse:
        """Test connection to GitHub API."""
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="No authentication token available",
                    error_code="no_token"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            async with self.session.get(
                f"{self.config.base_url}{self.endpoints['user']}",
                headers=headers
            ) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be invalid",
                        error_code="auth_failed"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Connection test failed: HTTP {response.status}",
                        error_code="connection_failed"
                    )
                
                result = await response.json()
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "login": result.get("login"),
                        "id": result.get("id"),
                        "name": result.get("name"),
                        "email": result.get("email"),
                        "company": result.get("company"),
                        "public_repos": result.get("public_repos"),
                        "followers": result.get("followers")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_test_failed")
    
    async def send_message(self, message: str, channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Create an issue or comment on GitHub.
        
        Args:
            message: Issue body or comment text
            channel: Repository (format: "owner/repo") or issue URL
            **kwargs: Additional parameters (title, labels, assignees, etc.)
        
        Returns:
            Issue creation/comment response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            if not channel:
                return IntegrationResponse(
                    success=False,
                    error="Repository is required (format: 'owner/repo' or 'owner/repo#issue_number')",
                    error_code="missing_repository"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Parse channel format
            if "#" in channel:
                # Comment on existing issue
                repo_part, issue_number = channel.split("#", 1)
                owner, repo = repo_part.split("/", 1)
                return await self._create_issue_comment(owner, repo, int(issue_number), message, headers)
            else:
                # Create new issue
                owner, repo = channel.split("/", 1)
                return await self._create_issue(owner, repo, message, headers, **kwargs)
                
        except ValueError:
            return IntegrationResponse(
                success=False,
                error="Invalid repository format. Use 'owner/repo' or 'owner/repo#issue_number'",
                error_code="invalid_format"
            )
        except Exception as e:
            return self._format_error_response(e, "github_send_failed")
    
    async def _create_issue(self, owner: str, repo: str, body: str, headers: Dict[str, str], **kwargs) -> IntegrationResponse:
        """Create a new issue in a GitHub repository."""
        try:
            title = kwargs.get("title", f"UAP Issue - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            issue_data = {
                "title": title,
                "body": body
            }
            
            # Add optional parameters
            if "labels" in kwargs:
                issue_data["labels"] = kwargs["labels"]
            if "assignees" in kwargs:
                issue_data["assignees"] = kwargs["assignees"]
            if "milestone" in kwargs:
                issue_data["milestone"] = kwargs["milestone"]
            
            url = f"{self.config.base_url}{self.endpoints['issues'].format(owner=owner, repo=repo)}"
            
            async with self.session.post(url, headers=headers, json=issue_data) as response:
                if response.status == 404:
                    return IntegrationResponse(
                        success=False,
                        error=f"Repository {owner}/{repo} not found or not accessible",
                        error_code="repo_not_found"
                    )
                elif response.status not in [200, 201]:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to create issue: HTTP {response.status} - {error_text}",
                        error_code="create_failed"
                    )
                
                result = await response.json()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"GitHub issue created: {owner}/{repo}#{result.get('number')}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "repository": f"{owner}/{repo}",
                        "issue_number": result.get("number"),
                        "issue_id": result.get("id")
                    },
                    "github_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "number": result.get("number"),
                        "title": result.get("title"),
                        "url": result.get("html_url"),
                        "state": result.get("state"),
                        "created_at": result.get("created_at"),
                        "repository": f"{owner}/{repo}"
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_create_issue_failed")
    
    async def _create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str, headers: Dict[str, str]) -> IntegrationResponse:
        """Add a comment to an existing GitHub issue."""
        try:
            comment_data = {"body": body}
            url = f"{self.config.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
            
            async with self.session.post(url, headers=headers, json=comment_data) as response:
                if response.status == 404:
                    return IntegrationResponse(
                        success=False,
                        error=f"Issue {owner}/{repo}#{issue_number} not found",
                        error_code="issue_not_found"
                    )
                elif response.status not in [200, 201]:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to create comment: HTTP {response.status} - {error_text}",
                        error_code="comment_failed"
                    )
                
                result = await response.json()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"GitHub comment created on {owner}/{repo}#{issue_number}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "repository": f"{owner}/{repo}",
                        "issue_number": issue_number,
                        "comment_id": result.get("id")
                    },
                    "github_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "url": result.get("html_url"),
                        "created_at": result.get("created_at"),
                        "issue_number": issue_number,
                        "repository": f"{owner}/{repo}"
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_create_comment_failed")
    
    async def get_user_info(self, user_id: str = None) -> IntegrationResponse:
        """
        Get user information from GitHub.
        
        Args:
            user_id: GitHub username (optional, defaults to authenticated user)
        
        Returns:
            User information response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            if user_id:
                url = f"{self.config.base_url}/users/{user_id}"
            else:
                url = f"{self.config.base_url}{self.endpoints['user']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be invalid",
                        error_code="auth_failed"
                    )
                elif response.status == 404:
                    return IntegrationResponse(
                        success=False,
                        error=f"User {user_id} not found",
                        error_code="user_not_found"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get user info: HTTP {response.status}",
                        error_code="user_info_failed"
                    )
                
                result = await response.json()
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "login": result.get("login"),
                        "name": result.get("name"),
                        "email": result.get("email"),
                        "avatar_url": result.get("avatar_url"),
                        "company": result.get("company"),
                        "location": result.get("location"),
                        "bio": result.get("bio"),
                        "public_repos": result.get("public_repos"),
                        "followers": result.get("followers"),
                        "following": result.get("following"),
                        "created_at": result.get("created_at")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_user_info_failed")
    
    async def refresh_credentials(self) -> IntegrationResponse:
        """
        Refresh authentication credentials.
        
        Note: GitHub tokens don't automatically refresh.
        """
        try:
            # Test current credentials
            test_response = await self.test_connection()
            if test_response.success:
                return IntegrationResponse(
                    success=True,
                    data={"status": "credentials_valid", "refresh_needed": False}
                )
            else:
                return IntegrationResponse(
                    success=False,
                    error="Credentials need manual refresh",
                    error_code="manual_refresh_required",
                    data={"refresh_needed": True}
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_refresh_failed")
    
    async def verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """
        Verify GitHub webhook signature.
        
        Args:
            payload: Raw webhook payload
            headers: HTTP headers from webhook request
        
        Returns:
            True if signature is valid
        """
        try:
            import hmac
            import hashlib
            
            signature = headers.get("X-Hub-Signature-256")
            if not signature:
                return False
            
            if not self.webhook_secret:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    "GitHub webhook secret not configured",
                    EventType.SECURITY,
                    {"integration_id": self.integration_id},
                    "github_integration"
                )
                return False
            
            expected_signature = "sha256=" + hmac.new(
                self.webhook_secret.encode(),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"GitHub webhook signature verification error: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "github_integration"
            )
            return False
    
    async def parse_webhook_event(self, payload: Dict[str, Any], headers: Dict[str, str]) -> IntegrationEvent:
        """
        Parse GitHub webhook payload into standardized event format.
        
        Args:
            payload: Webhook payload from GitHub
            headers: HTTP headers
        
        Returns:
            Standardized IntegrationEvent
        """
        try:
            # Get event type from headers
            event_type = headers.get("X-GitHub-Event", "unknown")
            action = payload.get("action")
            
            # Create event type string
            if action:
                full_event_type = f"{event_type}.{action}"
            else:
                full_event_type = event_type
            
            # Extract common metadata
            metadata = {
                "event_type": event_type,
                "action": action,
                "webhook_headers": dict(headers)
            }
            
            # Add event-specific metadata
            if "repository" in payload:
                metadata["repository"] = payload["repository"].get("full_name")
                metadata["repository_id"] = payload["repository"].get("id")
            
            if "sender" in payload:
                metadata["sender"] = payload["sender"].get("login")
                metadata["sender_id"] = payload["sender"].get("id")
            
            if "issue" in payload:
                metadata["issue_number"] = payload["issue"].get("number")
                metadata["issue_id"] = payload["issue"].get("id")
            
            if "pull_request" in payload:
                metadata["pull_request_number"] = payload["pull_request"].get("number")
                metadata["pull_request_id"] = payload["pull_request"].get("id")
            
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type=full_event_type,
                source="github",
                data=payload,
                metadata=metadata
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to parse GitHub webhook event: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "github_integration"
            )
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type="parse_error",
                source="github",
                data=payload,
                metadata={"error": str(e)}
            )
    
    async def receive_webhook(self, event: IntegrationEvent) -> IntegrationResponse:
        """
        Process incoming webhook from GitHub.
        
        Args:
            event: Webhook event data
        
        Returns:
            Processing response
        """
        try:
            event_type = event.event_type
            
            # Handle different event types
            if event_type.startswith("issues."):
                return await self._handle_issue_event(event)
            elif event_type.startswith("pull_request."):
                return await self._handle_pull_request_event(event)
            elif event_type.startswith("push"):
                return await self._handle_push_event(event)
            elif event_type.startswith("release."):
                return await self._handle_release_event(event)
            else:
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Received GitHub event type: {event_type}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": self.integration_id,
                        "event_type": event_type,
                        "repository": event.metadata.get("repository")
                    },
                    "github_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={"status": "acknowledged", "event_type": event_type}
                )
            
        except Exception as e:
            return self._format_error_response(e, "github_webhook_failed")
    
    async def _handle_issue_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle GitHub issue events."""
        try:
            action = event.metadata.get("action")
            issue_number = event.metadata.get("issue_number")
            repository = event.metadata.get("repository")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"GitHub issue {action}: {repository}#{issue_number}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "action": action,
                    "issue_number": issue_number,
                    "repository": repository
                },
                "github_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "issue",
                    "action": action,
                    "issue_number": issue_number,
                    "repository": repository
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "github_issue_event_failed")
    
    async def _handle_pull_request_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle GitHub pull request events."""
        try:
            action = event.metadata.get("action")
            pr_number = event.metadata.get("pull_request_number")
            repository = event.metadata.get("repository")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"GitHub pull request {action}: {repository}#{pr_number}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "action": action,
                    "pull_request_number": pr_number,
                    "repository": repository
                },
                "github_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "pull_request",
                    "action": action,
                    "pull_request_number": pr_number,
                    "repository": repository
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "github_pr_event_failed")
    
    async def _handle_push_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle GitHub push events."""
        try:
            repository = event.metadata.get("repository")
            commits = event.data.get("commits", [])
            ref = event.data.get("ref", "")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"GitHub push to {repository}: {len(commits)} commits on {ref}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "repository": repository,
                    "commit_count": len(commits),
                    "ref": ref
                },
                "github_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "push",
                    "repository": repository,
                    "commit_count": len(commits),
                    "ref": ref
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "github_push_event_failed")
    
    async def _handle_release_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle GitHub release events."""
        try:
            action = event.metadata.get("action")
            repository = event.metadata.get("repository")
            release = event.data.get("release", {})
            tag_name = release.get("tag_name")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"GitHub release {action}: {repository} - {tag_name}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "action": action,
                    "repository": repository,
                    "tag_name": tag_name
                },
                "github_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "release",
                    "action": action,
                    "repository": repository,
                    "tag_name": tag_name
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "github_release_event_failed")
    
    async def cleanup(self) -> IntegrationResponse:
        """Clean up GitHub integration resources."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.access_token = None
            self.user_login = None
            
            parent_response = await super().cleanup()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "GitHub integration cleanup completed",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "github_integration"
            )
            
            return parent_response
            
        except Exception as e:
            return self._format_error_response(e, "github_cleanup_failed")
    
    def get_oauth2_authorization_url(self, scopes: List[str], state: str = None, redirect_uri: str = None) -> str:
        """
        Generate GitHub OAuth2 authorization URL.
        
        Args:
            scopes: List of OAuth2 scopes to request
            state: Optional state parameter for security
            redirect_uri: Callback URL for authorization
        
        Returns:
            Authorization URL
        """
        client_id = self.config.auth_config.get("client_id")
        if not client_id:
            raise IntegrationError("Client ID not configured", self.integration_id)
        
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri or self.config.auth_config.get("redirect_uri"),
            "scope": " ".join(scopes)
        }
        
        if state:
            params["state"] = state
        
        return f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    
    async def get_repositories(self, visibility: str = "all", sort: str = "updated", per_page: int = 30) -> IntegrationResponse:
        """
        Get repositories for the authenticated user.
        
        Args:
            visibility: Repository visibility ("all", "public", "private")
            sort: Sort order ("created", "updated", "pushed", "full_name")
            per_page: Number of repositories per page
        
        Returns:
            Repositories list response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {
                "visibility": visibility,
                "sort": sort,
                "per_page": min(per_page, 100)
            }
            
            async with self.session.get(
                f"{self.config.base_url}{self.endpoints['repos']}",
                headers=headers,
                params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get repositories: HTTP {response.status} - {error_text}",
                        error_code="repos_failed"
                    )
                
                result = await response.json()
                
                repositories = []
                for repo in result:
                    repositories.append({
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "full_name": repo.get("full_name"),
                        "description": repo.get("description"),
                        "private": repo.get("private"),
                        "html_url": repo.get("html_url"),
                        "language": repo.get("language"),
                        "stargazers_count": repo.get("stargazers_count"),
                        "forks_count": repo.get("forks_count"),
                        "updated_at": repo.get("updated_at"),
                        "default_branch": repo.get("default_branch")
                    })
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "repositories": repositories,
                        "total_count": len(repositories)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "github_repos_failed")