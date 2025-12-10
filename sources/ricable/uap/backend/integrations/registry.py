# File: backend/integrations/registry.py
"""
Integration Registry - Centralized registry for discovering and managing integrations.
"""

from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timezone
from pydantic import BaseModel
from enum import Enum

from .base import IntegrationBase, IntegrationConfig, IntegrationType, AuthMethod


class IntegrationCategory(str, Enum):
    """Categories for organizing integrations"""
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    DEVELOPMENT = "development"
    BUSINESS = "business"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    AI_ML = "ai_ml"
    SECURITY = "security"
    MARKETING = "marketing"
    FINANCE = "finance"


class IntegrationTemplate(BaseModel):
    """Template for creating integration configurations"""
    name: str
    display_name: str
    description: str
    integration_type: IntegrationType
    category: IntegrationCategory
    auth_method: AuthMethod
    base_url: str
    api_version: Optional[str] = None
    logo_url: Optional[str] = None
    documentation_url: Optional[str] = None
    pricing_model: Optional[str] = None
    popularity_score: int = 0
    required_credentials: List[str] = []
    supported_features: List[str] = []
    webhook_events: List[str] = []
    rate_limits: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}


class IntegrationRegistry:
    """
    Central registry for all available integrations.
    
    Provides discovery, templates, and management for third-party integrations.
    """
    
    def __init__(self):
        self.integration_classes: Dict[str, Type[IntegrationBase]] = {}
        self.templates: Dict[str, IntegrationTemplate] = {}
        self._load_default_templates()
    
    def register_integration_class(self, name: str, integration_class: Type[IntegrationBase]):
        """
        Register an integration class.
        
        Args:
            name: Integration name/identifier
            integration_class: Integration class
        """
        self.integration_classes[name] = integration_class
    
    def get_integration_class(self, name: str) -> Optional[Type[IntegrationBase]]:
        """Get integration class by name."""
        return self.integration_classes.get(name)
    
    def register_template(self, template: IntegrationTemplate):
        """
        Register an integration template.
        
        Args:
            template: Integration template
        """
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[IntegrationTemplate]:
        """Get integration template by name."""
        return self.templates.get(name)
    
    def list_templates(self, category: IntegrationCategory = None, 
                      integration_type: IntegrationType = None) -> List[IntegrationTemplate]:
        """
        List available integration templates with optional filtering.
        
        Args:
            category: Optional category filter
            integration_type: Optional type filter
            
        Returns:
            List of matching templates
        """
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if integration_type:
            templates = [t for t in templates if t.integration_type == integration_type]
        
        # Sort by popularity score (descending)
        templates.sort(key=lambda t: t.popularity_score, reverse=True)
        
        return templates
    
    def search_templates(self, query: str) -> List[IntegrationTemplate]:
        """
        Search integration templates by name, description, or features.
        
        Args:
            query: Search query
            
        Returns:
            List of matching templates
        """
        query_lower = query.lower()
        matching_templates = []
        
        for template in self.templates.values():
            # Search in name, display name, description
            if (query_lower in template.name.lower() or
                query_lower in template.display_name.lower() or
                query_lower in template.description.lower()):
                matching_templates.append(template)
            # Search in supported features
            elif any(query_lower in feature.lower() for feature in template.supported_features):
                matching_templates.append(template)
        
        # Sort by popularity score
        matching_templates.sort(key=lambda t: t.popularity_score, reverse=True)
        
        return matching_templates
    
    def create_config_from_template(self, template_name: str, **overrides) -> IntegrationConfig:
        """
        Create integration configuration from template.
        
        Args:
            template_name: Name of template to use
            **overrides: Configuration overrides
            
        Returns:
            IntegrationConfig instance
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        config_data = {
            "name": template.name,
            "display_name": template.display_name,
            "description": template.description,
            "integration_type": template.integration_type,
            "auth_method": template.auth_method,
            "base_url": template.base_url,
            "api_version": template.api_version,
            "rate_limits": template.rate_limits.copy(),
            "metadata": template.metadata.copy()
        }
        
        # Apply overrides
        config_data.update(overrides)
        
        return IntegrationConfig(**config_data)
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories with counts.
        
        Returns:
            List of categories with template counts
        """
        category_counts = {}
        for template in self.templates.values():
            category = template.category
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        categories = []
        for category in IntegrationCategory:
            categories.append({
                "name": category.value,
                "display_name": category.value.replace("_", " ").title(),
                "count": category_counts.get(category, 0)
            })
        
        return categories
    
    def get_popular_integrations(self, limit: int = 10) -> List[IntegrationTemplate]:
        """
        Get most popular integrations.
        
        Args:
            limit: Maximum number of integrations to return
            
        Returns:
            List of popular integration templates
        """
        templates = list(self.templates.values())
        templates.sort(key=lambda t: t.popularity_score, reverse=True)
        return templates[:limit]
    
    def _load_default_templates(self):
        """Load default integration templates."""
        
        # Slack Integration
        self.register_template(IntegrationTemplate(
            name="slack",
            display_name="Slack",
            description="Send messages and notifications to Slack channels",
            integration_type=IntegrationType.CHAT,
            category=IntegrationCategory.COMMUNICATION,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://slack.com/api",
            api_version="v1",
            logo_url="https://cdn.worldvectorlogo.com/logos/slack-new-logo.svg",
            documentation_url="https://api.slack.com/",
            pricing_model="Freemium",
            popularity_score=95,
            required_credentials=["client_id", "client_secret"],
            supported_features=[
                "Send messages", "Create channels", "Manage users", 
                "File uploads", "Webhooks", "Slash commands"
            ],
            webhook_events=["message", "channel_created", "user_joined"],
            rate_limits={"requests_per_minute": 50, "messages_per_second": 1},
            metadata={
                "oauth_scopes": ["chat:write", "channels:read", "users:read"],
                "webhook_verification": "signature"
            }
        ))
        
        # Microsoft Teams Integration
        self.register_template(IntegrationTemplate(
            name="microsoft_teams",
            display_name="Microsoft Teams",
            description="Send messages and collaborate through Microsoft Teams",
            integration_type=IntegrationType.CHAT,
            category=IntegrationCategory.COMMUNICATION,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://graph.microsoft.com",
            api_version="v1.0",
            logo_url="https://cdn.worldvectorlogo.com/logos/microsoft-teams-1.svg",
            documentation_url="https://docs.microsoft.com/en-us/graph/",
            pricing_model="Enterprise",
            popularity_score=85,
            required_credentials=["client_id", "client_secret", "tenant_id"],
            supported_features=[
                "Send messages", "Create teams", "Manage members",
                "File sharing", "Webhooks", "Bot framework"
            ],
            webhook_events=["message", "team_created", "member_added"],
            rate_limits={"requests_per_second": 10},
            metadata={
                "oauth_scopes": ["Chat.ReadWrite", "Team.Create", "User.Read"],
                "authentication": "microsoft_identity_platform"
            }
        ))
        
        # Notion Integration
        self.register_template(IntegrationTemplate(
            name="notion",
            display_name="Notion",
            description="Create and manage pages, databases, and content in Notion",
            integration_type=IntegrationType.PRODUCTIVITY,
            category=IntegrationCategory.PRODUCTIVITY,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://api.notion.com",
            api_version="v1",
            logo_url="https://cdn.worldvectorlogo.com/logos/notion-logo-1.svg",
            documentation_url="https://developers.notion.com/",
            pricing_model="Freemium",
            popularity_score=80,
            required_credentials=["client_id", "client_secret"],
            supported_features=[
                "Create pages", "Query databases", "Update content",
                "File attachments", "Rich text formatting"
            ],
            webhook_events=["page_created", "page_updated", "database_updated"],
            rate_limits={"requests_per_second": 3},
            metadata={
                "oauth_scopes": ["read", "update", "insert"],
                "api_version_header": "Notion-Version"
            }
        ))
        
        # GitHub Integration
        self.register_template(IntegrationTemplate(
            name="github",
            display_name="GitHub",
            description="Manage repositories, issues, and pull requests on GitHub",
            integration_type=IntegrationType.DEVELOPMENT,
            category=IntegrationCategory.DEVELOPMENT,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://api.github.com",
            api_version="v3",
            logo_url="https://cdn.worldvectorlogo.com/logos/github-icon-1.svg",
            documentation_url="https://docs.github.com/en/rest",
            pricing_model="Freemium",
            popularity_score=90,
            required_credentials=["client_id", "client_secret"],
            supported_features=[
                "Repository management", "Issue tracking", "Pull requests",
                "Actions", "Webhooks", "Releases"
            ],
            webhook_events=["push", "pull_request", "issues", "release"],
            rate_limits={"requests_per_hour": 5000},
            metadata={
                "oauth_scopes": ["repo", "issues", "pull_requests"],
                "user_agent_required": True
            }
        ))
        
        # Discord Integration
        self.register_template(IntegrationTemplate(
            name="discord",
            display_name="Discord",
            description="Send messages and manage Discord servers",
            integration_type=IntegrationType.CHAT,
            category=IntegrationCategory.COMMUNICATION,
            auth_method=AuthMethod.BEARER_TOKEN,
            base_url="https://discord.com/api",
            api_version="v10",
            logo_url="https://cdn.worldvectorlogo.com/logos/discord-6.svg",
            documentation_url="https://discord.com/developers/docs",
            pricing_model="Free",
            popularity_score=75,
            required_credentials=["bot_token"],
            supported_features=[
                "Send messages", "Manage channels", "Server management",
                "Slash commands", "Webhooks", "Voice integration"
            ],
            webhook_events=["message", "guild_member_add", "channel_create"],
            rate_limits={"requests_per_second": 5, "messages_per_channel": 5},
            metadata={
                "bot_permissions": ["SEND_MESSAGES", "READ_MESSAGE_HISTORY"],
                "intents": ["GUILD_MESSAGES", "DIRECT_MESSAGES"]
            }
        ))
        
        # Trello Integration
        self.register_template(IntegrationTemplate(
            name="trello",
            display_name="Trello",
            description="Manage Trello boards, lists, and cards",
            integration_type=IntegrationType.PRODUCTIVITY,
            category=IntegrationCategory.PRODUCTIVITY,
            auth_method=AuthMethod.API_KEY,
            base_url="https://api.trello.com",
            api_version="1",
            logo_url="https://cdn.worldvectorlogo.com/logos/trello.svg",
            documentation_url="https://developer.atlassian.com/cloud/trello/",
            pricing_model="Freemium",
            popularity_score=70,
            required_credentials=["api_key", "api_token"],
            supported_features=[
                "Board management", "Card creation", "List management",
                "Attachments", "Webhooks", "Power-ups"
            ],
            webhook_events=["card_created", "card_moved", "board_updated"],
            rate_limits={"requests_per_second": 10, "requests_per_day": 300},
            metadata={
                "authentication": "api_key_token",
                "webhook_callback_url_required": True
            }
        ))
        
        # Google Drive Integration
        self.register_template(IntegrationTemplate(
            name="google_drive",
            display_name="Google Drive",
            description="Manage files and folders in Google Drive",
            integration_type=IntegrationType.STORAGE,
            category=IntegrationCategory.STORAGE,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://www.googleapis.com/drive",
            api_version="v3",
            logo_url="https://cdn.worldvectorlogo.com/logos/google-drive-2020.svg",
            documentation_url="https://developers.google.com/drive/api",
            pricing_model="Freemium",
            popularity_score=85,
            required_credentials=["client_id", "client_secret"],
            supported_features=[
                "File upload", "File download", "Folder management",
                "Sharing", "Search", "Permissions"
            ],
            webhook_events=["file_created", "file_modified", "file_shared"],
            rate_limits={"requests_per_second": 10, "queries_per_day": 1000000},
            metadata={
                "oauth_scopes": ["https://www.googleapis.com/auth/drive"],
                "mime_types": "all"
            }
        ))
        
        # Jira Integration
        self.register_template(IntegrationTemplate(
            name="jira",
            display_name="Jira",
            description="Manage Jira issues, projects, and workflows",
            integration_type=IntegrationType.DEVELOPMENT,
            category=IntegrationCategory.DEVELOPMENT,
            auth_method=AuthMethod.BASIC_AUTH,
            base_url="https://your-domain.atlassian.net/rest/api",
            api_version="3",
            logo_url="https://cdn.worldvectorlogo.com/logos/jira-1.svg",
            documentation_url="https://developer.atlassian.com/cloud/jira/platform/",
            pricing_model="Subscription",
            popularity_score=80,
            required_credentials=["username", "api_token", "base_url"],
            supported_features=[
                "Issue management", "Project administration", "Workflow automation",
                "Custom fields", "Webhooks", "JQL queries"
            ],
            webhook_events=["issue_created", "issue_updated", "project_updated"],
            rate_limits={"requests_per_second": 20},
            metadata={
                "authentication": "basic_auth_with_api_token",
                "cloud_instance_required": True
            }
        ))
        
        # Salesforce Integration
        self.register_template(IntegrationTemplate(
            name="salesforce",
            display_name="Salesforce",
            description="Manage Salesforce CRM data and processes",
            integration_type=IntegrationType.CRM,
            category=IntegrationCategory.BUSINESS,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://your-instance.salesforce.com/services/data",
            api_version="v58.0",
            logo_url="https://cdn.worldvectorlogo.com/logos/salesforce-2.svg",
            documentation_url="https://developer.salesforce.com/docs/api-explorer",
            pricing_model="Enterprise",
            popularity_score=85,
            required_credentials=["client_id", "client_secret", "instance_url"],
            supported_features=[
                "Contact management", "Lead tracking", "Opportunity management",
                "Custom objects", "Reports", "Workflows"
            ],
            webhook_events=["contact_created", "opportunity_updated", "lead_converted"],
            rate_limits={"requests_per_day": 15000},
            metadata={
                "oauth_scopes": ["api", "refresh_token"],
                "sandbox_supported": True
            }
        ))
        
        # HubSpot Integration
        self.register_template(IntegrationTemplate(
            name="hubspot",
            display_name="HubSpot",
            description="Manage HubSpot CRM, marketing, and sales data",
            integration_type=IntegrationType.CRM,
            category=IntegrationCategory.BUSINESS,
            auth_method=AuthMethod.OAUTH2,
            base_url="https://api.hubapi.com",
            api_version="v3",
            logo_url="https://cdn.worldvectorlogo.com/logos/hubspot.svg",
            documentation_url="https://developers.hubspot.com/docs/api/overview",
            pricing_model="Freemium",
            popularity_score=75,
            required_credentials=["client_id", "client_secret"],
            supported_features=[
                "Contact management", "Deal tracking", "Email campaigns",
                "Forms", "Workflows", "Analytics"
            ],
            webhook_events=["contact.creation", "deal.propertyChange", "company.creation"],
            rate_limits={"requests_per_second": 10, "requests_per_day": 40000},
            metadata={
                "oauth_scopes": ["contacts", "automation", "timeline"],
                "webhook_verification": "signature"
            }
        ))
        
        # Discord Integration (Example Plugin-based Integration)
        self.register_template(IntegrationTemplate(
            name="discord",
            display_name="Discord",
            description="Send messages and manage Discord servers",
            integration_type=IntegrationType.CHAT,
            category=IntegrationCategory.COMMUNICATION,
            auth_method=AuthMethod.BEARER_TOKEN,
            base_url="https://discord.com/api",
            api_version="v10",
            logo_url="https://cdn.worldvectorlogo.com/logos/discord-6.svg",
            documentation_url="https://discord.com/developers/docs",
            pricing_model="Free",
            popularity_score=85,
            required_credentials=["bot_token"],
            supported_features=[
                "Send messages", "Manage channels", "Server management",
                "Slash commands", "Webhooks", "Voice integration",
                "Bot commands", "Member management"
            ],
            webhook_events=["message", "guild_member_add", "channel_create", "interaction_create"],
            rate_limits={"requests_per_second": 5, "messages_per_channel": 5},
            metadata={
                "bot_permissions": ["SEND_MESSAGES", "READ_MESSAGE_HISTORY"],
                "intents": ["GUILD_MESSAGES", "DIRECT_MESSAGES"],
                "plugin_based": True
            }
        ))