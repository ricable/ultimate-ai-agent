from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from typing import Optional, List
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire

# Configure logging
logfire.configure(send_to_logfire='if-token-present')

# Load environment variables
load_dotenv()

# Initialize OpenAI model with OpenRouter
llm = os.getenv('LLM_MODEL', 'deepseek/deepseek-chat')
model = OpenAIModel(
    llm,
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPEN_ROUTER_API_KEY')
)

@dataclass
class HunterDeps:
    """Dependencies for the Hunter.io API agent."""
    client: httpx.AsyncClient
    hunter_api_key: str | None = None

system_prompt = """
You are a lead generation expert with access to Hunter.io API to help users find business email addresses and generate leads.

Your capabilities include:
1. Finding email addresses for specific domains
2. Getting email counts and statistics for domains
3. Verifying email addresses
4. Searching for specific people's email addresses

Always verify the domain/email exists before providing information. Be precise and professional in your responses.

When answering, format the response clearly with sections for:
- Domain Information
- Email Addresses Found (if applicable)
- Verification Status (if applicable)
- Department Statistics (if applicable)

For each response, start with a brief summary of what you found, then provide the detailed information.
"""

# Initialize the Hunter.io agent
hunter_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=HunterDeps,
    retries=2
)

# Mock data generators for simulating Hunter.io API responses
def generate_mock_email_count(domain: str) -> dict:
    """Generate mock email count data for a specific domain."""
    return {
        'total': 150,
        'personal_emails': 100,
        'generic_emails': 50,
        'department': {
            'executive': 15,
            'it': 35,
            'sales': 60,
            'marketing': 40
        }
    }

def generate_mock_domain_emails(domain: str, limit: int = 10) -> list:
    """Generate mock email data for a specific domain."""
    common_names = [
        ('john.doe', 'Software Engineer'),
        ('jane.smith', 'Product Manager'),
        ('mike.wilson', 'Sales Director'),
        ('sarah.jones', 'Marketing Manager'),
        ('alex.brown', 'CTO'),
        ('emma.davis', 'HR Manager'),
        ('david.miller', 'Frontend Developer'),
        ('lisa.taylor', 'Backend Engineer'),
        ('james.anderson', 'Data Scientist'),
        ('rachel.white', 'UX Designer')
    ]
    
    emails = []
    for i in range(min(limit, len(common_names))):
        name, position = common_names[i]
        emails.append({
            'value': f'{name}@{domain}',
            'type': 'personal',
            'confidence': 85 + (i % 3) * 5,  # Varies between 85-95
            'position': position
        })
    
    # Add one generic email if we haven't hit the limit
    if limit > len(emails):
        emails.append({
            'value': f'info@{domain}',
            'type': 'generic',
            'confidence': 100,
            'position': None
        })
    
    return emails

def generate_mock_email_verification(email: str) -> dict:
    """Generate mock verification data for a specific email."""
    # Parse domain from email
    _, domain = email.split('@')
    
    # More realistic verification based on email pattern
    is_generic = email.startswith(('info@', 'contact@', 'support@'))
    return {
        'status': 'valid',
        'score': 90 if not is_generic else 95,
        'disposable': False,
        'webmail': domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'],
        'mx_records': True,
        'smtp_check': True
    }

@hunter_agent.tool
async def get_email_count(ctx: RunContext[HunterDeps], domain: str) -> str:
    """Get the count and statistics of email addresses for a domain.

    Args:
        ctx: The run context containing dependencies.
        domain: The domain name to search (e.g., 'google.com').

    Returns:
        str: Formatted string containing email statistics for the domain.
    """
    data = generate_mock_email_count(domain)
    
    return (
        f"Domain: {domain}\n"
        f"Total emails found: {data['total']}\n"
        f"Personal emails: {data['personal_emails']}\n"
        f"Generic emails: {data['generic_emails']}\n"
        f"\nDepartment breakdown:\n"
        f"- Executive: {data['department']['executive']}\n"
        f"- IT: {data['department']['it']}\n"
        f"- Sales: {data['department']['sales']}\n"
        f"- Marketing: {data['department']['marketing']}"
    )

@hunter_agent.tool
async def domain_search(
    ctx: RunContext[HunterDeps], 
    domain: str,
    limit: Optional[int] = 10
) -> str:
    """Search for email addresses from a specific domain."""
    try:
        emails = generate_mock_domain_emails(domain, limit)
        
        result = [f"Domain: {domain}"]
        for email in emails:
            confidence = email.get('confidence', 'N/A')
            position = email.get('position', 'N/A')
            result.append(
                f"\nEmail: {email['value']}\n"
                f"Type: {email['type']}\n"
                f"Confidence: {confidence}%\n"
                f"Position: {position}"
            )
        
        return "\n".join(result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

@hunter_agent.tool
async def verify_email(ctx: RunContext[HunterDeps], email: str) -> str:
    """Verify if an email address is valid and get detailed information.

    Args:
        ctx: The run context containing dependencies.
        email: The email address to verify.

    Returns:
        str: Formatted string containing verification results and status.
    """
    data = generate_mock_email_verification(email)
    
    return (
        f"Email: {email}\n"
        f"Status: {data['status']}\n"
        f"Score: {data['score']}\n"
        f"Disposable: {'Yes' if data['disposable'] else 'No'}\n"
        f"Webmail: {'Yes' if data['webmail'] else 'No'}\n"
        f"MX Records: {'Yes' if data['mx_records'] else 'No'}\n"
        f"SMTP Valid: {'Yes' if data['smtp_check'] else 'No'}"
    )
