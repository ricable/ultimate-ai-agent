import asyncio
import httpx
import logging
import time
from langchain_core.tools import tool
from bs4 import BeautifulSoup, Comment
from dataclasses import dataclass
from typing import Dict, Optional, Any
from urllib.parse import urlparse
import pypandoc
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class NetworkError(Exception):
    """Raised when network-related errors occur."""
    pass

class TransientError(Exception):
    """Raised for temporary errors that may succeed on retry."""
    pass

class PermanentError(Exception):
    """Raised for errors that won't succeed on retry."""
    pass

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    jitter: float = 0.1

@dataclass
class RateLimitConfig:
    max_concurrent: int = 10
    burst_size: int = 5
    refill_rate: float = 1.0

class RetryStrategy:
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def should_retry(self, attempt: int, error: Exception) -> bool:
        if attempt >= self.config.max_retries:
            return False
        return isinstance(error, TransientError)

class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = asyncio.Lock()
        self.concurrent_requests = 0
        self.tokens = config.burst_size
        self.domain_tokens = {}

def clean_html_only(html_content: str) -> str:
    """Clean HTML content without converting to markdown.

    Args:
        html_content: Raw HTML content to clean

    Returns:
        Cleaned HTML with scripts and styles removed
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements but preserve 'pre' tags
        for element in soup.find_all(['script', 'style']):
            element.decompose()

        # Remove comments
        for comment in soup.find_all(string=lambda string: isinstance(string, Comment)):
            comment.extract()

        # Optionally, remove unnecessary attributes
        for tag in soup.find_all():
            attrs = tag.attrs
            # Preserve 'href' in 'a' tags and 'src' in 'img' tags
            if tag.name == 'a':
                tag.attrs = {'href': attrs.get('href', '')}
            elif tag.name == 'img':
                tag.attrs = {'src': attrs.get('src', '')}
            else:
                tag.attrs = {}

        return str(soup)
    except Exception as e:
        logger.error(f"Error cleaning HTML content: {str(e)}")
        return html_content

@tool("scrape_url")
def scrape_url_tool(
    url: str,
    use_playwright: bool = False,
    verify_ssl: bool = True,
    user_agent: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None,
    output_format: str = 'markdown'
) -> Dict[str, Any]:
    """Scrape content from a URL using either httpx or Playwright.
    
    Args:
        url: URL to scrape
        use_playwright: Whether to use Playwright for JavaScript rendering
        verify_ssl: Whether to verify SSL certificates
        user_agent: Custom user agent string
        retry_config: Configuration for retry behavior
        output_format: Output format ('markdown' or 'html')
        
    Returns:
        Dict containing scraped content and metadata
    """
    # Validate URL
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")
    except Exception:
        raise ValueError("Invalid URL")

    retry_config = retry_config or RetryConfig()
    strategy = RetryStrategy(retry_config)
    
    headers = DEFAULT_HEADERS.copy()
    if user_agent:
        headers["User-Agent"] = user_agent

    metrics = {"attempts": 0}
    
    async def scrape_with_httpx() -> str:
        async with httpx.AsyncClient(
            verify=verify_ssl,
            headers=headers,
            follow_redirects=True
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                raise NetworkError(f"HTTP error: {str(e)}")

    def scrape_with_playwright() -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                user_agent=user_agent if user_agent else DEFAULT_HEADERS["User-Agent"]
            )
            page = context.new_page()
            try:
                page.goto(url, wait_until="networkidle")
                return page.content()
            finally:
                context.close()
                browser.close()

    attempt = 0
    last_error = None
    
    while True:
        try:
            metrics["attempts"] += 1
            
            # Get raw HTML content
            if use_playwright:
                html_content = scrape_with_playwright()
            else:
                html_content = asyncio.run(scrape_with_httpx())
            
            # Clean the HTML
            cleaned_html = clean_html_only(html_content)
            
            # Convert to markdown if requested
            if output_format == 'markdown':
                try:
                    content = pypandoc.convert_text(
                        cleaned_html,
                        'markdown',
                        format='html'
                    )
                except Exception as e:
                    logger.warning(f"Pandoc conversion failed: {e}")
                    content = cleaned_html
            else:
                content = cleaned_html

            return {
                "content": content,
                "success": True,
                "metrics": metrics
            }

        except Exception as e:
            last_error = e
            if not strategy.should_retry(attempt, e):
                break
            attempt += 1
            time.sleep(retry_config.base_delay * (2 ** attempt))
    
    if last_error:
        raise last_error
    
    return {
        "content": "",
        "success": False,
        "metrics": metrics
    }
