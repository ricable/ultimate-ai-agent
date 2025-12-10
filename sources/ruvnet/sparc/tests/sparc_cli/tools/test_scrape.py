import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from bs4 import BeautifulSoup
from sparc_cli.tools.scrape import RateLimiter, RateLimitConfig
from sparc_cli.tools.scrape import (
    scrape_url_tool,
    DEFAULT_HEADERS,
    RetryConfig,
    RetryStrategy,
    NetworkError,
    TransientError,
    PermanentError
)

@pytest.fixture(autouse=True)
def setup_rate_limiter():
    """Create fresh rate limiter between tests"""
    global rate_limiter
    limiter = RateLimiter(RateLimitConfig(max_concurrent=10))
    rate_limiter = limiter
    yield limiter
    with limiter._lock:
        limiter.concurrent_requests = 0
        limiter.tokens = limiter.config.burst_size
        limiter.domain_tokens.clear()

# Test HTML content
TEST_HTML = """
<html>
    <body>
        <div>
            <h1>Example Domain</h1>
            <p>This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.</p>
            <p><a href="https://www.iana.org/domains/example">More information...</a></p>
        </div>
    </body>
</html>
"""

TEST_MARKDOWN = "# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.\n\n[More information...](https://www.iana.org/domains/example)"


@pytest_asyncio.fixture
async def mock_httpx():
    with patch('httpx.AsyncClient') as mock:
        response_mock = AsyncMock()
        client_mock = AsyncMock()
        context_mock = AsyncMock()
        
        client_mock.get = AsyncMock(side_effect=NetworkError("Network error"))
        context_mock.__aenter__.return_value = client_mock
        mock.return_value = context_mock
        
        yield mock

@pytest_asyncio.fixture 
async def mock_playwright():
    with patch('playwright.sync_api.sync_playwright') as mock:
        browser_mock = MagicMock()
        context_mock = MagicMock()
        page_mock = MagicMock()
        
        mock.return_value.__enter__.return_value.chromium.launch = MagicMock(return_value=browser_mock)
        browser_mock.new_context = MagicMock(return_value=context_mock)
        context_mock.new_page = MagicMock(return_value=page_mock)
        
        yield mock

@pytest_asyncio.fixture
async def mock_pandoc():
    with patch('pypandoc.convert_text') as mock:
        mock.return_value = TEST_MARKDOWN
        yield mock

@pytest.mark.asyncio
async def test_successful_playwright_scrape(mock_pandoc):
    """Test successful scraping using Playwright."""
    with patch('playwright.sync_api.sync_playwright') as mock:
        browser_mock = MagicMock()
        context_mock = MagicMock()
        page_mock = MagicMock()
        
        mock.return_value.__enter__.return_value.chromium.launch = MagicMock(return_value=browser_mock)
        browser_mock.new_context = MagicMock(return_value=context_mock) 
        context_mock.new_page = MagicMock(return_value=page_mock)
        page_mock.content = MagicMock(return_value=TEST_HTML)
        
        result = await scrape_url_tool(
            url="https://example.com",
            use_playwright=True,
            verify_ssl=True
        )
        
        assert isinstance(result, dict)
        assert "content" in result
        assert "Example Domain" in result["content"]
        assert "This domain is for use in illustrative examples" in result["content"]
        assert "script" not in result["content"]
        assert "style" not in result["content"]

@pytest.mark.asyncio
async def test_successful_httpx_scrape(mock_httpx, mock_pandoc):
    """Test successful scraping using httpx."""
    # Setup mock response
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = TEST_HTML

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    client_mock.get.side_effect = None  # Ensure no side effect

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    result = await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        verify_ssl=True
    )
    
    assert isinstance(result, dict)
    assert "content" in result
    assert "Example Domain" in result["content"]

@pytest.mark.asyncio
async def test_failed_scrape_playwright_with_retry(mock_playwright):
    """Test handling of Playwright scraping failure with retry."""
    page_mock = AsyncMock()
    page_mock.content.return_value = TEST_HTML
    mock_playwright.return_value.__aenter__.return_value.new_page.side_effect = [
        Exception("Temporary error"),
        Exception("Temporary error"),
        page_mock  # Success on third try
    ]
    
    # First try - should succeed after retries
    result = await scrape_url_tool(
        url="https://example.com", 
        use_playwright=True,
        retry_config=RetryConfig(max_retries=3)
    )
    
    assert result["success"] is True
    assert "metrics" in result

@pytest.mark.asyncio
async def test_retry_strategy():
    """Test retry strategy behavior"""
    strategy = RetryStrategy(RetryConfig(max_retries=3))
    
    # Should retry on transient errors
    assert strategy.should_retry(0, TransientError()) is True
    assert strategy.should_retry(1, TransientError()) is True
    assert strategy.should_retry(2, TransientError()) is True
    assert strategy.should_retry(3, TransientError()) is False
    
    # Should not retry on permanent errors
    assert strategy.should_retry(0, PermanentError()) is False

@pytest.mark.asyncio
async def test_failed_scrape_httpx(mock_httpx):
    """Test handling of httpx scraping failure."""
    with pytest.raises(NetworkError) as exc_info:
        await scrape_url_tool(url="https://example.com", use_playwright=False)
    assert "Network error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_ssl_verification(mock_httpx):
    """Test SSL verification options."""
    # Setup mock response
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = "<html><body><h1>SSL Test</h1></body></html>"

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    client_mock.get.side_effect = None  # Ensure no side effect

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        verify_ssl=False
    )

    # Verify httpx was called with verify=False
    mock_httpx.assert_called_with(verify=False, headers=DEFAULT_HEADERS, follow_redirects=True)

@pytest.mark.asyncio
async def test_user_agent_handling(mock_httpx):
    """Test custom user agent handling."""
    # Setup async mocks
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = "<html><body><h1>Test Page</h1></body></html>"

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    custom_ua = "Custom User Agent"
    expected_headers = DEFAULT_HEADERS.copy()
    expected_headers["User-Agent"] = custom_ua

    await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        user_agent=custom_ua
    )

    # Verify headers
    mock_httpx.assert_called_once_with(
        headers=expected_headers,
        verify=True, 
        follow_redirects=True
    )

@pytest.mark.asyncio
async def test_html_cleaning(mock_httpx):
    """Test HTML content cleaning functionality with different output formats."""
    html = """<!DOCTYPE html>
<html>
    <head>
        <title>Test Page</title>
        <style>
            /* CSS that should be removed */
            body { color: red; }
            div.important { display: none; }
        </style>
    </head>
    <body>
        <h1>Main Title</h1>
        <script>
            // JavaScript with nested content that should be removed
            document.write('<p>Hidden content</p>');
            const text = "This script text should be removed";
        </script>
        <div class="content">
            <p>First paragraph with  multiple    spaces</p>
            <script>console.log('<p>More hidden content</p>');</script>
            <p>Second paragraph</p>
            <style>.nested { color: blue; }</style>
            <div class="nested">
                <p>Nested content that should remain</p>
            </div>
        </div>
        <pre>
            Preformatted
                text should
                    preserve spacing
        </pre>
    </body>
</html>"""

    # Create response mock with proper context manager structure
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = html
    
    # Create client mock
    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    
    # Setup context manager mock
    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    result = await scrape_url_tool(url="https://example.com", use_playwright=False)
    cleaned = result["content"]
    soup = BeautifulSoup(cleaned, 'html.parser')
    
    # Verify removal of script and style tags
    assert soup.find('script') is None
    assert soup.find('style') is None
    
    # Verify content preservation
    assert "Main Title" in cleaned
    assert "First paragraph" in cleaned
    assert "Second paragraph" in cleaned
    assert "Nested content that should remain" in cleaned
    assert "Preformatted" in cleaned
    
    # Verify removal of script/style content
    assert "Hidden content" not in cleaned
    assert "More hidden content" not in cleaned
    assert "document.write" not in cleaned
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned
    assert "color: blue" not in cleaned
        
    # Test HTML output format
    result_html = await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        output_format='html'
    )
    cleaned_html = result_html["content"]
        
    # Verify HTML structure is preserved
    assert "<h1>" in cleaned_html
    assert "<p>" in cleaned_html
    assert "<div" in cleaned_html
    assert "<pre>" in cleaned_html
        
    # Verify unwanted elements are removed
    assert "<script>" not in cleaned_html
    assert "<style>" not in cleaned_html
    assert "document.write" not in cleaned_html
        
    # Verify whitespace handling
    assert "multiple    spaces" not in cleaned  # Multiple spaces should be normalized
    assert "multiple spaces" in cleaned
    
    # Verify preformatted text spacing is preserved
    pre_text = soup.find('pre').get_text()
    assert "    preserve spacing" in pre_text

    @pytest.mark.asyncio
    async def test_pandoc_conversion_failure(mock_httpx):
        """Test fallback when pandoc conversion fails."""
        # Setup mock response
        response_mock = AsyncMock()
        response_mock.status_code = 200
        response_mock.text = "<html><body><h1>Pandoc Test</h1></body></html>"
    
        client_mock = AsyncMock()
        client_mock.get.return_value = response_mock
        client_mock.get.side_effect = None  # Ensure no side effect
    
        context_mock = AsyncMock()
        context_mock.__aenter__.return_value = client_mock
        mock_httpx.return_value = context_mock
    
        with patch('pypandoc.convert_text') as mock_pandoc:
            mock_pandoc.side_effect = Exception("Pandoc error")
            
            result = await scrape_url_tool(
                url="https://example.com",
                use_playwright=False
            )
            
            assert isinstance(result, dict)
            assert "content" in result
            # Should still contain cleaned HTML content
            assert "Pandoc Test" in result["content"]
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from bs4 import BeautifulSoup
from sparc_cli.tools.scrape import RateLimiter, RateLimitConfig
from sparc_cli.tools.scrape import (
    scrape_url_tool,
    DEFAULT_HEADERS,
    RetryConfig,
    RetryStrategy,
    NetworkError,
    TransientError,
    PermanentError
)

@pytest.fixture(autouse=True)
def setup_rate_limiter():
    """Create fresh rate limiter between tests"""
    global rate_limiter
    limiter = RateLimiter(RateLimitConfig(max_concurrent=10))
    rate_limiter = limiter
    yield limiter
    with limiter._lock:
        limiter.concurrent_requests = 0
        limiter.tokens = limiter.config.burst_size
        limiter.domain_tokens.clear()

# Test HTML content
TEST_HTML = """
<html>
    <body>
        <div>
            <h1>Example Domain</h1>
            <p>This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.</p>
            <p><a href="https://www.iana.org/domains/example">More information...</a></p>
        </div>
    </body>
</html>
"""

TEST_MARKDOWN = "# Example Domain\n\nThis domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.\n\n[More information...](https://www.iana.org/domains/example)"


@pytest_asyncio.fixture
async def mock_httpx():
    with patch('httpx.AsyncClient') as mock:
        response_mock = AsyncMock()
        client_mock = AsyncMock()
        context_mock = AsyncMock()
        
        client_mock.get = AsyncMock(side_effect=NetworkError("Network error"))
        context_mock.__aenter__.return_value = client_mock
        mock.return_value = context_mock
        
        yield mock

@pytest_asyncio.fixture 
async def mock_playwright():
    with patch('playwright.sync_api.sync_playwright') as mock:
        browser_mock = MagicMock()
        context_mock = MagicMock()
        page_mock = MagicMock()
        
        mock.return_value.__enter__.return_value.chromium.launch = MagicMock(return_value=browser_mock)
        browser_mock.new_context = MagicMock(return_value=context_mock)
        context_mock.new_page = MagicMock(return_value=page_mock)
        
        yield mock

@pytest_asyncio.fixture
async def mock_pandoc():
    with patch('pypandoc.convert_text') as mock:
        mock.return_value = TEST_MARKDOWN
        yield mock

@pytest.mark.asyncio
async def test_successful_playwright_scrape(mock_pandoc):
    """Test successful scraping using Playwright."""
    with patch('playwright.sync_api.sync_playwright') as mock:
        browser_mock = MagicMock()
        context_mock = MagicMock()
        page_mock = MagicMock()
        
        mock.return_value.__enter__.return_value.chromium.launch = MagicMock(return_value=browser_mock)
        browser_mock.new_context = MagicMock(return_value=context_mock) 
        context_mock.new_page = MagicMock(return_value=page_mock)
        page_mock.content = MagicMock(return_value=TEST_HTML)
        
        result = await scrape_url_tool(
            url="https://example.com",
            use_playwright=True,
            verify_ssl=True
        )
        
        assert isinstance(result, dict)
        assert "content" in result
        assert "Example Domain" in result["content"]
        assert "This domain is for use in illustrative examples" in result["content"]
        assert "script" not in result["content"]
        assert "style" not in result["content"]

@pytest.mark.asyncio
async def test_successful_httpx_scrape(mock_httpx, mock_pandoc):
    """Test successful scraping using httpx."""
    # Setup mock response
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = TEST_HTML

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    client_mock.get.side_effect = None  # Ensure no side effect

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    result = await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        verify_ssl=True
    )
    
    assert isinstance(result, dict)
    assert "content" in result
    assert "Example Domain" in result["content"]

@pytest.mark.asyncio
async def test_failed_scrape_playwright_with_retry(mock_playwright):
    """Test handling of Playwright scraping failure with retry."""
    page_mock = AsyncMock()
    page_mock.content.return_value = TEST_HTML
    mock_playwright.return_value.__aenter__.return_value.new_page.side_effect = [
        Exception("Temporary error"),
        Exception("Temporary error"),
        page_mock  # Success on third try
    ]
    
    # First try - should succeed after retries
    result = await scrape_url_tool(
        url="https://example.com", 
        use_playwright=True,
        retry_config=RetryConfig(max_retries=3)
    )
    
    assert result["success"] is True
    assert "metrics" in result

@pytest.mark.asyncio
async def test_retry_strategy():
    """Test retry strategy behavior"""
    strategy = RetryStrategy(RetryConfig(max_retries=3))
    
    # Should retry on transient errors
    assert strategy.should_retry(0, TransientError()) is True
    assert strategy.should_retry(1, TransientError()) is True
    assert strategy.should_retry(2, TransientError()) is True
    assert strategy.should_retry(3, TransientError()) is False
    
    # Should not retry on permanent errors
    assert strategy.should_retry(0, PermanentError()) is False

@pytest.mark.asyncio
async def test_failed_scrape_httpx(mock_httpx):
    """Test handling of httpx scraping failure."""
    with pytest.raises(NetworkError) as exc_info:
        await scrape_url_tool(url="https://example.com", use_playwright=False)
    assert "Network error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_ssl_verification(mock_httpx):
    """Test SSL verification options."""
    # Setup mock response
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = "<html><body><h1>SSL Test</h1></body></html>"

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    client_mock.get.side_effect = None  # Ensure no side effect

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        verify_ssl=False
    )

    # Verify httpx was called with verify=False
    mock_httpx.assert_called_with(verify=False, headers=DEFAULT_HEADERS, follow_redirects=True)

@pytest.mark.asyncio
async def test_user_agent_handling(mock_httpx):
    """Test custom user agent handling."""
    # Setup async mocks
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = "<html><body><h1>Test Page</h1></body></html>"

    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    custom_ua = "Custom User Agent"
    expected_headers = DEFAULT_HEADERS.copy()
    expected_headers["User-Agent"] = custom_ua

    await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        user_agent=custom_ua
    )

    # Verify headers
    mock_httpx.assert_called_once_with(
        headers=expected_headers,
        verify=True, 
        follow_redirects=True
    )

@pytest.mark.asyncio
async def test_html_cleaning(mock_httpx):
    """Test HTML content cleaning functionality with different output formats."""
    html = """<!DOCTYPE html>
<html>
    <head>
        <title>Test Page</title>
        <style>
            /* CSS that should be removed */
            body { color: red; }
            div.important { display: none; }
        </style>
    </head>
    <body>
        <h1>Main Title</h1>
        <script>
            // JavaScript with nested content that should be removed
            document.write('<p>Hidden content</p>');
            const text = "This script text should be removed";
        </script>
        <div class="content">
            <p>First paragraph with  multiple    spaces</p>
            <script>console.log('<p>More hidden content</p>');</script>
            <p>Second paragraph</p>
            <style>.nested { color: blue; }</style>
            <div class="nested">
                <p>Nested content that should remain</p>
            </div>
        </div>
        <pre>
            Preformatted
                text should
                    preserve spacing
        </pre>
    </body>
</html>"""

    # Create response mock with proper context manager structure
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = html
    
    # Create client mock
    client_mock = AsyncMock()
    client_mock.get.return_value = response_mock
    
    # Setup context manager mock
    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = client_mock
    mock_httpx.return_value = context_mock

    result = await scrape_url_tool(url="https://example.com", use_playwright=False)
    cleaned = result["content"]
    soup = BeautifulSoup(cleaned, 'html.parser')
    
    # Verify removal of script and style tags
    assert soup.find('script') is None
    assert soup.find('style') is None
    
    # Verify content preservation
    assert "Main Title" in cleaned
    assert "First paragraph" in cleaned
    assert "Second paragraph" in cleaned
    assert "Nested content that should remain" in cleaned
    assert "Preformatted" in cleaned
    
    # Verify removal of script/style content
    assert "Hidden content" not in cleaned
    assert "More hidden content" not in cleaned
    assert "document.write" not in cleaned
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned
    assert "color: blue" not in cleaned
        
    # Test HTML output format
    result_html = await scrape_url_tool(
        url="https://example.com",
        use_playwright=False,
        output_format='html'
    )
    cleaned_html = result_html["content"]
        
    # Verify HTML structure is preserved
    assert "<h1>" in cleaned_html
    assert "<p>" in cleaned_html
    assert "<div" in cleaned_html
    assert "<pre>" in cleaned_html
        
    # Verify unwanted elements are removed
    assert "<script>" not in cleaned_html
    assert "<style>" not in cleaned_html
    assert "document.write" not in cleaned_html
        
    # Verify whitespace handling
    assert "multiple    spaces" not in cleaned  # Multiple spaces should be normalized
    assert "multiple spaces" in cleaned
    
    # Verify preformatted text spacing is preserved
    pre_text = soup.find('pre').get_text()
    assert "    preserve spacing" in pre_text

@pytest.mark.asyncio
async def test_pandoc_conversion_failure(mock_httpx):
    """Test fallback when pandoc conversion fails."""
    with patch('pypandoc.convert_text') as mock_pandoc:
        mock_pandoc.side_effect = Exception("Pandoc error")
        
        result = await scrape_url_tool(
            url="https://example.com",
            use_playwright=False
        )
        
        assert isinstance(result, dict)
        assert "content" in result
        # Should still contain cleaned HTML content
        assert "Test Title" in result["content"]

@pytest.mark.asyncio
async def test_invalid_url():
    """Test handling of invalid URLs."""
    with pytest.raises(ValueError) as exc_info:
        await scrape_url_tool(url="not-a-valid-url")
    assert "Invalid URL" in str(exc_info.value)
