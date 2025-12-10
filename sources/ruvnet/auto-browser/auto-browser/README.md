# Browser Automation CLI

A command-line tool for configurable web scraping with AI-assisted template creation.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and configure your environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Creating Templates

Use the `create-template` command to create new site templates with AI assistance:

```bash
auto-browser create-template https://example.com/page
```

The tool will:
1. Visit the page and analyze its structure
2. Use AI to identify key elements and suggest selectors
3. Generate a URL pattern for similar pages
4. Preview the template for your review
5. Save the approved template to your config file

### Processing URLs

Process a single URL using a site template:

```bash
auto-browser process https://example.com/page --site template_name
```

Process multiple URLs from a file:

```bash
auto-browser batch urls.txt --site template_name
```

### Other Commands

- `list-sites`: List available site templates
- `init`: Create an example configuration file

## Configuration

Templates are stored in `config.yaml`. Each template defines:

- Name and description
- URL pattern for matching similar pages
- CSS selectors for extracting content
- Output format settings

Example template:
```yaml
sites:
  example_site:
    name: Example Site
    description: Extract content from example pages
    url_pattern: https://example.com/{id}
    selectors:
      title:
        css: h1.title
        description: Page title
      content:
        css: div.main-content
        multiple: true
        description: Main content sections
```

## Environment Variables

- `OPENAI_API_KEY`: API key for LLM functionality
- `LLM_MODEL`: Model to use (defaults to gpt-4o)
- `BROWSER_HEADLESS`: Run browser in headless mode (true/false)
- `BROWSER_DELAY`: Delay between page loads in seconds
