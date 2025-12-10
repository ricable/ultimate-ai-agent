#!/usr/bin/env python3

import asyncio
from pathlib import Path
import sys
import os
import json
from typing import Optional
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import yaml
from dotenv import load_dotenv

# Load environment variables from the correct location
def load_environment(verbose: bool = False):
    """Load environment variables from .env file."""
    env_paths = [
        Path('/workspaces/auto-browser/.env'),  # Root directory
        Path.cwd() / '.env',  # Current directory
        Path.cwd() / 'auto-browser' / '.env',  # auto-browser subdirectory
    ]
    
    loaded = False
    for env_path in env_paths:
        if env_path.exists():
            if verbose:
                console.print(f"[blue]Loading environment from:[/blue] {env_path}")
            load_dotenv(env_path, override=True)  # Override existing variables
            loaded = True
    return loaded

from .config import load_config, create_example_config
from .browser import BrowserAutomation
from .template_generator import TemplateGenerator
from .processors.content import ContentProcessor
from .processors.interactive import InteractiveProcessor
from .processors.report import ReportGenerator
from .formatters.markdown import MarkdownFormatter

console = Console()

def create_progress() -> Progress:
    """Create a rich progress bar with custom formatting"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

@click.group()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default='config.yaml',
    help='Path to config file'
)
@click.pass_context
def cli(ctx, config):
    """Browser automation CLI tool for web interaction and data extraction"""
    ctx.ensure_object(dict)
    try:
        ctx.obj['config'] = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.argument('url')
@click.argument('prompt')
@click.option('--interactive', is_flag=True, help='Enable interactive mode')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--report', '-r', is_flag=True, help='Generate a structured report')
@click.option('--site', help='Site template to use')
@click.pass_context
def easy(ctx, url: str, prompt: str, interactive: bool, verbose: bool, report: bool, site: str):
    """Easy mode: Describe what you want to do with the webpage"""
    try:
        # Configure logging
        from . import configure_logging
        configure_logging(verbose)
        
        config = ctx.obj['config']
        site_config = None
        
        # Use site template if specified
        if site:
            if site not in config.sites:
                console.print(f"[red]Error:[/red] Site template '{site}' not found")
                ctx.exit(1)
            site_config = config.sites[site]
        
        # Run task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with create_progress() as progress:
                task = progress.add_task("Processing...", total=1)
                output_path = loop.run_until_complete(
                    run_all_tasks(
                        url=url,
                        prompt=prompt,
                        interactive=interactive,
                        verbose=verbose,
                        report=report,
                        site_config=site_config,
                        progress=progress,
                        task_id=task
                    )
                )
        finally:
            loop.close()
            
        console.print(f"[green]Success![/green] Output saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        ctx.exit(1)

@cli.command()
@click.argument('url')
@click.option('--name', prompt=True, help='Name for the template')
@click.option('--description', prompt=True, help='Description of what this template extracts')
@click.option('--config-path', type=str, default='config.yaml', help='Path to save the template')
@click.pass_context
def create_template(ctx, url: str, name: str, description: str, config_path: Path):
    """Create a new template by analyzing a webpage with AI assistance"""
    try:
        generator = TemplateGenerator()
        
        with create_progress() as progress:
            task = progress.add_task(f"Analyzing {url}...", total=1)
            template = asyncio.run(generator.create_template(url, name, description))
            progress.update(task, advance=1)
            
        # Preview the template
        console.print("\n[bold]Generated Template:[/bold]")
        console.print(f"Name: {template.name}")
        console.print(f"Description: {template.description}")
        console.print(f"URL Pattern: {template.url_pattern}")
        console.print("\nSelectors:")
        for name, selector in template.selectors.items():
            console.print(f"  - {name}:")
            console.print(f"    CSS: {selector.css}")
            if selector.description:
                console.print(f"    Description: {selector.description}")
            console.print(f"    Multiple: {selector.multiple}")
            
        # Confirm save
        if click.confirm("\nSave this template?", default=True):
            generator.save_template(template, config_path)
            console.print(f"\n[green]Template saved to {config_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error creating template:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.argument('output_path', type=click.Path(path_type=Path))
def init(output_path: Path):
    """Create an example configuration file"""
    try:
        create_example_config(output_path)
        console.print(f"[green]Created example config at:[/green] {output_path}")
    except Exception as e:
        console.print(f"[red]Error creating config:[/red] {e}")
        ctx.exit(1)

@cli.command()
@click.pass_context
def list_sites(ctx):
    """List available site templates"""
    config = ctx.obj['config']
    
    console.print("\n[bold]Available site templates:[/bold]\n")
    
    for name, site in config.sites.items():
        console.print(f"[blue]{name}[/blue]")
        if site.description:
            console.print(f"  Description: {site.description}")
        console.print(f"  URL Pattern: {site.url_pattern}")
        console.print(f"  Selectors:")
        for key, selector in site.selectors.items():
            if isinstance(selector, str):
                console.print(f"    - {key}: {selector}")
            else:
                console.print(f"    - {key}:")
                console.print(f"      CSS: {selector.css}")
                if selector.description:
                    console.print(f"      Description: {selector.description}")
        console.print("")

async def run_all_tasks(url: str, prompt: str, interactive: bool, verbose: bool, report: bool, 
                       site_config: Optional[dict] = None, progress: Optional[Progress] = None, 
                       task_id: Optional[int] = None):
    """Run all async tasks in sequence."""
    # Initialize processors
    content_processor = ContentProcessor()
    interactive_processor = InteractiveProcessor()
    formatter = MarkdownFormatter()
    
    if verbose:
        console.print("\n[blue]🔧 Initializing processors and configuration...[/blue]")
    
    # Create temporary config
    config = {
        'sites': {
            'temp': {
                'name': 'Temporary Site',
                'description': prompt,
                'url_pattern': url,
                'interactive': interactive
            }
        },
        'output_dir': 'output',
        'browser': {
            'headless': True,  # Always use headless mode
            'viewport': {'width': 1280, 'height': 720}
        }
    }
    
    # Use site template if provided
    if site_config:
        config['sites']['temp'].update(site_config)
    
    if verbose:
        console.print("\n[blue]🔍 Analyzing webpage structure...[/blue]")
    else:
        console.print("[blue]Analyzing webpage...[/blue]")
        
    # Generate template if not using existing one
    if not site_config:
        generator = TemplateGenerator()
        template = await generator.create_template(url, 'temp', prompt)
        
        if verbose:
            console.print("\n[yellow]📋 Template generated:[/yellow]")
            console.print(json.dumps({
                'name': template.name,
                'description': template.description,
                'url_pattern': template.url_pattern,
                'selectors': {name: sel.css for name, sel in template.selectors.items()}
            }, indent=2))
        
        # Update config with template
        config['sites']['temp'].update({
            'selectors': {name: sel.css for name, sel in template.selectors.items()},
            'actions': interactive_processor.analyze_prompt(prompt) if interactive else []
        })
    
    if verbose:
        console.print("\n[blue]🌐 Processing webpage...[/blue]")
        if interactive:
            console.print("[yellow]Interactive actions:[/yellow]")
            for action in config['sites']['temp'].get('actions', []):
                console.print(f"  - {action['action_type']}: {action.get('description', '')}")
    
    # Process webpage
    automation = BrowserAutomation(config['sites']['temp'], config['output_dir'])
    result = await automation.process_url(url)
    
    if progress and task_id is not None:
        progress.update(task_id, advance=0.5)
    
    if verbose:
        console.print("\n[yellow]📄 Raw extraction result:[/yellow]")
        console.print(json.dumps(result, indent=2))
    
    if verbose:
        console.print("\n[blue]📝 Formatting output...[/blue]")
    
    # Process the content
    analyzed = content_processor.analyze_page(result)
    
    if report:
        if verbose:
            console.print("\n[blue]📊 Generating structured report...[/blue]")
        # Generate structured report
        report_generator = ReportGenerator()
        markdown = report_generator.generate_report(analyzed, prompt)
    else:
        # Use standard formatting
        markdown = formatter.format_content(analyzed)
        
    output_path = formatter.save_markdown(markdown, url)
    
    if progress and task_id is not None:
        progress.update(task_id, advance=0.5)
    
    if verbose:
        console.print("\n[yellow]📊 Content analysis:[/yellow]")
        console.print(json.dumps(analyzed, indent=2))
        
    return output_path

def main():
    """CLI entry point"""
    try:
        # Load environment variables
        if not load_environment(verbose=False):
            console.print("[red]Error:[/red] No .env file found")
            sys.exit(1)
            
        # Check for required environment variables
        if not os.getenv('OPENAI_API_KEY'):
            console.print("[red]Error:[/red] OPENAI_API_KEY environment variable must be set")
            sys.exit(1)
            
        # Create output directory
        Path('output').mkdir(exist_ok=True)
            
        cli(obj={}, standalone_mode=False)
    except click.exceptions.Abort:
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
