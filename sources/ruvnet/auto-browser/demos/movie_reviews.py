import asyncio
import subprocess
import sys
from pathlib import Path
import shutil

async def process_movies(output_dir):
    """Extract movie reviews from Rotten Tomatoes."""
    
    # Create a copy of download_trial.py as a base and modify it for our needs
    base_script = Path('../../browser-use-main/examples/download_trial.py').read_text()
    
    # Modify the script for movie reviews
    new_script = base_script.replace(
        '''async def extract_files(browser: BrowserContext):''',
        '''async def extract_reviews(browser: BrowserContext):'''
    ).replace(
        '''task=\'\'\'
        1. Navigate to https://euclinicaltrials.eu/ctis-public/view/2023-509462-38-00
        2. Click the "Download clinical trial" button
        3. Use the extract_files action to extract the downloaded zip file
        
        Note: After downloading, you must use the extract_files action to handle the zip file properly.
        \'\'\'''',
        '''task=\'\'\'
        1. Navigate to https://www.rottentomatoes.com/browse/movies_in_theaters/
        2. Wait for movie elements to load
        3. Extract movie information and save to markdown
        \'\'\'''',
    ).replace(
        'extract_files',  # Replace action name in controller.action decorator
        'extract_reviews'
    ).replace(
        '''async def extract_files(browser: BrowserContext):
    try:
        page = await browser.get_current_page()
        
        # Configure download behavior
        client = await page.context.new_cdp_session(page)
        await client.send('Page.setDownloadBehavior', {
            'behavior': 'allow',
            'downloadPath': str(OUTPUT_DIR)
        })''',
        '''async def extract_reviews(browser: BrowserContext):
    try:
        page = await browser.get_current_page()
        
        # Wait for movie elements to load
        await page.wait_for_selector('.media-list__item')
        
        # Extract movie information
        movies = await page.evaluate("""
            Array.from(document.querySelectorAll('.media-list__item'))
                .slice(0, 5)
                .map(movie => ({
                    title: movie.querySelector('h2.p--small')?.textContent.trim(),
                    score: movie.querySelector('score-board__tomatometer')?.textContent.trim() || 'No score yet',
                    audience: movie.querySelector('score-board__audience-score')?.textContent.trim() || 'No audience score yet'
                }))
        """)
        
        # Format as markdown
        markdown = "# Latest Movies in Theaters\\n\\n"
        for movie in movies:
            markdown += f"## {movie['title']}\\n"
            markdown += f"- Tomatometer: {movie['score']}\\n"
            markdown += f"- Audience Score: {movie['audience']}\\n\\n"
        
        # Save to file
        output_file = OUTPUT_DIR / "movie_reviews.markdown"  # Use .markdown extension to match other demos
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
            
        return ActionResult(extracted_content=f"Saved reviews to {output_file}")'''
    )
    
    # Write the modified script
    temp_script = Path('temp_movie_reviews.py')
    temp_script.write_text(new_script)
    
    print("\nExtracting movie reviews...")
    try:
        # Run the script
        subprocess.run([sys.executable, str(temp_script)], check=True)
        
        # Move files from browser-use-main/examples/output to our output directory
        example_output = Path('../../browser-use-main/examples/output')
        if example_output.exists():
            for file in example_output.glob('*'):
                shutil.move(str(file), str(output_dir / file.name))
            
            # Clean up example output directory
            shutil.rmtree(example_output)
            
            # Print final markdown file location
            markdown_file = output_dir / "movie_reviews.markdown"
            if markdown_file.exists():
                print(f"\nMovie reviews saved to: {markdown_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error extracting reviews: {e}")
    finally:
        # Clean up
        temp_script.unlink()

async def main():
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    await process_movies(output_dir)

if __name__ == '__main__':
    asyncio.run(main())
