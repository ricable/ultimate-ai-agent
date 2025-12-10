"""Document conversion using IBM Docling with simple HTML fallback."""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logging import get_logger
from ..utils.config import DoclingConfig

# Import docling with fallback
try:
    from docling.document_converter import DocumentConverter as DoclingConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    DoclingConverter = None
    InputFormat = None
    PdfPipelineOptions = None
    PdfFormatOption = None

# Import simple converter fallback
from .simple_converter import SimpleHTMLConverter
from .cleaner import HTMLCleaner

logger = get_logger(__name__)


class DocumentConverter:
    """Document converter using IBM Docling for HTML and PDF processing."""
    
    def __init__(self, config: DoclingConfig, disable_filtering: bool = False):
        """Initialize the document converter.
        
        Args:
            config: Docling configuration
            disable_filtering: Whether to disable HTML content filtering
        """
        self.config = config
        self.disable_filtering = disable_filtering
        self.converter = None
        self.simple_converter = None
        self.html_cleaner = HTMLCleaner()
        
        if HAS_DOCLING:
            try:
                self._init_docling_converter()
            except Exception as e:
                logger.error(f"Failed to initialize Docling converter: {e}")
                self.converter = None
        
        if not self.converter:
            logger.warning("Docling not available, using simple HTML converter fallback")
            self.simple_converter = SimpleHTMLConverter(config, disable_filtering)
            if not self.simple_converter.is_available():
                logger.error("Simple converter also not available. Install beautifulsoup4 with: pip install beautifulsoup4")
    
    def _init_docling_converter(self) -> None:
        """Initialize the Docling converter with configuration."""
        if not HAS_DOCLING:
            return
        
        try:
            # Configure PDF pipeline options
            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = self.config.do_ocr
            pdf_options.do_table_structure = self.config.extract_tables
            pdf_options.table_structure_options.do_cell_matching = getattr(
                self.config, 'table_cell_matching', True
            )
            
            # Configure format options
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
            
            # Initialize converter
            self.converter = DoclingConverter(
                format_options=format_options,
                parallel_workers=self.config.num_threads
            )
            
            logger.info("Docling converter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            self.converter = None
    
    def convert_file(self, file_path: str, output_dir: str) -> Optional[str]:
        """Convert a single file to Markdown.
        
        Args:
            file_path: Path to input file
            output_dir: Output directory for converted file
            
        Returns:
            Path to converted Markdown file or None if failed
        """
        # Check if HTML file should be skipped (simple filtering)
        if not self.disable_filtering and file_path.lower().endswith('.html'):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                if self.html_cleaner.should_skip_file(html_content):
                    logger.info(f"Skipping file based on content filter: {os.path.basename(file_path)}")
                    return None
            except Exception as e:
                logger.warning(f"Could not perform content filtering on {file_path}: {e}")
        
        # Use Docling if available, otherwise fall back to simple converter
        if self.converter:
            return self._convert_with_docling(file_path, output_dir)
        elif self.simple_converter and self.simple_converter.is_available():
            return self.simple_converter.convert_file(file_path, output_dir)
        else:
            logger.error("No converter available")
            return None
    
    def _convert_with_docling(self, file_path: str, output_dir: str) -> Optional[str]:
        """Convert file using Docling converter.
        
        Args:
            file_path: Path to input file
            output_dir: Output directory for converted file
            
        Returns:
            Path to converted Markdown file or None if failed
        """
        try:
            input_path = Path(file_path)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = output_path / f"{input_path.stem}.md"
            
            logger.info(f"Converting {input_path.name} to Markdown...")
            
            # Convert document
            result = self.converter.convert(file_path)
            
            if not result:
                logger.error(f"Conversion failed for {file_path}")
                return None
            
            # Extract markdown content
            markdown_content = result.document.export_to_markdown()
            
            # Save markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Extract and save tables if enabled
            if self.config.extract_tables and hasattr(result.document, 'tables'):
                self._extract_tables(result.document, output_path, input_path.stem)
            
            # Extract and save figures if enabled
            if self.config.extract_figures and hasattr(result.document, 'pictures'):
                self._extract_figures(result.document, output_path, input_path.stem)
            
            logger.info(f"Successfully converted {input_path.name}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")
            return None
    
    def _extract_tables(self, document, output_dir: Path, base_name: str) -> None:
        """Extract tables from document.
        
        Args:
            document: Docling document object
            output_dir: Output directory
            base_name: Base filename
        """
        try:
            tables_dir = output_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            
            for i, table in enumerate(document.tables):
                # Save table data
                table_file = tables_dir / f"{base_name}_table_{i+1}.csv"
                
                # Export table to CSV if possible
                if hasattr(table, 'export_to_csv'):
                    table_data = table.export_to_csv()
                    with open(table_file, 'w', encoding='utf-8') as f:
                        f.write(table_data)
                else:
                    # Fallback: save table metadata
                    table_meta = {
                        "table_id": i + 1,
                        "rows": getattr(table, 'num_rows', 0),
                        "cols": getattr(table, 'num_cols', 0),
                        "content": str(table)
                    }
                    with open(table_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
                        json.dump(table_meta, f, indent=2)
                
                logger.debug(f"Extracted table {i+1} from {base_name}")
                
        except Exception as e:
            logger.warning(f"Failed to extract tables from {base_name}: {e}")
    
    def _extract_figures(self, document, output_dir: Path, base_name: str) -> None:
        """Extract figures from document.
        
        Args:
            document: Docling document object
            output_dir: Output directory
            base_name: Base filename
        """
        try:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for i, figure in enumerate(document.pictures):
                # Save figure metadata
                figure_meta = {
                    "figure_id": i + 1,
                    "filename": f"{base_name}_figure_{i+1}",
                    "type": getattr(figure, 'type', 'unknown'),
                    "caption": getattr(figure, 'caption', ''),
                    "page": getattr(figure, 'page', 0)
                }
                
                meta_file = figures_dir / f"{base_name}_figure_{i+1}.json"
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(figure_meta, f, indent=2)
                
                # Try to save actual image data if available
                if hasattr(figure, 'image') and figure.image:
                    img_file = figures_dir / f"{base_name}_figure_{i+1}.png"
                    try:
                        with open(img_file, 'wb') as f:
                            f.write(figure.image)
                    except Exception as e:
                        logger.warning(f"Failed to save image data for figure {i+1}: {e}")
                
                logger.debug(f"Extracted figure {i+1} from {base_name}")
                
        except Exception as e:
            logger.warning(f"Failed to extract figures from {base_name}: {e}")
    
    def batch_convert(
        self, 
        file_paths: List[str], 
        output_dir: str, 
        num_workers: int = 4
    ) -> Tuple[List[str], List[str]]:
        """Convert multiple files in parallel.
        
        Args:
            file_paths: List of input file paths
            output_dir: Output directory
            num_workers: Number of parallel workers
            
        Returns:
            Tuple of (successful_files, failed_files)
        """
        # Use appropriate converter
        if self.converter:
            return self._batch_convert_with_docling(file_paths, output_dir, num_workers)
        elif self.simple_converter and self.simple_converter.is_available():
            return self.simple_converter.batch_convert(file_paths, output_dir, num_workers)
        else:
            logger.error("No converter available")
            return [], file_paths
    
    def _batch_convert_with_docling(
        self, 
        file_paths: List[str], 
        output_dir: str, 
        num_workers: int = 4
    ) -> Tuple[List[str], List[str]]:
        """Convert multiple files using Docling in parallel.
        
        Args:
            file_paths: List of input file paths
            output_dir: Output directory
            num_workers: Number of parallel workers
            
        Returns:
            Tuple of (successful_files, failed_files)
        """
        successful_files = []
        failed_files = []
        
        logger.info(f"Converting {len(file_paths)} files using {num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all conversion tasks
            future_to_file = {
                executor.submit(self._convert_with_docling, file_path, output_dir): file_path
                for file_path in file_paths
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful_files.append(result)
                        logger.info(f"✓ Converted: {os.path.basename(file_path)}")
                    else:
                        failed_files.append(file_path)
                        logger.warning(f"✗ Failed: {os.path.basename(file_path)}")
                except Exception as e:
                    failed_files.append(file_path)
                    logger.error(f"✗ Error converting {os.path.basename(file_path)}: {e}")
        
        logger.info(f"Conversion complete: {len(successful_files)} successful, {len(failed_files)} failed")
        return successful_files, failed_files
    
    def is_available(self) -> bool:
        """Check if the converter is available and functional.
        
        Returns:
            True if converter is available, False otherwise
        """
        return (HAS_DOCLING and self.converter is not None) or (self.simple_converter and self.simple_converter.is_available())