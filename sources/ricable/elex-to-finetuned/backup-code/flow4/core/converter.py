"""Unified document converter using Docling."""

import os
import tempfile
import platform
import json
import base64
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from bs4 import BeautifulSoup

from ..utils.config import DoclingConfig
from ..utils.logging import get_logger

# Try to import PyYAML for enhanced metadata
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = get_logger(__name__)

# Try to import docling with proper error handling
try:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter as DoclingConverter, PdfFormatOption
    from docling.datamodel.document import DoclingDocument as Document, Figure, TableItem as TableElement
    from docling.datamodel.base_models import Table, FigureElement
    HAS_DOCLING = True
except ImportError:
    logger.error("Docling not installed. Please install with: uv pip install docling")
    HAS_DOCLING = False
    # Fallback classes for type hints
    TableFormerMode = None
    Document = None
    Figure = None
    Table = None


class DocumentConverter:
    """Unified document converter using Docling with advanced features."""
    
    def __init__(self, config: Optional[DoclingConfig] = None):
        """Initialize the document converter.
        
        Args:
            config: Docling configuration options
        """
        if not HAS_DOCLING:
            raise ImportError("Docling is required for document conversion. Install with: uv pip install docling")
        
        self.config = config or DoclingConfig()
        self._converter = None
    
    def _get_converter(self) -> DoclingConverter:
        """Get or create the Docling converter instance."""
        if self._converter is None:
            self._converter = self._create_converter()
        return self._converter
    
    def _create_converter(self) -> DoclingConverter:
        """Create a new Docling converter with configuration."""
        # Set up accelerator options
        accelerator_device = self._get_accelerator_device()
        accelerator_options = AcceleratorOptions(
            num_threads=self.config.num_threads,
            device=accelerator_device
        )
        
        # Set up pipeline options with enhanced multimodal features
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = self.config.do_ocr
        pipeline_options.do_table_structure = self.config.extract_tables
        
        # Enhanced table processing
        if pipeline_options.do_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = self.config.table_cell_matching
            # Set table processing mode
            if self.config.table_mode == "fast":
                pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            else:
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        # Enhanced multimodal processing
        if self.config.multimodal:
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = self.config.image_scale_factor
            pipeline_options.do_picture_classification = self.config.image_classification
            
            # Enable VLM if available and requested
            if self.config.enable_vlm:
                try:
                    pipeline_options.do_vlm = True
                except AttributeError:
                    logger.warning("VLM support not available in this Docling version")
        
        # Set artifacts path for model caching
        if self.config.artifacts_path:
            pipeline_options.artifacts_path = self.config.artifacts_path
        
        # Initialize DocumentConverter with proper options
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        
        converter = DoclingConverter(
            format_options=format_options
        )
        
        # Enable profiling if in debug mode
        settings.debug.profile_pipeline_timings = True
        
        return converter
    
    def _get_accelerator_device(self) -> AcceleratorDevice:
        """Determine the best accelerator device for the current platform."""
        if not self.config.with_accelerator:
            return AcceleratorDevice.CPU
        
        # Detect platform and use appropriate accelerator
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Apple Silicon Mac - use MPS
            logger.info("Using MPS acceleration for Apple Silicon")
            return AcceleratorDevice.MPS
        elif platform.system() == 'Linux':
            # Try CUDA for Linux
            logger.info("Using CUDA acceleration for Linux")
            return AcceleratorDevice.CUDA
        else:
            logger.info("Using auto-detected acceleration")
            return AcceleratorDevice.AUTO
    
    def _should_skip_file(self, html_content: str) -> bool:
        """Check if the HTML file should be skipped based on content analysis."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Console logging for debugging
        print(f"[DEBUG] Analyzing HTML file for skip conditions...")
        
        # Check document type for debugging (but don't skip based on MOM status)
        docname_meta = soup.find('meta', attrs={'name': 'DOCNAME'})
        if docname_meta and docname_meta.get('content'):
            docname = docname_meta.get('content').strip().upper()
            print(f"[DEBUG] Found document type: '{docname}'")
        
        # Check if title exists and indicates pure code documentation (not MOM)
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip().lower()
            print(f"[DEBUG] Found title: '{soup.title.string.strip()}'")
            
            # Skip various code documentation patterns, but not MOM-specific classes
            skip_patterns = ["class", "enum", "struct", "derivedDataType", "deriveddatatype", "module", "interface", "function ", "method ", "namespace "]
            for pattern in skip_patterns:
                if title_text.startswith(pattern.lower()):
                    print(f"[SKIP] Skipping code documentation with title: {title_text}")
                    logger.info(f"Skipping code documentation with title: {title_text}")
                    return True
        
        # Check for specific anchor patterns
        # Looking for <a name="TITLE">derivedDataType&#160;RuleDataType</a> or similar
        title_anchors = soup.find_all('a', attrs={'name': 'TITLE'})
        for anchor in title_anchors:
            if anchor.get_text():
                anchor_text = anchor.get_text().strip()
                anchor_text_lower = anchor_text.lower()
                print(f"[DEBUG] Found title anchor: '{anchor_text}'")
                
                # Check for all skip patterns in anchor text
                skip_patterns = ["class", "enum", "struct", "derivedDataType", "deriveddatatype", "module", "interface"]
                for pattern in skip_patterns:
                    if pattern.lower() in anchor_text_lower:
                        print(f"[SKIP] Skipping {pattern} documentation: {anchor_text}")
                        logger.info(f"Skipping {pattern} documentation: {anchor_text}")
                        return True
        
        # Also check meta tags with name="TITLE"
        title_meta = soup.find('meta', attrs={'name': 'TITLE'})
        if title_meta and title_meta.get('content'):
            content_text = title_meta.get('content').strip()
            content_text_lower = content_text.lower()
            print(f"[DEBUG] Found meta title: '{content_text}'")
            
            # Check for all skip patterns in meta title
            skip_patterns = ["class", "enum", "struct", "derivedDataType", "deriveddatatype", "module", "interface"]
            for pattern in skip_patterns:
                if pattern.lower() in content_text_lower:
                    print(f"[SKIP] Skipping {pattern} meta documentation: {content_text}")
                    logger.info(f"Skipping {pattern} meta documentation: {content_text}")
                    return True
        
        print(f"[DEBUG] File passed skip checks - will be processed")
        return False
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content by removing footer elements and legal information."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove footer-info divs (contains legal links and copyright)
        for footer_div in soup.find_all('div', class_='footer-info'):
            logger.debug(f"Removing footer div: {footer_div.get('class')}")
            footer_div.decompose()
        
        # Remove copyright tables
        for table in soup.find_all('table', id='copyright_table'):
            logger.debug(f"Removing copyright table with id: {table.get('id')}")
            table.decompose()
        
        # Remove other footer-related tables
        for table in soup.find_all('table', class_='copytbl'):
            logger.debug(f"Removing footer table with class: {table.get('class')}")
            table.decompose()
        
        # Remove script tags
        for script in soup.find_all('script'):
            logger.debug("Removing script tag")
            script.decompose()
        
        # Remove other common footer elements
        for element in soup.find_all(['footer', 'div'], class_=['footer', 'page-footer', 'document-footer']):
            logger.debug(f"Removing footer element: {element.name} with class {element.get('class')}")
            element.decompose()
        
        # Remove elements with footer-related IDs
        footer_ids = ['footer', 'page-footer', 'document-footer', 'legal-footer']
        for footer_id in footer_ids:
            element = soup.find(id=footer_id)
            if element:
                logger.debug(f"Removing element with footer ID: {footer_id}")
                element.decompose()
        
        return str(soup)
    
    def convert_file(self, file_path: str, output_path: str) -> Optional[str]:
        """Convert a single document to Markdown.
        
        Args:
            file_path: Path to the input document
            output_path: Path to save the output Markdown file
            
        Returns:
            Path to the output file if successful, None otherwise
        """
        print(f"\n[CONVERT] Starting conversion: {os.path.basename(file_path)}")
        logger.info(f"Converting {file_path} to {output_path}")
        
        try:
            cleaned_file_path = file_path
            temp_file_path = None
            
            # Handle HTML files - clean content and check if we should skip
            if file_path.lower().endswith('.html'):
                print(f"[HTML] Processing HTML file: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                print(f"[HTML] Read {len(html_content)} characters from HTML file")
                
                # Check if we should skip this file
                if self._should_skip_file(html_content):
                    print(f"[SKIP] ✗ Skipped {os.path.basename(file_path)} - identified as code documentation")
                    logger.info(f"Skipping {file_path} - identified as code documentation")
                    return None
                
                print(f"[HTML] ✓ File passed skip checks, proceeding with cleaning...")
                
                # Clean footer content from HTML
                print(f"[CLEAN] Cleaning HTML content...")
                cleaned_content = self._clean_html_content(html_content)
                print(f"[CLEAN] Cleaned content: {len(cleaned_content)} characters (was {len(html_content)})")
                
                # Save cleaned content to temporary file for processing
                temp_fd, temp_file_path = tempfile.mkstemp(suffix='.html', prefix='flow4_cleaned_')
                try:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                        temp_file.write(cleaned_content)
                    cleaned_file_path = temp_file_path
                    print(f"[TEMP] Created temporary cleaned file: {os.path.basename(temp_file_path)}")
                    logger.debug(f"Created cleaned temporary file: {temp_file_path}")
                except Exception as e:
                    os.close(temp_fd)
                    logger.warning(f"Failed to create temporary cleaned file: {e}")
                    # Fall back to original file if cleaning fails
                    cleaned_file_path = file_path
            
            # Convert document using Docling
            print(f"[DOCLING] Converting with Docling...")
            converter = self._get_converter()
            conversion_result = converter.convert(cleaned_file_path)
            document = conversion_result.document
            print(f"[DOCLING] ✓ Docling conversion completed")
            
            # Create output directory for extracted content if needed
            base_output_dir = os.path.dirname(output_path)
            file_stem = Path(file_path).stem
            extracted_dir = os.path.join(base_output_dir, 'extracted', file_stem)
            
            # Handle tables with enhanced processing
            tables_exported = []
            table_metadata = []
            if self.config.extract_tables:
                print(f"[TABLES] Extracting tables with enhanced processing...")
                tables_dir = os.path.join(extracted_dir, 'tables')
                os.makedirs(tables_dir, exist_ok=True)
                
                table_data = self._extract_enhanced_tables(document, tables_dir)
                tables_exported = table_data['exported_files']
                table_metadata = table_data['metadata']
            
            # Handle figures with enhanced multimodal processing
            figures_exported = []
            figure_metadata = []
            if self.config.extract_figures:
                print(f"[FIGURES] Extracting figures with multimodal processing...")
                figures_dir = os.path.join(extracted_dir, 'figures')
                os.makedirs(figures_dir, exist_ok=True)
                
                figure_data = self._extract_enhanced_figures(document, figures_dir)
                figures_exported = figure_data['exported_files']
                figure_metadata = figure_data['metadata']
            
            # Export to markdown
            print(f"[MARKDOWN] Exporting to markdown...")
            markdown_content = document.export_to_markdown()
            print(f"[MARKDOWN] Generated {len(markdown_content)} characters of markdown")
            
            # Add file information as metadata
            filename = os.path.basename(file_path)
            file_stem = Path(filename).stem
            
            # Generate multimodal training data if requested
            multimodal_data = None
            if self.config.generate_multimodal_data:
                print(f"[MULTIMODAL] Generating multimodal training data...")
                multimodal_data = self._generate_multimodal_training_data(
                    document, table_metadata, figure_metadata, markdown_content
                )
                
                # Save multimodal data
                multimodal_path = os.path.join(extracted_dir, f'{file_stem}_multimodal.{self.config.multimodal_export_format}')
                with open(multimodal_path, 'w', encoding='utf-8') as f:
                    if self.config.multimodal_export_format == 'jsonl':
                        for item in multimodal_data:
                            f.write(json.dumps(item) + '\n')
                    else:
                        json.dump(multimodal_data, f, indent=2)
                print(f"[MULTIMODAL] ✓ Saved multimodal training data: {multimodal_path}")
            
            # Enhanced YAML frontmatter metadata
            enhanced_metadata = {
                'title': file_stem,
                'source': filename,
                'converted_with': 'Flow4-Docling-Enhanced',
                'features': {
                    'accelerator': self.config.with_accelerator,
                    'tables': self.config.extract_tables,
                    'figures': self.config.extract_figures,
                    'multimodal': self.config.multimodal,
                    'custom_rules': self.config.custom_convert,
                    'vlm_enabled': self.config.enable_vlm,
                    'table_mode': self.config.table_mode,
                    'image_classification': self.config.image_classification
                },
                'extraction_stats': {
                    'tables_found': len(table_metadata),
                    'figures_found': len(figure_metadata),
                    'tables_exported': len(tables_exported),
                    'figures_exported': len(figures_exported)
                }
            }
            
            if table_metadata:
                enhanced_metadata['table_metadata'] = table_metadata[:3]  # Limit metadata size
            if figure_metadata:
                enhanced_metadata['figure_metadata'] = figure_metadata[:3]  # Limit metadata size
            
            # Convert metadata to YAML format
            if HAS_YAML:
                try:
                    yaml_metadata = yaml.dump(enhanced_metadata, default_flow_style=False)
                except Exception as e:
                    logger.warning(f"Error converting metadata to YAML: {e}")
                    yaml_metadata = json.dumps(enhanced_metadata, indent=2)
            else:
                # Fallback to JSON format if PyYAML not available
                yaml_metadata = json.dumps(enhanced_metadata, indent=2)
            
            markdown_with_metadata = f"""---
{yaml_metadata}---

# {file_stem}

{markdown_content}
"""
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write markdown to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_with_metadata)
            
            # Log multimodal extraction summary
            if tables_exported or figures_exported:
                print(f"[MULTIMODAL] 📊 Extracted {len(tables_exported)} tables and {len(figures_exported)} figures")
                logger.info(f"Multimodal extraction: {len(tables_exported)} tables, {len(figures_exported)} figures")
            
            print(f"[SUCCESS] ✓ Conversion completed: {os.path.basename(output_path)}")
            logger.info(f"Conversion completed: {output_path}")
            
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
            
            return output_path
        
        except Exception as e:
            print(f"[ERROR] ✗ Failed to convert {os.path.basename(file_path)}: {str(e)}")
            logger.error(f"Error converting {file_path}: {str(e)}")
            
            # Clean up temporary file if created, even on error
            if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file after error: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
            
            return None
    
    def batch_convert(
        self, 
        input_files: List[str], 
        output_dir: str, 
        num_workers: int = 4
    ) -> Tuple[List[str], List[str]]:
        """Convert multiple files in parallel.
        
        Args:
            input_files: List of input file paths
            output_dir: Directory to save output files
            num_workers: Number of parallel workers
            
        Returns:
            Tuple of (successfully converted files, skipped files)
        """
        if not HAS_DOCLING:
            raise ImportError("Docling is required for document conversion.")
        
        print(f"\n[BATCH] Starting batch conversion of {len(input_files)} files with {num_workers} workers")
        print(f"[BATCH] Output directory: {output_dir}")
        logger.info(f"Starting batch conversion of {len(input_files)} files")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        skipped = []
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for input_file in input_files:
                file_ext = os.path.splitext(input_file)[1].lower()
                output_path = os.path.join(
                    output_dir, 
                    os.path.basename(input_file).replace(file_ext, '.md')
                )
                futures.append(executor.submit(self.convert_file, input_file, output_path))
            
            # Process results
            processed_count = 0
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    processed_count += 1
                    if result:
                        results.append(result)
                        print(f"[BATCH] [{processed_count}/{len(futures)}] ✓ Converted: {os.path.basename(input_files[i])}")
                    else:
                        skipped.append(input_files[i])
                        print(f"[BATCH] [{processed_count}/{len(futures)}] ✗ Skipped: {os.path.basename(input_files[i])}")
                except Exception as e:
                    print(f"[BATCH] [{processed_count}/{len(futures)}] ✗ Error: {os.path.basename(input_files[i])} - {str(e)}")
                    logger.error(f"Error processing file {input_files[i]}: {e}")
                    skipped.append(input_files[i])
        
        print(f"\n[BATCH] ✓ Batch conversion completed: {len(results)} converted, {len(skipped)} skipped")
        logger.info(f"Batch conversion completed: {len(results)} converted, {len(skipped)} skipped")
        return results, skipped
    
    def _extract_enhanced_tables(self, document: 'Document', tables_dir: str) -> Dict[str, List]:
        """Extract tables with enhanced metadata and multiple export formats."""
        exported_files = []
        metadata = []
        
        for i, page in enumerate(document.pages):
            if not hasattr(page, 'tables'):
                logger.warning(f"Page {i+1} does not have tables attribute, skipping table extraction")
                continue
            
            try:
                for j, table in enumerate(page.tables):
                    table_info = {
                        'page': i + 1,
                        'table_index': j + 1,
                        'bbox': getattr(table, 'bbox', None),
                        'caption': self._extract_table_caption(table, page) if self.config.match_table_captions else None,
                        'structure_confidence': getattr(table, 'confidence', None),
                        'exported_formats': []
                    }
                    
                    # Export in multiple formats
                    for export_format in self.config.table_export_formats:
                        try:
                            if export_format == 'csv':
                                table_path = os.path.join(tables_dir, f'page_{i+1}_table_{j+1}.csv')
                                table_df = table.export_to_dataframe()
                                table_df.to_csv(table_path, index=False)
                                exported_files.append(table_path)
                                table_info['exported_formats'].append({'format': 'csv', 'path': table_path})
                                
                            elif export_format == 'json':
                                table_path = os.path.join(tables_dir, f'page_{i+1}_table_{j+1}.json')
                                table_data = self._table_to_structured_data(table)
                                with open(table_path, 'w', encoding='utf-8') as f:
                                    json.dump(table_data, f, indent=2)
                                exported_files.append(table_path)
                                table_info['exported_formats'].append({'format': 'json', 'path': table_path})
                                
                            elif export_format == 'html':
                                table_path = os.path.join(tables_dir, f'page_{i+1}_table_{j+1}.html')
                                table_html = self._table_to_html(table)
                                with open(table_path, 'w', encoding='utf-8') as f:
                                    f.write(table_html)
                                exported_files.append(table_path)
                                table_info['exported_formats'].append({'format': 'html', 'path': table_path})
                                
                            print(f"[TABLES] ✓ Exported table {j+1} from page {i+1} as {export_format}")
                            logger.info(f"Exported table {j+1} from page {i+1} to {table_path}")
                            
                        except Exception as e:
                            logger.warning(f"Could not export table {j+1} from page {i+1} as {export_format}: {str(e)}")
                    
                    metadata.append(table_info)
                    
            except Exception as e:
                logger.warning(f"Error accessing tables on page {i+1}: {str(e)}")
        
        return {'exported_files': exported_files, 'metadata': metadata}
    
    def _extract_enhanced_figures(self, document: 'Document', figures_dir: str) -> Dict[str, List]:
        """Extract figures with enhanced metadata and descriptions."""
        exported_files = []
        metadata = []
        
        for i, page in enumerate(document.pages):
            if not hasattr(page, 'figures'):
                logger.warning(f"Page {i+1} does not have figures attribute, skipping figure extraction")
                continue
            
            try:
                for j, figure in enumerate(page.figures):
                    figure_info = {
                        'page': i + 1,
                        'figure_index': j + 1,
                        'bbox': getattr(figure, 'bbox', None),
                        'caption': self._extract_figure_caption(figure, page) if self.config.match_figure_captions else None,
                        'classification': getattr(figure, 'classification', None) if self.config.image_classification else None,
                        'description': None,
                        'exported_formats': []
                    }
                    
                    # Export image in specified formats
                    for export_format in self.config.image_export_formats:
                        try:
                            if export_format == 'png' and hasattr(figure, 'image') and figure.image:
                                figure_path = os.path.join(figures_dir, f'page_{i+1}_figure_{j+1}.png')
                                with open(figure_path, 'wb') as f:
                                    f.write(figure.image)
                                exported_files.append(figure_path)
                                figure_info['exported_formats'].append({'format': 'png', 'path': figure_path})
                                
                                # Generate description if VLM enabled
                                if self.config.generate_image_descriptions and self.config.enable_vlm:
                                    try:
                                        figure_info['description'] = self._generate_image_description(figure)
                                    except Exception as e:
                                        logger.warning(f"Could not generate description for figure {j+1} from page {i+1}: {str(e)}")
                                
                            elif export_format == 'json':
                                figure_metadata_path = os.path.join(figures_dir, f'page_{i+1}_figure_{j+1}_metadata.json')
                                figure_metadata = {
                                    'bbox': figure_info['bbox'],
                                    'caption': figure_info['caption'],
                                    'classification': figure_info['classification'],
                                    'description': figure_info['description']
                                }
                                with open(figure_metadata_path, 'w', encoding='utf-8') as f:
                                    json.dump(figure_metadata, f, indent=2)
                                exported_files.append(figure_metadata_path)
                                figure_info['exported_formats'].append({'format': 'json', 'path': figure_metadata_path})
                            
                            print(f"[FIGURES] ✓ Exported figure {j+1} from page {i+1} as {export_format}")
                            logger.info(f"Exported figure {j+1} from page {i+1}")
                            
                        except Exception as e:
                            logger.warning(f"Could not export figure {j+1} from page {i+1} as {export_format}: {str(e)}")
                    
                    metadata.append(figure_info)
                    
            except Exception as e:
                logger.warning(f"Error accessing figures on page {i+1}: {str(e)}")
        
        return {'exported_files': exported_files, 'metadata': metadata}
    
    def _extract_table_caption(self, table: 'Table', page) -> Optional[str]:
        """Extract caption for a table using spatial analysis."""
        try:
            # This is a simplified approach - in practice, you'd use more sophisticated
            # spatial analysis to match captions with tables
            if hasattr(table, 'caption'):
                return table.caption
            
            # Look for nearby text elements that might be captions
            table_bbox = getattr(table, 'bbox', None)
            if table_bbox and hasattr(page, 'texts'):
                for text_element in page.texts:
                    # Simple heuristic: look for text near the table
                    if 'table' in text_element.text.lower() or 'figure' in text_element.text.lower():
                        return text_element.text
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting table caption: {str(e)}")
            return None
    
    def _extract_figure_caption(self, figure: 'Figure', page) -> Optional[str]:
        """Extract caption for a figure using spatial analysis."""
        try:
            if hasattr(figure, 'caption'):
                return figure.caption
            
            # Look for nearby text elements that might be captions
            figure_bbox = getattr(figure, 'bbox', None)
            if figure_bbox and hasattr(page, 'texts'):
                for text_element in page.texts:
                    if 'figure' in text_element.text.lower() or 'fig.' in text_element.text.lower():
                        return text_element.text
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting figure caption: {str(e)}")
            return None
    
    def _table_to_structured_data(self, table: 'Table') -> Dict[str, Any]:
        """Convert table to structured JSON format."""
        try:
            # Export as DataFrame first, then convert to structured format
            df = table.export_to_dataframe()
            
            return {
                'headers': df.columns.tolist(),
                'rows': df.values.tolist(),
                'shape': df.shape,
                'structure': {
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'has_header': True
                }
            }
        except Exception as e:
            logger.warning(f"Error converting table to structured data: {str(e)}")
            return {'error': str(e)}
    
    def _table_to_html(self, table: 'Table') -> str:
        """Convert table to HTML format."""
        try:
            df = table.export_to_dataframe()
            return df.to_html(index=False, escape=False)
        except Exception as e:
            logger.warning(f"Error converting table to HTML: {str(e)}")
            return f"<p>Error converting table: {str(e)}</p>"
    
    def _generate_image_description(self, figure: 'Figure') -> Optional[str]:
        """Generate description for an image using VLM."""
        try:
            # This would integrate with the VLM pipeline in Docling
            # For now, return a placeholder that could be enhanced with actual VLM integration
            if hasattr(figure, 'description'):
                return figure.description
            
            # Placeholder for VLM integration
            classification = getattr(figure, 'classification', 'unknown')
            return f"Image classified as: {classification}"
            
        except Exception as e:
            logger.warning(f"Error generating image description: {str(e)}")
            return None
    
    def _generate_multimodal_training_data(
        self, 
        document: 'Document', 
        table_metadata: List[Dict], 
        figure_metadata: List[Dict],
        markdown_content: str
    ) -> List[Dict[str, Any]]:
        """Generate structured data for multimodal fine-tuning."""
        training_data = []
        
        try:
            # Create entries for each table
            for table_info in table_metadata:
                for export_info in table_info['exported_formats']:
                    if export_info['format'] == 'json':
                        entry = {
                            'type': 'table',
                            'page': table_info['page'],
                            'index': table_info['table_index'],
                            'text': table_info.get('caption', ''),
                            'data_path': export_info['path'],
                            'bbox': table_info.get('bbox'),
                            'context': self._extract_context_around_element(document, table_info)
                        }
                        training_data.append(entry)
            
            # Create entries for each figure
            for figure_info in figure_metadata:
                for export_info in figure_info['exported_formats']:
                    if export_info['format'] == 'png':
                        entry = {
                            'type': 'figure',
                            'page': figure_info['page'],
                            'index': figure_info['figure_index'],
                            'text': figure_info.get('caption', ''),
                            'description': figure_info.get('description', ''),
                            'image_path': export_info['path'],
                            'classification': figure_info.get('classification'),
                            'bbox': figure_info.get('bbox'),
                            'context': self._extract_context_around_element(document, figure_info)
                        }
                        
                        # Add base64 encoding if requested
                        if self.config.include_vision_embeddings:
                            try:
                                with open(export_info['path'], 'rb') as img_file:
                                    img_data = img_file.read()
                                    entry['image_base64'] = base64.b64encode(img_data).decode('utf-8')
                            except Exception as e:
                                logger.warning(f"Could not encode image to base64: {str(e)}")
                        
                        training_data.append(entry)
            
            # Add document-level entry
            document_entry = {
                'type': 'document',
                'title': getattr(document, 'title', 'Untitled'),
                'content': markdown_content[:1000],  # Truncated content
                'full_content_length': len(markdown_content),
                'num_tables': len(table_metadata),
                'num_figures': len(figure_metadata),
                'pages': len(document.pages) if hasattr(document, 'pages') else 0
            }
            training_data.append(document_entry)
            
        except Exception as e:
            logger.error(f"Error generating multimodal training data: {str(e)}")
        
        return training_data
    
    def _extract_context_around_element(self, document: 'Document', element_info: Dict) -> str:
        """Extract text context around a table or figure element."""
        try:
            # This is a simplified version - in practice, you'd use more sophisticated
            # spatial analysis to extract relevant context
            page_num = element_info['page'] - 1
            if hasattr(document, 'pages') and page_num < len(document.pages):
                page = document.pages[page_num]
                if hasattr(page, 'texts'):
                    # Extract some text from the page as context
                    texts = [text.text for text in page.texts[:5]]  # First 5 text elements
                    return ' '.join(texts)[:500]  # Limit context length
            
            return ''
        except Exception as e:
            logger.warning(f"Error extracting context: {str(e)}")
            return ''