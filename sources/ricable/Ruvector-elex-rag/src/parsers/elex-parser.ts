/**
 * ELEX HTML Documentation Parser
 *
 * Parses Ericsson ELEX documentation from HTML files with embedded images.
 * Handles extraction of:
 * - Document structure (sections, headings)
 * - Text content
 * - Embedded images (base64)
 * - Tables
 * - Code blocks
 * - Metadata
 */

import * as cheerio from 'cheerio';
import AdmZip from 'adm-zip';
import { promises as fs } from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import type {
  ELEXDocument,
  DocumentSection,
  ExtractedTable,
  DocumentChunk,
} from '../core/types.js';
import { logger } from '../utils/logger.js';

export interface ELEXParserOptions {
  /** Maximum chunk size in characters */
  maxChunkSize: number;
  /** Chunk overlap in characters */
  chunkOverlap: number;
  /** Extract embedded images */
  extractImages: boolean;
  /** Process tables as structured data */
  processTables: boolean;
  /** Preserve formatting */
  preserveFormatting: boolean;
}

const DEFAULT_OPTIONS: ELEXParserOptions = {
  maxChunkSize: 2000,
  chunkOverlap: 200,
  extractImages: true,
  processTables: true,
  preserveFormatting: true,
};

export class ELEXParser {
  private options: ELEXParserOptions;

  constructor(options: Partial<ELEXParserOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Process a ZIP file containing ELEX HTML documentation
   */
  async processZipFile(zipPath: string): Promise<ELEXDocument[]> {
    logger.info('Processing ELEX ZIP file', { zipPath });

    const zip = new AdmZip(zipPath);
    const entries = zip.getEntries();
    const documents: ELEXDocument[] = [];

    for (const entry of entries) {
      if (entry.entryName.endsWith('.html') || entry.entryName.endsWith('.htm')) {
        try {
          const htmlContent = entry.getData().toString('utf8');
          const doc = await this.parseHTMLDocument(htmlContent, entry.entryName, zipPath);
          documents.push(doc);
          logger.info('Parsed ELEX document', {
            documentId: doc.id,
            title: doc.title,
            sections: doc.sections.length
          });
        } catch (error) {
          logger.error('Failed to parse ELEX document', {
            entryName: entry.entryName,
            error: (error as Error).message
          });
        }
      }
    }

    logger.info('Completed processing ELEX ZIP', {
      zipPath,
      documentsProcessed: documents.length
    });

    return documents;
  }

  /**
   * Process multiple ZIP files
   */
  async processMultipleZips(zipPaths: string[]): Promise<ELEXDocument[]> {
    const allDocuments: ELEXDocument[] = [];

    for (const zipPath of zipPaths) {
      const docs = await this.processZipFile(zipPath);
      allDocuments.push(...docs);
    }

    return allDocuments;
  }

  /**
   * Parse a single HTML document
   */
  async parseHTMLDocument(
    htmlContent: string,
    sourcePath: string,
    zipSource?: string
  ): Promise<ELEXDocument> {
    const $ = cheerio.load(htmlContent);
    const documentId = uuidv4();

    // Extract metadata
    const metadata = this.extractMetadata($);
    const title = this.extractTitle($, metadata);
    const version = metadata['version'] || metadata['doc-version'] || '1.0';

    // Extract images
    const images = this.options.extractImages ? this.extractImages($, documentId) : [];

    // Extract document structure
    const sections = this.extractSections($);

    // Extract full text content
    const textContent = this.extractTextContent($);

    // Determine category from path or metadata
    const category = this.determineCategory(sourcePath, metadata);

    return {
      id: documentId,
      title,
      version,
      sourcePath,
      category,
      rawHtml: htmlContent,
      textContent,
      images,
      sections,
      metadata,
      processedAt: new Date(),
    };
  }

  /**
   * Extract metadata from HTML head and body
   */
  private extractMetadata($: cheerio.CheerioAPI): Record<string, string> {
    const metadata: Record<string, string> = {};

    // Extract from meta tags
    $('meta').each((_, element) => {
      const name = $(element).attr('name') || $(element).attr('property');
      const content = $(element).attr('content');
      if (name && content) {
        metadata[name.toLowerCase().replace(/\s+/g, '-')] = content;
      }
    });

    // Extract from data attributes in body
    const body = $('body');
    const dataAttrs = body.data() as Record<string, string>;
    for (const [key, value] of Object.entries(dataAttrs)) {
      if (typeof value === 'string') {
        metadata[key.toLowerCase()] = value;
      }
    }

    // Look for Ericsson-specific metadata patterns
    const patterns = [
      { selector: '.document-number', key: 'document-number' },
      { selector: '.revision', key: 'revision' },
      { selector: '.product-name', key: 'product' },
      { selector: '.release', key: 'release' },
      { selector: '[data-doc-id]', key: 'doc-id', attr: 'data-doc-id' },
    ];

    for (const pattern of patterns) {
      const element = $(pattern.selector);
      if (element.length > 0) {
        const value = pattern.attr ? element.attr(pattern.attr) : element.text().trim();
        if (value) {
          metadata[pattern.key] = value;
        }
      }
    }

    return metadata;
  }

  /**
   * Extract document title
   */
  private extractTitle($: cheerio.CheerioAPI, metadata: Record<string, string>): string {
    // Try various sources for title
    const titleSources = [
      () => $('title').text().trim(),
      () => $('h1').first().text().trim(),
      () => metadata['title'],
      () => metadata['dc.title'],
      () => $('.document-title').text().trim(),
      () => $('[data-title]').attr('data-title'),
    ];

    for (const source of titleSources) {
      const title = source();
      if (title && title.length > 0) {
        return title;
      }
    }

    return 'Untitled Document';
  }

  /**
   * Extract embedded images
   */
  private extractImages(
    $: cheerio.CheerioAPI,
    documentId: string
  ): ELEXDocument['images'] {
    const images: ELEXDocument['images'] = [];

    $('img').each((index, element) => {
      const src = $(element).attr('src');
      const alt = $(element).attr('alt') || '';

      if (src) {
        // Handle base64 encoded images
        if (src.startsWith('data:')) {
          const match = src.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            images.push({
              id: `${documentId}-img-${index}`,
              alt,
              mimeType: match[1],
              data: match[2],
            });
          }
        }
        // Handle relative paths (store reference)
        else {
          images.push({
            id: `${documentId}-img-${index}`,
            alt,
            mimeType: this.guessMimeType(src),
            data: src, // Store path for later resolution
          });
        }
      }
    });

    return images;
  }

  /**
   * Extract document sections hierarchically
   */
  private extractSections($: cheerio.CheerioAPI): DocumentSection[] {
    const sections: DocumentSection[] = [];
    const sectionStack: { section: DocumentSection; level: number }[] = [];

    // Find all heading elements
    $('h1, h2, h3, h4, h5, h6').each((_, element) => {
      const $el = $(element);
      const tagName = element.tagName.toLowerCase();
      const level = parseInt(tagName[1], 10);
      const title = $el.text().trim();
      const sectionId = $el.attr('id') || uuidv4();

      // Get content until next heading
      const content = this.extractSectionContent($, $el);

      // Extract tables within section
      const tables = this.options.processTables
        ? this.extractTablesInSection($, $el)
        : [];

      // Extract code blocks
      const codeBlocks = this.extractCodeBlocksInSection($, $el);

      const section: DocumentSection = {
        id: sectionId,
        title,
        level,
        content,
        childIds: [],
        tables,
        codeBlocks,
      };

      // Build hierarchy
      while (sectionStack.length > 0 && sectionStack[sectionStack.length - 1].level >= level) {
        sectionStack.pop();
      }

      if (sectionStack.length > 0) {
        const parent = sectionStack[sectionStack.length - 1].section;
        section.parentId = parent.id;
        parent.childIds.push(sectionId);
      }

      sectionStack.push({ section, level });
      sections.push(section);
    });

    return sections;
  }

  /**
   * Extract content from a section until the next heading
   */
  private extractSectionContent($: cheerio.CheerioAPI, $heading: cheerio.Cheerio<any>): string {
    const content: string[] = [];
    let $current = $heading.next();

    while ($current.length > 0 && !$current.is('h1, h2, h3, h4, h5, h6')) {
      const text = $current.text().trim();
      if (text) {
        content.push(text);
      }
      $current = $current.next();
    }

    return content.join('\n\n');
  }

  /**
   * Extract tables within a section
   */
  private extractTablesInSection(
    $: cheerio.CheerioAPI,
    $heading: cheerio.Cheerio<any>
  ): ExtractedTable[] {
    const tables: ExtractedTable[] = [];
    let $current = $heading.next();

    while ($current.length > 0 && !$current.is('h1, h2, h3, h4, h5, h6')) {
      if ($current.is('table')) {
        tables.push(this.parseTable($, $current));
      }
      // Also check for tables nested in divs
      $current.find('table').each((_, table) => {
        tables.push(this.parseTable($, $(table)));
      });
      $current = $current.next();
    }

    return tables;
  }

  /**
   * Parse a single table
   */
  private parseTable($: cheerio.CheerioAPI, $table: cheerio.Cheerio<any>): ExtractedTable {
    const caption = $table.find('caption').text().trim() || undefined;
    const headers: string[] = [];
    const rows: string[][] = [];

    // Extract headers
    $table.find('thead th, thead td, tr:first-child th').each((_, th) => {
      headers.push($(th).text().trim());
    });

    // If no explicit headers, use first row
    if (headers.length === 0) {
      $table.find('tr:first-child td').each((_, td) => {
        headers.push($(td).text().trim());
      });
    }

    // Extract rows
    $table.find('tbody tr, tr').each((rowIndex, tr) => {
      // Skip header row if we extracted headers from it
      if (rowIndex === 0 && headers.length > 0) {
        const firstRowCells = $(tr).find('th, td');
        let isHeader = true;
        firstRowCells.each((i, cell) => {
          if ($(cell).text().trim() !== headers[i]) {
            isHeader = false;
          }
        });
        if (isHeader) return;
      }

      const row: string[] = [];
      $(tr).find('td, th').each((_, cell) => {
        row.push($(cell).text().trim());
      });
      if (row.length > 0) {
        rows.push(row);
      }
    });

    return {
      id: uuidv4(),
      caption,
      headers,
      rows,
    };
  }

  /**
   * Extract code blocks within a section
   */
  private extractCodeBlocksInSection(
    $: cheerio.CheerioAPI,
    $heading: cheerio.Cheerio<any>
  ): string[] {
    const codeBlocks: string[] = [];
    let $current = $heading.next();

    while ($current.length > 0 && !$current.is('h1, h2, h3, h4, h5, h6')) {
      if ($current.is('pre, code')) {
        codeBlocks.push($current.text().trim());
      }
      $current.find('pre, code').each((_, code) => {
        codeBlocks.push($(code).text().trim());
      });
      $current = $current.next();
    }

    return codeBlocks;
  }

  /**
   * Extract full text content from document
   */
  private extractTextContent($: cheerio.CheerioAPI): string {
    // Remove script and style elements
    $('script, style, noscript').remove();

    // Get text with some structure preservation
    const textParts: string[] = [];

    $('body').find('*').each((_, element) => {
      const $el = $(element);
      const tagName = element.tagName.toLowerCase();

      // Handle block elements
      if (['p', 'div', 'section', 'article', 'li', 'td', 'th'].includes(tagName)) {
        const text = $el.clone().children().remove().end().text().trim();
        if (text) {
          textParts.push(text);
        }
      }
      // Handle headings
      else if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
        const text = $el.text().trim();
        if (text) {
          textParts.push(`\n${'#'.repeat(parseInt(tagName[1]))} ${text}\n`);
        }
      }
    });

    return textParts.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  }

  /**
   * Determine document category from path or metadata
   */
  private determineCategory(sourcePath: string, metadata: Record<string, string>): string {
    // Check metadata first
    if (metadata['category']) return metadata['category'];
    if (metadata['doc-type']) return metadata['doc-type'];

    // Infer from path
    const pathLower = sourcePath.toLowerCase();

    const categoryMappings: [string, string][] = [
      ['radio', 'Radio Access Network'],
      ['baseband', 'Baseband Processing'],
      ['enm', 'Network Manager'],
      ['parameter', 'Parameter Reference'],
      ['troubleshoot', 'Troubleshooting'],
      ['install', 'Installation'],
      ['config', 'Configuration'],
      ['upgrade', 'Upgrade'],
      ['kpi', 'KPI Reference'],
      ['counter', 'Counter Reference'],
      ['alarm', 'Alarm Reference'],
      ['api', 'API Reference'],
      ['mom', 'Managed Object Model'],
      ['nr', '5G New Radio'],
      ['lte', 'LTE'],
      ['power', 'Power Control'],
      ['antenna', 'Antenna System'],
    ];

    for (const [keyword, category] of categoryMappings) {
      if (pathLower.includes(keyword)) {
        return category;
      }
    }

    return 'General Documentation';
  }

  /**
   * Guess MIME type from file extension
   */
  private guessMimeType(filename: string): string {
    const ext = path.extname(filename).toLowerCase();
    const mimeTypes: Record<string, string> = {
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml',
      '.webp': 'image/webp',
    };
    return mimeTypes[ext] || 'image/png';
  }

  /**
   * Convert ELEX document to chunks for RAG
   */
  chunkDocument(document: ELEXDocument): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    let chunkIndex = 0;

    // Chunk each section
    for (const section of document.sections) {
      const sectionChunks = this.chunkText(
        section.content,
        section.title,
        document,
        section
      );

      for (const chunkContent of sectionChunks) {
        chunks.push({
          id: `${document.id}-chunk-${chunkIndex}`,
          documentId: document.id,
          documentType: 'elex',
          content: chunkContent,
          metadata: {
            title: document.title,
            section: section.title,
            chunkIndex,
            totalChunks: 0, // Will be updated later
            sourceFile: document.sourcePath,
          },
          tokenCount: this.estimateTokens(chunkContent),
        });
        chunkIndex++;
      }

      // Add table content as separate chunks
      for (const table of section.tables) {
        const tableContent = this.tableToText(table);
        chunks.push({
          id: `${document.id}-chunk-${chunkIndex}`,
          documentId: document.id,
          documentType: 'elex',
          content: tableContent,
          metadata: {
            title: document.title,
            section: section.title,
            chunkIndex,
            totalChunks: 0,
            sourceFile: document.sourcePath,
          },
          tokenCount: this.estimateTokens(tableContent),
        });
        chunkIndex++;
      }
    }

    // Update total chunks count
    for (const chunk of chunks) {
      chunk.metadata.totalChunks = chunks.length;
    }

    return chunks;
  }

  /**
   * Split text into chunks with overlap
   */
  private chunkText(
    text: string,
    sectionTitle: string,
    document: ELEXDocument,
    section: DocumentSection
  ): string[] {
    if (!text || text.length <= this.options.maxChunkSize) {
      return text ? [`## ${sectionTitle}\n\n${text}`] : [];
    }

    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
      let end = start + this.options.maxChunkSize;

      // Try to break at sentence boundary
      if (end < text.length) {
        const lastPeriod = text.lastIndexOf('.', end);
        const lastNewline = text.lastIndexOf('\n', end);
        const breakPoint = Math.max(lastPeriod, lastNewline);

        if (breakPoint > start + this.options.maxChunkSize / 2) {
          end = breakPoint + 1;
        }
      }

      const chunkText = text.slice(start, end).trim();
      if (chunkText) {
        chunks.push(`## ${sectionTitle}\n\n${chunkText}`);
      }

      start = end - this.options.chunkOverlap;
    }

    return chunks;
  }

  /**
   * Convert table to text format
   */
  private tableToText(table: ExtractedTable): string {
    const lines: string[] = [];

    if (table.caption) {
      lines.push(`Table: ${table.caption}\n`);
    }

    // Add headers
    if (table.headers.length > 0) {
      lines.push(`| ${table.headers.join(' | ')} |`);
      lines.push(`| ${table.headers.map(() => '---').join(' | ')} |`);
    }

    // Add rows
    for (const row of table.rows) {
      lines.push(`| ${row.join(' | ')} |`);
    }

    return lines.join('\n');
  }

  /**
   * Estimate token count (rough approximation)
   */
  private estimateTokens(text: string): number {
    // Rough estimate: ~4 characters per token
    return Math.ceil(text.length / 4);
  }
}

export default ELEXParser;
