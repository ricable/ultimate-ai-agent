/**
 * Compression Middleware for Proxy Responses
 * Provides 30-70% bandwidth reduction with Brotli/Gzip
 */

import { brotliCompress, gzip, constants } from 'zlib';
import { promisify } from 'util';
import { logger } from './logger.js';

const brotliCompressAsync = promisify(brotliCompress);
const gzipAsync = promisify(gzip);

export type CompressionEncoding = 'br' | 'gzip' | 'identity';

export interface CompressionConfig {
  minSize: number; // Minimum size to compress (bytes)
  level?: number; // Compression level (0-11 for Brotli, 0-9 for Gzip)
  preferredEncoding: CompressionEncoding;
  enableBrotli: boolean;
  enableGzip: boolean;
}

export interface CompressionResult {
  compressed: Buffer;
  encoding: CompressionEncoding;
  originalSize: number;
  compressedSize: number;
  ratio: number;
  duration: number;
}

export class CompressionMiddleware {
  private config: Required<CompressionConfig>;

  constructor(config: Partial<CompressionConfig> = {}) {
    this.config = {
      minSize: config.minSize || 1024, // 1KB minimum
      level: config.level ?? constants.BROTLI_DEFAULT_QUALITY,
      preferredEncoding: config.preferredEncoding || 'br',
      enableBrotli: config.enableBrotli ?? true,
      enableGzip: config.enableGzip ?? true
    };
  }

  /**
   * Compress data using best available algorithm
   */
  async compress(
    data: Buffer,
    acceptedEncodings?: string
  ): Promise<CompressionResult> {
    const startTime = Date.now();
    const originalSize = data.length;

    // Skip compression for small payloads
    if (originalSize < this.config.minSize) {
      return {
        compressed: data,
        encoding: 'identity',
        originalSize,
        compressedSize: originalSize,
        ratio: 1.0,
        duration: 0
      };
    }

    // Determine encoding based on accept-encoding header
    const encoding = this.selectEncoding(acceptedEncodings);

    let compressed: Buffer;
    try {
      switch (encoding) {
        case 'br':
          compressed = await this.compressBrotli(data);
          break;
        case 'gzip':
          compressed = await this.compressGzip(data);
          break;
        default:
          compressed = data;
      }
    } catch (error) {
      logger.error('Compression failed, using uncompressed', {
        error: (error as Error).message
      });
      compressed = data;
    }

    const duration = Date.now() - startTime;
    const compressedSize = compressed.length;
    const ratio = compressedSize / originalSize;

    logger.debug('Compression complete', {
      encoding,
      originalSize,
      compressedSize,
      ratio: `${(ratio * 100).toFixed(2)}%`,
      savings: `${((1 - ratio) * 100).toFixed(2)}%`,
      duration
    });

    return {
      compressed,
      encoding,
      originalSize,
      compressedSize,
      ratio,
      duration
    };
  }

  /**
   * Compress using Brotli (best compression ratio)
   */
  private async compressBrotli(data: Buffer): Promise<Buffer> {
    return brotliCompressAsync(data, {
      params: {
        [constants.BROTLI_PARAM_QUALITY]: this.config.level,
        [constants.BROTLI_PARAM_MODE]: constants.BROTLI_MODE_TEXT
      }
    });
  }

  /**
   * Compress using Gzip (faster, broader support)
   */
  private async compressGzip(data: Buffer): Promise<Buffer> {
    return gzipAsync(data, {
      level: Math.min(this.config.level, 9) // Gzip max level is 9
    });
  }

  /**
   * Select best encoding based on client support
   */
  private selectEncoding(acceptedEncodings?: string): CompressionEncoding {
    if (!acceptedEncodings) {
      return this.config.preferredEncoding === 'br' && this.config.enableBrotli
        ? 'br'
        : this.config.enableGzip
        ? 'gzip'
        : 'identity';
    }

    const encodings = acceptedEncodings.toLowerCase().split(',').map(e => e.trim());

    // Prefer Brotli if supported and enabled
    if (encodings.includes('br') && this.config.enableBrotli) {
      return 'br';
    }

    // Fall back to Gzip if supported and enabled
    if (encodings.includes('gzip') && this.config.enableGzip) {
      return 'gzip';
    }

    return 'identity';
  }

  /**
   * Check if compression is recommended for content type
   */
  shouldCompress(contentType?: string): boolean {
    if (!contentType) return true;

    const type = contentType.toLowerCase();

    // Compressible types
    const compressible = [
      'text/',
      'application/json',
      'application/javascript',
      'application/xml',
      'application/x-www-form-urlencoded'
    ];

    return compressible.some(prefix => type.includes(prefix));
  }

  /**
   * Get compression statistics
   */
  getStats(): {
    config: Required<CompressionConfig>;
    capabilities: {
      brotli: boolean;
      gzip: boolean;
    };
  } {
    return {
      config: this.config,
      capabilities: {
        brotli: this.config.enableBrotli,
        gzip: this.config.enableGzip
      }
    };
  }
}
