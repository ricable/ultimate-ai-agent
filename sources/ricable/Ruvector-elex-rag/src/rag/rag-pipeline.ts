/**
 * Self-Learning RAG Pipeline for Ericsson RAN Documentation
 *
 * Implements a complete Retrieval-Augmented Generation pipeline with:
 * - Multi-modal document support (ELEX HTML, 3GPP XML)
 * - Self-learning relevance adaptation
 * - Uncertainty-aware response generation
 * - Parameter-specific knowledge retrieval
 */

import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { v4 as uuidv4 } from 'uuid';
import type {
  DocumentChunk,
  RAGQuery,
  RAGResult,
  ELEXDocument,
  ThreeGPPMOM,
  MOMAttribute,
} from '../core/types.js';
import { getConfig } from '../core/config.js';
import { logger, logRAGQuery } from '../utils/logger.js';
import { SelfLearningVectorStore } from './vector-store.js';
import ELEXParser from '../parsers/elex-parser.js';
import ThreeGPPParser from '../parsers/threegpp-parser.js';

export interface RAGPipelineConfig {
  /** Use OpenAI for embeddings */
  embeddingProvider: 'openai' | 'anthropic' | 'local';
  /** Use for completions */
  completionProvider: 'openai' | 'anthropic';
  /** Temperature for generation */
  temperature: number;
  /** Maximum tokens in response */
  maxTokens: number;
  /** Enable self-learning */
  selfLearning: boolean;
  /** Include confidence scores */
  includeConfidence: boolean;
  /** System prompt for RAG */
  systemPrompt: string;
}

const DEFAULT_RAG_CONFIG: RAGPipelineConfig = {
  embeddingProvider: 'openai',
  completionProvider: 'openai',
  temperature: 0.3,
  maxTokens: 2000,
  selfLearning: true,
  includeConfidence: true,
  systemPrompt: `You are an expert Ericsson RAN (Radio Access Network) technical assistant specializing in LTE and 5G NR networks.

Your knowledge comes from official Ericsson ELEX documentation and 3GPP MOM (Managed Object Model) specifications.

When answering questions:
1. Be precise and technical, using proper 3GPP terminology
2. Reference specific parameters (e.g., pZeroNominalPusch, alpha) when relevant
3. Explain the relationship between parameters and network performance
4. Consider the Tuning Paradox when discussing power control optimization
5. Provide parameter ranges and default values when available
6. Cite the source documents when possible

Key concepts you understand deeply:
- Uplink Power Control (P0, alpha, fractional path loss compensation)
- The Tuning Paradox in RAN optimization
- Coverage vs. Capacity trade-offs
- Inter-cell interference management
- 3GPP TS 36.213 (LTE) and TS 38.213 (NR) specifications`,
};

/**
 * Self-Learning RAG Pipeline
 */
export class RAGPipeline {
  private config: RAGPipelineConfig;
  private systemConfig = getConfig();
  private vectorStore: SelfLearningVectorStore;
  private openai: OpenAI | null = null;
  private anthropic: Anthropic | null = null;
  private elexParser: ELEXParser;
  private threeGPPParser: ThreeGPPParser;
  private moms: ThreeGPPMOM[] = [];
  private initialized = false;

  constructor(config: Partial<RAGPipelineConfig> = {}) {
    this.config = { ...DEFAULT_RAG_CONFIG, ...config };
    this.vectorStore = new SelfLearningVectorStore({
      storagePath: this.systemConfig.vectorDb.storagePath,
      dimensions: this.systemConfig.llm.embeddingDimensions,
      selfLearning: this.config.selfLearning,
    });
    this.elexParser = new ELEXParser();
    this.threeGPPParser = new ThreeGPPParser();
  }

  /**
   * Initialize the RAG pipeline
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Initialize LLM clients
    if (this.systemConfig.llm.openaiApiKey) {
      this.openai = new OpenAI({
        apiKey: this.systemConfig.llm.openaiApiKey,
      });
    }

    if (this.systemConfig.llm.anthropicApiKey) {
      this.anthropic = new Anthropic({
        apiKey: this.systemConfig.llm.anthropicApiKey,
      });
    }

    // Initialize vector store
    await this.vectorStore.initialize();

    this.initialized = true;
    logger.info('RAG pipeline initialized');
  }

  /**
   * Ingest ELEX documentation from ZIP files
   */
  async ingestELEX(zipPaths: string[]): Promise<number> {
    await this.initialize();

    let totalChunks = 0;

    for (const zipPath of zipPaths) {
      try {
        const documents = await this.elexParser.processZipFile(zipPath);

        for (const doc of documents) {
          const chunks = this.elexParser.chunkDocument(doc);

          // Generate embeddings
          const chunksWithEmbeddings = await this.embedChunks(chunks);

          // Add to vector store
          await this.vectorStore.addChunks(chunksWithEmbeddings);
          totalChunks += chunksWithEmbeddings.length;

          logger.info('Ingested ELEX document', {
            documentId: doc.id,
            title: doc.title,
            chunks: chunksWithEmbeddings.length,
          });
        }
      } catch (error) {
        logger.error('Failed to ingest ELEX ZIP', {
          zipPath,
          error: (error as Error).message,
        });
      }
    }

    return totalChunks;
  }

  /**
   * Ingest 3GPP MOM XML files
   */
  async ingest3GPP(dirPath: string): Promise<number> {
    await this.initialize();

    let totalChunks = 0;

    try {
      const moms = await this.threeGPPParser.parseDirectory(dirPath);
      this.moms = moms;

      for (const mom of moms) {
        const chunks = this.threeGPPParser.chunkMOM(mom);

        // Generate embeddings
        const chunksWithEmbeddings = await this.embedChunks(chunks);

        // Add to vector store
        await this.vectorStore.addChunks(chunksWithEmbeddings);
        totalChunks += chunksWithEmbeddings.length;

        logger.info('Ingested 3GPP MOM', {
          momId: mom.id,
          name: mom.name,
          technology: mom.technology,
          classes: mom.classes.size,
          chunks: chunksWithEmbeddings.length,
        });
      }
    } catch (error) {
      logger.error('Failed to ingest 3GPP directory', {
        dirPath,
        error: (error as Error).message,
      });
    }

    return totalChunks;
  }

  /**
   * Generate embeddings for chunks
   */
  private async embedChunks(chunks: DocumentChunk[]): Promise<DocumentChunk[]> {
    const batchSize = 100;
    const result: DocumentChunk[] = [];

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const texts = batch.map((c) => c.content);

      const embeddings = await this.getEmbeddings(texts);

      for (let j = 0; j < batch.length; j++) {
        result.push({
          ...batch[j],
          embedding: embeddings[j],
        });
      }
    }

    return result;
  }

  /**
   * Get embeddings from provider
   */
  private async getEmbeddings(texts: string[]): Promise<Float32Array[]> {
    if (this.config.embeddingProvider === 'openai' && this.openai) {
      const response = await this.openai.embeddings.create({
        model: this.systemConfig.llm.embeddingModel,
        input: texts,
      });

      return response.data.map((d) => new Float32Array(d.embedding));
    }

    // Fallback to simple embedding (for testing without API)
    logger.warn('Using fallback embedding generation');
    return texts.map(() => {
      const embedding = new Float32Array(this.systemConfig.llm.embeddingDimensions);
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] = Math.random() - 0.5;
      }
      return embedding;
    });
  }

  /**
   * Query the RAG system
   */
  async query(queryText: string, options: Partial<RAGQuery> = {}): Promise<RAGResult> {
    await this.initialize();

    const startTime = Date.now();
    const queryId = uuidv4();

    const query: RAGQuery = {
      query: queryText,
      topK: options.topK ?? 10,
      minSimilarity: options.minSimilarity ?? 0.5,
      documentTypes: options.documentTypes,
      technologies: options.technologies,
      parameterNames: options.parameterNames,
      includeMetadata: options.includeMetadata ?? true,
    };

    // Get query embedding
    const [queryEmbedding] = await this.getEmbeddings([queryText]);

    // Search vector store
    const { chunks, scores } = await this.vectorStore.search(queryEmbedding, query);

    // Check if query is about a specific parameter
    const parameterInfo = this.findParameterInfo(queryText);

    // Generate response
    const answer = await this.generateResponse(queryText, chunks, parameterInfo);

    // Calculate confidence
    const confidence = this.calculateConfidence(scores, chunks.length);

    const processingTime = Date.now() - startTime;

    logRAGQuery(queryId, queryText, chunks.length, processingTime);

    return {
      chunks,
      scores,
      answer,
      confidence,
      sources: [...new Set(chunks.map((c) => c.metadata.sourceFile))],
      processingTime,
    };
  }

  /**
   * Find parameter information from MOMs
   */
  private findParameterInfo(query: string): MOMAttribute | null {
    const parameterPatterns = [
      /p[_\s]?zero[_\s]?nominal/i,
      /pZeroNominalPusch/i,
      /alpha/i,
      /pucch/i,
      /pusch/i,
    ];

    for (const pattern of parameterPatterns) {
      const match = query.match(pattern);
      if (match) {
        const paramName = match[0].replace(/[_\s]/g, '');
        return this.threeGPPParser.findParameter(this.moms, paramName);
      }
    }

    return null;
  }

  /**
   * Generate response using LLM
   */
  private async generateResponse(
    query: string,
    chunks: DocumentChunk[],
    parameterInfo: MOMAttribute | null
  ): Promise<string> {
    // Build context from chunks
    const context = chunks
      .map((c, i) => `[Source ${i + 1}: ${c.metadata.sourceFile}]\n${c.content}`)
      .join('\n\n---\n\n');

    // Add parameter info if available
    let parameterContext = '';
    if (parameterInfo) {
      parameterContext = `\n\n[Parameter Reference]\n${this.formatParameterInfo(parameterInfo)}`;
    }

    const userMessage = `Context from Ericsson documentation:

${context}${parameterContext}

User Question: ${query}

Please provide a detailed, technically accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge what is known and what is uncertain.`;

    if (this.config.completionProvider === 'anthropic' && this.anthropic) {
      const response = await this.anthropic.messages.create({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: this.config.maxTokens,
        system: this.config.systemPrompt,
        messages: [{ role: 'user', content: userMessage }],
      });

      return response.content[0].type === 'text' ? response.content[0].text : '';
    }

    if (this.openai) {
      const response = await this.openai.chat.completions.create({
        model: this.systemConfig.llm.completionModel,
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
        messages: [
          { role: 'system', content: this.config.systemPrompt },
          { role: 'user', content: userMessage },
        ],
      });

      return response.choices[0]?.message?.content || '';
    }

    // Fallback response
    return `Based on the retrieved documentation:\n\n${chunks.slice(0, 3).map((c) => c.content).join('\n\n')}`;
  }

  /**
   * Format parameter info for context
   */
  private formatParameterInfo(param: MOMAttribute): string {
    const lines = [
      `Parameter: ${param.name}`,
      `Type: ${param.type}`,
    ];

    if (param.defaultValue) {
      lines.push(`Default: ${param.defaultValue}`);
    }

    if (param.range) {
      if (param.range.min !== undefined || param.range.max !== undefined) {
        lines.push(`Range: [${param.range.min ?? ''}..${param.range.max ?? ''}]`);
      }
      if (param.range.enum) {
        lines.push(`Allowed Values: ${param.range.enum.join(', ')}`);
      }
    }

    if (param.reference) {
      lines.push(`3GPP Reference: ${param.reference}`);
    }

    if (param.description) {
      lines.push(`Description: ${param.description}`);
    }

    return lines.join('\n');
  }

  /**
   * Calculate confidence score
   */
  private calculateConfidence(scores: number[], resultCount: number): number {
    if (scores.length === 0) return 0;

    // Average similarity of top results
    const avgSimilarity = scores.reduce((a, b) => a + b, 0) / scores.length;

    // Penalize if few results
    const coverageFactor = Math.min(resultCount / 5, 1);

    // Penalize high variance in scores (inconsistent results)
    const mean = avgSimilarity;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const consistencyFactor = 1 - Math.min(Math.sqrt(variance), 0.5);

    return avgSimilarity * coverageFactor * consistencyFactor;
  }

  /**
   * Provide feedback for self-learning
   */
  async provideFeedback(
    queryId: string,
    chunkId: string,
    helpful: boolean
  ): Promise<void> {
    const feedback = helpful ? 1 : 0;
    this.vectorStore.recordFeedback(queryId, chunkId, feedback);
  }

  /**
   * Get similar questions (for suggestion)
   */
  async getSimilarQuestions(query: string, limit: number = 5): Promise<string[]> {
    // This would use the query history in production
    const suggestions = [
      'What is the optimal alpha value for urban deployments?',
      'How does pZeroNominalPusch affect uplink SINR?',
      'What is the relationship between P0 and alpha in power control?',
      'How to solve the Tuning Paradox in RAN optimization?',
      'What are the 3GPP specifications for fractional power control?',
    ];

    return suggestions.slice(0, limit);
  }

  /**
   * Get vector store statistics
   */
  getStats(): {
    totalChunks: number;
    byDocumentType: Record<string, number>;
    momCount: number;
  } {
    const storeStats = this.vectorStore.getStats();
    return {
      ...storeStats,
      momCount: this.moms.length,
    };
  }

  /**
   * Persist state
   */
  async persist(): Promise<void> {
    await this.vectorStore.persist();
  }
}

export default RAGPipeline;
