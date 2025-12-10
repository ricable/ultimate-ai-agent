/**
 * RuVector Logging Utility
 *
 * Structured logging for the Ericsson RAN Cognitive Automation Platform
 */

import winston from 'winston';
import { getConfig } from '../core/config.js';

const config = getConfig();

const logFormat = config.logging.format === 'json'
  ? winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json()
    )
  : winston.format.combine(
      winston.format.timestamp(),
      winston.format.colorize(),
      winston.format.errors({ stack: true }),
      winston.format.printf(({ level, message, timestamp, ...meta }) => {
        const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
        return `${timestamp} [${level}]: ${message}${metaStr}`;
      })
    );

export const logger = winston.createLogger({
  level: config.logging.level,
  format: logFormat,
  defaultMeta: { service: 'ruvector-telecom-rag' },
  transports: [
    new winston.transports.Console(),
  ],
});

// Add file transport in production
if (process.env.NODE_ENV === 'production') {
  logger.add(new winston.transports.File({
    filename: 'logs/error.log',
    level: 'error'
  }));
  logger.add(new winston.transports.File({
    filename: 'logs/combined.log'
  }));
}

// Helper functions for structured logging
export function logDocumentIngestion(documentId: string, type: string, status: 'started' | 'completed' | 'failed', details?: Record<string, unknown>) {
  logger.info('Document ingestion', {
    event: 'document_ingestion',
    documentId,
    documentType: type,
    status,
    ...details,
  });
}

export function logGNNTraining(epoch: number, loss: number, metrics: Record<string, number>) {
  logger.info('GNN training progress', {
    event: 'gnn_training',
    epoch,
    loss,
    ...metrics,
  });
}

export function logAgentAction(agentId: string, action: string, details: Record<string, unknown>) {
  logger.info('Agent action', {
    event: 'agent_action',
    agentId,
    action,
    ...details,
  });
}

export function logOptimization(clusterId: string, status: string, metrics?: Record<string, number>) {
  logger.info('Optimization event', {
    event: 'optimization',
    clusterId,
    status,
    ...metrics,
  });
}

export function logRAGQuery(queryId: string, query: string, resultsCount: number, latencyMs: number) {
  logger.info('RAG query', {
    event: 'rag_query',
    queryId,
    queryLength: query.length,
    resultsCount,
    latencyMs,
  });
}
