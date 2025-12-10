/**
 * Medical Analysis REST API
 * Express-based API with WebSocket support for real-time updates
 */

import express, { Express, Request, Response, NextFunction } from 'express';
import { createServer, Server as HTTPServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { v4 as uuidv4 } from 'uuid';

import { MedicalAnalysisService } from '../services/medical-analysis.service';
import { AntiHallucinationService } from '../services/anti-hallucination.service';
import { AgentDBLearningService } from '../services/agentdb-learning.service';
import { ProviderService } from '../services/provider.service';
import { authMiddleware } from '../middleware/auth.middleware';
import { loggingMiddleware } from '../middleware/logging.middleware';

import type {
  AnalysisRequest,
  AnalysisResponse,
  AnalysisResult,
  ApiError,
  WebSocketMessage
} from '../types/medical.types';

export class MedicalAnalysisAPI {
  private app: Express;
  private server: HTTPServer;
  private wss: WebSocketServer;

  private analysisService: MedicalAnalysisService;
  private antiHallucinationService: AntiHallucinationService;
  private learningService: AgentDBLearningService;
  private providerService: ProviderService;

  private connections: Map<string, WebSocket> = new Map();

  constructor(private port: number = 3000) {
    this.app = express();
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });

    this.analysisService = new MedicalAnalysisService();
    this.antiHallucinationService = new AntiHallucinationService();
    this.learningService = new AgentDBLearningService();
    this.providerService = new ProviderService();

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
    this.setupErrorHandling();
  }

  /**
   * Setup Express middleware
   */
  private setupMiddleware(): void {
    // Security
    this.app.use(helmet());
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
      credentials: true
    }));

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 60 * 1000, // 1 minute
      max: 100, // limit each IP to 100 requests per minute
      message: 'Too many requests from this IP, please try again later.'
    });
    this.app.use('/api/', limiter);

    // Custom middleware
    this.app.use(loggingMiddleware);
    this.app.use('/api/', authMiddleware);
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    const router = express.Router();

    // Health check
    router.get('/health', (req: Request, res: Response) => {
      res.json({
        status: 'healthy',
        timestamp: new Date(),
        version: '1.0.0'
      });
    });

    // POST /analyze - Submit medical query for analysis
    router.post('/analyze', async (req: Request, res: Response) => {
      try {
        const request: AnalysisRequest = req.body;
        const requestId = uuidv4();

        // Validate request
        if (!request.symptoms || request.symptoms.length === 0) {
          return res.status(400).json(this.createErrorResponse(
            'INVALID_REQUEST',
            'Symptoms are required',
            requestId
          ));
        }

        // Start analysis
        const startTime = Date.now();

        // Send initial status via WebSocket if client connected
        const ws = this.connections.get(req.ip || '');
        if (ws) {
          this.sendWebSocketMessage(ws, {
            type: 'analysis_update',
            payload: {
              analysisId: requestId,
              status: 'analyzing',
              progress: 0,
              currentStep: 'Starting analysis'
            },
            timestamp: new Date()
          });
        }

        // Perform analysis
        const result = await this.analysisService.analyze(request);

        // Calculate confidence score
        const confidenceScore = this.antiHallucinationService.calculateConfidenceScore(result);
        result.confidenceScore = confidenceScore;

        // Generate warnings
        const warnings = this.antiHallucinationService.generateWarnings(result, confidenceScore);
        result.warnings = warnings;

        // Check for pattern recognition
        const patterns = await this.learningService.recognizePatterns(
          request.symptoms,
          { condition: request.condition }
        );

        // Determine if provider review is required
        result.requiresProviderReview = this.antiHallucinationService.requiresProviderReview(confidenceScore);

        if (result.requiresProviderReview) {
          result.status = 'provider_review';
          await this.providerService.notifyProvider(result.id, result);
        }

        const processingTime = Date.now() - startTime;

        // Send final status via WebSocket
        if (ws) {
          this.sendWebSocketMessage(ws, {
            type: 'analysis_update',
            payload: {
              analysisId: result.id,
              status: result.status,
              progress: 100,
              currentStep: 'Completed'
            },
            timestamp: new Date()
          });
        }

        const response: AnalysisResponse = {
          success: true,
          data: result,
          metadata: {
            requestId,
            timestamp: new Date(),
            processingTimeMs: processingTime,
            version: '1.0.0'
          }
        };

        res.json(response);

      } catch (error) {
        console.error('Analysis error:', error);
        res.status(500).json(this.createErrorResponse(
          'ANALYSIS_ERROR',
          error instanceof Error ? error.message : 'Unknown error',
          uuidv4()
        ));
      }
    });

    // GET /analysis/:id - Get analysis results
    router.get('/analysis/:id', async (req: Request, res: Response) => {
      try {
        const { id } = req.params;
        const result = await this.analysisService.getAnalysis(id);

        if (!result) {
          return res.status(404).json(this.createErrorResponse(
            'NOT_FOUND',
            'Analysis not found',
            uuidv4()
          ));
        }

        res.json({
          success: true,
          data: result,
          metadata: {
            requestId: uuidv4(),
            timestamp: new Date(),
            processingTimeMs: 0,
            version: '1.0.0'
          }
        });

      } catch (error) {
        console.error('Retrieval error:', error);
        res.status(500).json(this.createErrorResponse(
          'RETRIEVAL_ERROR',
          error instanceof Error ? error.message : 'Unknown error',
          uuidv4()
        ));
      }
    });

    // POST /provider/review - Provider review endpoint
    router.post('/provider/review', async (req: Request, res: Response) => {
      try {
        const { analysisId, decision, comments } = req.body;

        if (!analysisId || !decision) {
          return res.status(400).json(this.createErrorResponse(
            'INVALID_REQUEST',
            'Analysis ID and decision are required',
            uuidv4()
          ));
        }

        await this.providerService.submitReview(analysisId, {
          decision,
          comments
        });

        // Learn from provider feedback
        const analysis = await this.analysisService.getAnalysis(analysisId);
        if (analysis) {
          await this.learningService.learnFromAnalysis(
            analysis,
            decision === 'approved' ? 'successful' : 'modified',
            comments
          );
        }

        res.json({
          success: true,
          message: 'Provider review submitted successfully',
          metadata: {
            requestId: uuidv4(),
            timestamp: new Date(),
            processingTimeMs: 0,
            version: '1.0.0'
          }
        });

      } catch (error) {
        console.error('Review submission error:', error);
        res.status(500).json(this.createErrorResponse(
          'REVIEW_ERROR',
          error instanceof Error ? error.message : 'Unknown error',
          uuidv4()
        ));
      }
    });

    // POST /provider/notify - Notify provider
    router.post('/provider/notify', async (req: Request, res: Response) => {
      try {
        const { analysisId, urgent } = req.body;

        if (!analysisId) {
          return res.status(400).json(this.createErrorResponse(
            'INVALID_REQUEST',
            'Analysis ID is required',
            uuidv4()
          ));
        }

        const analysis = await this.analysisService.getAnalysis(analysisId);
        if (!analysis) {
          return res.status(404).json(this.createErrorResponse(
            'NOT_FOUND',
            'Analysis not found',
            uuidv4()
          ));
        }

        await this.providerService.notifyProvider(analysisId, analysis, urgent);

        res.json({
          success: true,
          message: 'Provider notified successfully',
          metadata: {
            requestId: uuidv4(),
            timestamp: new Date(),
            processingTimeMs: 0,
            version: '1.0.0'
          }
        });

      } catch (error) {
        console.error('Notification error:', error);
        res.status(500).json(this.createErrorResponse(
          'NOTIFICATION_ERROR',
          error instanceof Error ? error.message : 'Unknown error',
          uuidv4()
        ));
      }
    });

    // GET /metrics - Get learning metrics
    router.get('/metrics', async (req: Request, res: Response) => {
      try {
        const metrics = await this.learningService.getMetrics();

        res.json({
          success: true,
          data: metrics,
          metadata: {
            requestId: uuidv4(),
            timestamp: new Date(),
            processingTimeMs: 0,
            version: '1.0.0'
          }
        });

      } catch (error) {
        console.error('Metrics error:', error);
        res.status(500).json(this.createErrorResponse(
          'METRICS_ERROR',
          error instanceof Error ? error.message : 'Unknown error',
          uuidv4()
        ));
      }
    });

    this.app.use('/api', router);
  }

  /**
   * Setup WebSocket for real-time updates
   */
  private setupWebSocket(): void {
    this.wss.on('connection', (ws: WebSocket, req) => {
      const clientId = req.socket.remoteAddress || uuidv4();
      this.connections.set(clientId, ws);

      console.log(`WebSocket client connected: ${clientId}`);

      ws.on('message', (message: string) => {
        try {
          const data = JSON.parse(message.toString());
          this.handleWebSocketMessage(ws, data, clientId);
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      });

      ws.on('close', () => {
        this.connections.delete(clientId);
        console.log(`WebSocket client disconnected: ${clientId}`);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.connections.delete(clientId);
      });

      // Send welcome message
      this.sendWebSocketMessage(ws, {
        type: 'analysis_update',
        payload: { message: 'Connected to MedAI Analysis System' },
        timestamp: new Date()
      });
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleWebSocketMessage(ws: WebSocket, data: any, clientId: string): void {
    // Handle subscribe/unsubscribe, ping/pong, etc.
    if (data.type === 'ping') {
      this.sendWebSocketMessage(ws, {
        type: 'analysis_update',
        payload: { message: 'pong' },
        timestamp: new Date()
      });
    }
  }

  /**
   * Send WebSocket message
   */
  private sendWebSocketMessage(ws: WebSocket, message: WebSocketMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  /**
   * Broadcast to all connected clients
   */
  private broadcast(message: WebSocketMessage): void {
    this.connections.forEach(ws => {
      this.sendWebSocketMessage(ws, message);
    });
  }

  /**
   * Setup error handling
   */
  private setupErrorHandling(): void {
    this.app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
      console.error('Unhandled error:', err);
      res.status(500).json(this.createErrorResponse(
        'INTERNAL_ERROR',
        'An unexpected error occurred',
        uuidv4()
      ));
    });
  }

  /**
   * Create standardized error response
   */
  private createErrorResponse(code: string, message: string, requestId: string): AnalysisResponse {
    return {
      success: false,
      error: {
        code,
        message,
        timestamp: new Date()
      },
      metadata: {
        requestId,
        timestamp: new Date(),
        processingTimeMs: 0,
        version: '1.0.0'
      }
    };
  }

  /**
   * Start the server
   */
  public start(): void {
    this.server.listen(this.port, () => {
      console.log(`🏥 MedAI Analysis API started on port ${this.port}`);
      console.log(`📡 WebSocket endpoint: ws://localhost:${this.port}`);
      console.log(`🔒 Security: Helmet, CORS, Rate Limiting enabled`);
      console.log(`🧠 Anti-Hallucination: Active`);
      console.log(`📚 AgentDB Learning: Active`);
    });
  }

  /**
   * Stop the server
   */
  public async stop(): Promise<void> {
    return new Promise((resolve) => {
      this.wss.close(() => {
        this.server.close(() => {
          console.log('Server stopped');
          resolve();
        });
      });
    });
  }
}

// Export singleton instance
export const api = new MedicalAnalysisAPI(parseInt(process.env.PORT || '3000'));

// Start server if run directly
if (require.main === module) {
  api.start();
}
