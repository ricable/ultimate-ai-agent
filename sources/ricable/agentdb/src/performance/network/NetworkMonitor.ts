/**
 * Network and Communication Monitoring System
 *
 * Real-time monitoring of QUIC synchronization latency, inter-agent communication,
 * data transfer optimization, and network bottleneck detection
 */

import { EventEmitter } from 'events';

export interface NetworkMetrics {
  quicSynchronization: {
    connectionMetrics: {
      activeConnections: number;
      totalConnections: number;
      failedConnections: number;
      connectionSuccessRate: number;
      averageConnectionTime: number; // ms
    };
    performanceMetrics: {
      synchronizationLatency: number; // ms
      throughput: number; // Mbps
      packetLossRate: number; // percentage
      roundTripTime: number; // ms
      jitter: number; // ms
    };
    protocolMetrics: {
      version: string;
      congestionControlAlgorithm: string;
      maxDatagramSize: number;
      maxIdleTimeout: number; // ms
      keepAliveInterval: number; // ms
    };
    dataMetrics: {
      totalDataTransferred: number; // bytes
      compressionRatio: number;
      encryptedDataRatio: number;
      deltaSyncEfficiency: number; // percentage
      batchEfficiency: number; // percentage
    };
  };
  interAgentCommunication: {
    messageMetrics: {
      totalMessages: number;
      messagesPerSecond: number;
      averageMessageSize: number; // bytes
      messageTypes: Map<string, number>;
    };
    latencyMetrics: {
      averageLatency: number; // ms
      p95Latency: number; // ms
      p99Latency: number; // ms
      minLatency: number; // ms
      maxLatency: number; // ms
    };
    reliabilityMetrics: {
      deliverySuccessRate: number; // percentage
      retryRate: number; // percentage
      timeoutRate: number; // percentage
      errorRate: number; // percentage
    };
    bandwidthMetrics: {
      utilizedBandwidth: number; // Mbps
      availableBandwidth: number; // Mbps
      bandwidthUtilization: number; // percentage
      peakBandwidthUsage: number; // Mbps
    };
  };
  networkHealth: {
    overallHealthScore: number; // 0-100
    connectivityStatus: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
    bottleneckIndicators: {
      congestionDetected: boolean;
      highLatencyDetected: boolean;
      packetLossDetected: boolean;
      bandwidthSaturationDetected: boolean;
    };
    networkTopology: {
      nodeCount: number;
      edgeCount: number;
      averagePathLength: number;
      networkDiameter: number;
      clusteringCoefficient: number;
    };
  };
  optimizationMetrics: {
    compressionEffectiveness: {
      beforeCompression: number; // bytes
      afterCompression: number; // bytes
      compressionRatio: number;
      compressionOverhead: number; // ms
    };
    cachingEffectiveness: {
      cacheHitRate: number; // percentage
      bytesServedFromCache: number; // bytes
      cacheSize: number; // bytes
      cacheEvictions: number;
    };
    loadBalancingEffectiveness: {
      distributionBalance: number; // 0-1
      averageNodeLoad: number; // percentage
      overloadedNodes: number;
      loadRedistributionEvents: number;
    };
  };
}

export interface NetworkEvent {
  id: string;
  timestamp: Date;
  type: 'connection_established' | 'connection_lost' | 'high_latency' | 'packet_loss' | 'bandwidth_saturation' | 'topology_change';
  severity: 'info' | 'warning' | 'critical';
  source: string;
  target?: string;
  description: string;
  metrics: Record<string, number>;
  resolution?: string;
  resolutionTime?: number;
}

export interface NetworkOptimizationRecommendation {
  id: string;
  timestamp: Date;
  category: 'quic_optimization' | 'compression_tuning' | 'load_balancing' | 'topology_optimization';
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  currentPerformance: any;
  expectedImprovement: {
    latencyReduction: number; // percentage
    throughputIncrease: number; // percentage
    reliabilityImprovement: number; // percentage
  };
  configuration: {
    parameters: Record<string, any>;
    estimatedRisk: number; // 0-1
    rollbackPlan: string;
  };
  validation: {
    successCriteria: string[];
    monitoringPeriod: number; // minutes
    rollbackConditions: string[];
  };
}

export interface ConnectionAnalysis {
  connectionId: string;
  sourceAgent: string;
  targetAgent: string;
  establishedAt: Date;
  lastActivity: Date;
  status: 'active' | 'idle' | 'closing' | 'failed';
  metrics: {
    totalDataTransferred: number;
    averageLatency: number;
    packetLossRate: number;
    connectionStability: number; // 0-1
    efficiency: number; // 0-1
  };
  issues: Array<{
    type: string;
    severity: 'warning' | 'critical';
    description: string;
    timestamp: Date;
  }>;
}

export class NetworkMonitor extends EventEmitter {
  private metricsHistory: NetworkMetrics[] = [];
  private maxHistorySize: number = 1000;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private networkEvents: NetworkEvent[] = [];
  private activeConnections: Map<string, ConnectionAnalysis> = new Map();
  private optimizationRecommendations: NetworkOptimizationRecommendation[] = [];
  private thresholds = {
    synchronizationLatency: {
      warning: 5, // ms
      critical: 10 // ms
    },
    packetLossRate: {
      warning: 0.01, // 1%
      critical: 0.05 // 5%
    },
    throughput: {
      warning: 100, // Mbps
      critical: 50 // Mbps
    },
    connectionSuccessRate: {
      warning: 0.95, // 95%
      critical: 0.90 // 90%
    },
    bandwidthUtilization: {
      warning: 0.8, // 80%
      critical: 0.95 // 95%
    }
  };

  constructor() {
    super();
    this.startMonitoring();
  }

  public startMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(async () => {
      try {
        const metrics = await this.collectNetworkMetrics();
        this.processMetrics(metrics);
      } catch (error) {
        console.error('Error collecting network metrics:', error);
      }
    }, 5000); // Collect every 5 seconds

    this.emit('monitoring_started');
  }

  public stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    this.emit('monitoring_stopped');
  }

  private async collectNetworkMetrics(): Promise<NetworkMetrics> {
    // Simulate realistic network metrics for the RAN automation system

    // QUIC Synchronization Metrics
    const activeConnections = 8 + Math.floor(Math.random() * 4);
    const totalConnections = 12 + Math.floor(Math.random() * 6);
    const failedConnections = Math.floor(Math.random() * 2);

    const quicSynchronization = {
      connectionMetrics: {
        activeConnections,
        totalConnections,
        failedConnections,
        connectionSuccessRate: (totalConnections - failedConnections) / totalConnections,
        averageConnectionTime: 50 + Math.random() * 100 // ms
      },
      performanceMetrics: {
        synchronizationLatency: 0.8 + Math.random() * 4, // ms
        throughput: 200 + Math.random() * 300, // Mbps
        packetLossRate: Math.random() * 0.02, // 0-2%
        roundTripTime: 5 + Math.random() * 15, // ms
        jitter: 0.5 + Math.random() * 2 // ms
      },
      protocolMetrics: {
        version: 'QUIC v1',
        congestionControlAlgorithm: 'BBR',
        maxDatagramSize: 1200,
        maxIdleTimeout: 30000, // ms
        keepAliveInterval: 10000 // ms
      },
      dataMetrics: {
        totalDataTransferred: Math.floor(1e9 + Math.random() * 5e8), // bytes
        compressionRatio: 0.6 + Math.random() * 0.2, // 60-80%
        encryptedDataRatio: 1.0, // 100% encrypted
        deltaSyncEfficiency: 0.85 + Math.random() * 0.1, // 85-95%
        batchEfficiency: 0.75 + Math.random() * 0.2 // 75-95%
      }
    };

    // Inter-Agent Communication Metrics
    const messageTypes = new Map([
      ['task_assignment', 100 + Math.floor(Math.random() * 50)],
      ['result_sharing', 200 + Math.floor(Math.random() * 100)],
      ['coordination', 150 + Math.floor(Math.random() * 75)],
      ['heartbeat', 500 + Math.floor(Math.random() * 250)],
      ['data_sync', 80 + Math.floor(Math.random() * 40)]
    ]);

    const totalMessages = Array.from(messageTypes.values()).reduce((sum, count) => sum + count, 0);

    const interAgentCommunication = {
      messageMetrics: {
        totalMessages,
        messagesPerSecond: totalMessages / 5, // Per 5-second interval
        averageMessageSize: 1024 + Math.random() * 4096, // bytes
        messageTypes
      },
      latencyMetrics: {
        averageLatency: quicSynchronization.performanceMetrics.synchronizationLatency,
        p95Latency: quicSynchronization.performanceMetrics.synchronizationLatency * 2,
        p99Latency: quicSynchronization.performanceMetrics.synchronizationLatency * 3,
        minLatency: quicSynchronization.performanceMetrics.synchronizationLatency * 0.5,
        maxLatency: quicSynchronization.performanceMetrics.synchronizationLatency * 5
      },
      reliabilityMetrics: {
        deliverySuccessRate: 0.98 + Math.random() * 0.02, // 98-100%
        retryRate: Math.random() * 0.02, // 0-2%
        timeoutRate: Math.random() * 0.01, // 0-1%
        errorRate: Math.random() * 0.005 // 0-0.5%
      },
      bandwidthMetrics: {
        utilizedBandwidth: quicSynchronization.performanceMetrics.throughput,
        availableBandwidth: 1000, // 1 Gbps network
        bandwidthUtilization: quicSynchronization.performanceMetrics.throughput / 1000,
        peakBandwidthUsage: quicSynchronization.performanceMetrics.throughput * 1.2
      }
    };

    // Network Health Metrics
    const healthScore = this.calculateNetworkHealthScore(quicSynchronization, interAgentCommunication);

    const networkHealth = {
      overallHealthScore: healthScore,
      connectivityStatus: this.getConnectivityStatus(healthScore) as 'excellent' | 'good' | 'fair' | 'poor' | 'critical',
      bottleneckIndicators: {
        congestionDetected: interAgentCommunication.bandwidthMetrics.bandwidthUtilization > 0.8,
        highLatencyDetected: quicSynchronization.performanceMetrics.synchronizationLatency > 5,
        packetLossDetected: quicSynchronization.performanceMetrics.packetLossRate > 0.01,
        bandwidthSaturationDetected: interAgentCommunication.bandwidthMetrics.bandwidthUtilization > 0.95
      },
      networkTopology: {
        nodeCount: 12,
        edgeCount: 20,
        averagePathLength: 2.5,
        networkDiameter: 4,
        clusteringCoefficient: 0.6
      }
    };

    // Optimization Metrics
    const optimizationMetrics = {
      compressionEffectiveness: {
        beforeCompression: Math.floor(1e9 + Math.random() * 5e8),
        afterCompression: Math.floor((1e9 + Math.random() * 5e8) * quicSynchronization.dataMetrics.compressionRatio),
        compressionRatio: quicSynchronization.dataMetrics.compressionRatio,
        compressionOverhead: 1 + Math.random() * 3 // ms
      },
      cachingEffectiveness: {
        cacheHitRate: 0.85 + Math.random() * 0.1, // 85-95%
        bytesServedFromCache: Math.floor(5e8 + Math.random() * 2e8),
        cacheSize: 1024 * 1024 * 1024, // 1GB
        cacheEvictions: Math.floor(Math.random() * 100)
      },
      loadBalancingEffectiveness: {
        distributionBalance: 0.8 + Math.random() * 0.15, // 0-1
        averageNodeLoad: 0.6 + Math.random() * 0.3, // percentage
        overloadedNodes: Math.floor(Math.random() * 2),
        loadRedistributionEvents: Math.floor(Math.random() * 5)
      }
    };

    return {
      quicSynchronization,
      interAgentCommunication,
      networkHealth,
      optimizationMetrics
    };
  }

  private calculateNetworkHealthScore(quic: any, comm: any): number {
    let score = 0;
    let factors = 0;

    // Latency score (lower is better)
    const latencyScore = Math.max(0, 1 - quic.performanceMetrics.synchronizationLatency / 10);
    score += latencyScore;
    factors++;

    // Throughput score (higher is better)
    const throughputScore = Math.min(1, comm.bandwidthMetrics.utilizedBandwidth / 500);
    score += throughputScore;
    factors++;

    // Reliability score
    const reliabilityScore = comm.reliabilityMetrics.deliverySuccessRate;
    score += reliabilityScore;
    factors++;

    // Packet loss score (lower is better)
    const packetLossScore = Math.max(0, 1 - quic.performanceMetrics.packetLossRate * 20);
    score += packetLossScore;
    factors++;

    // Connection success rate
    const connectionScore = quic.connectionMetrics.connectionSuccessRate;
    score += connectionScore;
    factors++;

    // Bandwidth utilization (optimal range 60-80%)
    const utilizationScore = comm.bandwidthMetrics.bandwidthUtilization < 0.8
      ? comm.bandwidthMetrics.bandwidthUtilization / 0.8
      : Math.max(0, 2 - comm.bandwidthMetrics.bandwidthUtilization / 0.4);
    score += utilizationScore;
    factors++;

    return factors > 0 ? (score / factors) * 100 : 0;
  }

  private getConnectivityStatus(healthScore: number): string {
    if (healthScore >= 90) return 'excellent';
    if (healthScore >= 75) return 'good';
    if (healthScore >= 60) return 'fair';
    if (healthScore >= 40) return 'poor';
    return 'critical';
  }

  private processMetrics(metrics: NetworkMetrics): void {
    // Add to history
    this.metricsHistory.push(metrics);
    if (this.metricsHistory.length > this.maxHistorySize) {
      this.metricsHistory.shift();
    }

    // Update connection analysis
    this.updateConnectionAnalysis(metrics);

    // Check for network events
    this.checkNetworkEvents(metrics);

    // Generate optimization recommendations
    this.generateOptimizationRecommendations(metrics);

    // Emit metrics update
    this.emit('metrics_updated', metrics);
  }

  private updateConnectionAnalysis(metrics: NetworkMetrics): void {
    // Update existing connections and create new ones as needed
    const connectionCount = metrics.quicSynchronization.connectionMetrics.activeConnections;

    // Simulate connection updates
    for (let i = 0; i < connectionCount; i++) {
      const connectionId = `conn_${i}`;
      let connection = this.activeConnections.get(connectionId);

      if (!connection) {
        // Create new connection
        connection = {
          connectionId,
          sourceAgent: `agent_${i}`,
          targetAgent: `agent_${(i + 1) % 12}`,
          establishedAt: new Date(),
          lastActivity: new Date(),
          status: 'active',
          metrics: {
            totalDataTransferred: 0,
            averageLatency: metrics.quicSynchronization.performanceMetrics.synchronizationLatency,
            packetLossRate: metrics.quicSynchronization.performanceMetrics.packetLossRate,
            connectionStability: 0.9 + Math.random() * 0.1,
            efficiency: 0.8 + Math.random() * 0.2
          },
          issues: []
        };
        this.activeConnections.set(connectionId, connection);

        // Emit connection established event
        this.emitNetworkEvent({
          id: `conn_est_${Date.now()}_${i}`,
          timestamp: new Date(),
          type: 'connection_established',
          severity: 'info',
          source: connection.sourceAgent,
          target: connection.targetAgent,
          description: `Connection established between ${connection.sourceAgent} and ${connection.targetAgent}`,
          metrics: {
            latency: connection.metrics.averageLatency,
            efficiency: connection.metrics.efficiency
          }
        });
      } else {
        // Update existing connection
        connection.lastActivity = new Date();
        connection.metrics.totalDataTransferred += Math.floor(Math.random() * 1024 * 1024);
        connection.metrics.averageLatency = metrics.quicSynchronization.performanceMetrics.synchronizationLatency;
        connection.metrics.packetLossRate = metrics.quicSynchronization.performanceMetrics.packetLossRate;

        // Check for connection issues
        this.checkConnectionIssues(connection);
      }
    }

    // Remove inactive connections
    const cutoffTime = new Date(Date.now() - 60000); // 1 minute
    for (const [connectionId, connection] of this.activeConnections) {
      if (connection.lastActivity < cutoffTime) {
        connection.status = 'idle';
        this.activeConnections.delete(connectionId);

        // Emit connection lost event
        this.emitNetworkEvent({
          id: `conn_lost_${Date.now()}`,
          timestamp: new Date(),
          type: 'connection_lost',
          severity: 'warning',
          source: connection.sourceAgent,
          target: connection.targetAgent,
          description: `Connection lost between ${connection.sourceAgent} and ${connection.targetAgent}`,
          metrics: {
            totalDataTransferred: connection.metrics.totalDataTransferred,
            duration: connection.lastActivity.getTime() - connection.establishedAt.getTime()
          }
        });
      }
    }
  }

  private checkConnectionIssues(connection: ConnectionAnalysis): void {
    const issues: Array<{
      type: string;
      severity: 'warning' | 'critical';
      description: string;
      timestamp: Date;
    }> = [];

    // Check for high latency
    if (connection.metrics.averageLatency > this.thresholds.synchronizationLatency.critical) {
      issues.push({
        type: 'high_latency',
        severity: 'critical',
        description: `High latency detected: ${Math.round(connection.metrics.averageLatency)}ms`,
        timestamp: new Date()
      });
    } else if (connection.metrics.averageLatency > this.thresholds.synchronizationLatency.warning) {
      issues.push({
        type: 'high_latency',
        severity: 'warning',
        description: `Elevated latency: ${Math.round(connection.metrics.averageLatency)}ms`,
        timestamp: new Date()
      });
    }

    // Check for packet loss
    if (connection.metrics.packetLossRate > this.thresholds.packetLossRate.critical) {
      issues.push({
        type: 'packet_loss',
        severity: 'critical',
        description: `Critical packet loss: ${Math.round(connection.metrics.packetLossRate * 100)}%`,
        timestamp: new Date()
      });
    }

    // Check for low stability
    if (connection.metrics.connectionStability < 0.7) {
      issues.push({
        type: 'instability',
        severity: 'warning',
        description: `Connection instability detected: ${Math.round(connection.metrics.connectionStability * 100)}%`,
        timestamp: new Date()
      });
    }

    connection.issues = issues;
  }

  private checkNetworkEvents(metrics: NetworkMetrics): void {
    // Check for high latency events
    if (metrics.quicSynchronization.performanceMetrics.synchronizationLatency > this.thresholds.synchronizationLatency.critical) {
      this.emitNetworkEvent({
        id: `high_latency_${Date.now()}`,
        timestamp: new Date(),
        type: 'high_latency',
        severity: 'critical',
        source: 'Network_Monitor',
        description: `Critical synchronization latency: ${Math.round(metrics.quicSynchronization.performanceMetrics.synchronizationLatency)}ms`,
        metrics: {
          latency: metrics.quicSynchronization.performanceMetrics.synchronizationLatency,
          threshold: this.thresholds.synchronizationLatency.critical
        }
      });
    }

    // Check for packet loss events
    if (metrics.quicSynchronization.performanceMetrics.packetLossRate > this.thresholds.packetLossRate.critical) {
      this.emitNetworkEvent({
        id: `packet_loss_${Date.now()}`,
        timestamp: new Date(),
        type: 'packet_loss',
        severity: 'critical',
        source: 'Network_Monitor',
        description: `Critical packet loss detected: ${Math.round(metrics.quicSynchronization.performanceMetrics.packetLossRate * 100)}%`,
        metrics: {
          packetLossRate: metrics.quicSynchronization.performanceMetrics.packetLossRate,
          threshold: this.thresholds.packetLossRate.critical
        }
      });
    }

    // Check for bandwidth saturation
    if (metrics.interAgentCommunication.bandwidthMetrics.bandwidthUtilization > this.thresholds.bandwidthUtilization.critical) {
      this.emitNetworkEvent({
        id: `bandwidth_saturation_${Date.now()}`,
        timestamp: new Date(),
        type: 'bandwidth_saturation',
        severity: 'critical',
        source: 'Network_Monitor',
        description: `Network bandwidth saturation: ${Math.round(metrics.interAgentCommunication.bandwidthMetrics.bandwidthUtilization * 100)}%`,
        metrics: {
          utilization: metrics.interAgentCommunication.bandwidthMetrics.bandwidthUtilization,
          utilizedBandwidth: metrics.interAgentCommunication.bandwidthMetrics.utilizedBandwidth
        }
      });
    }
  }

  private emitNetworkEvent(event: NetworkEvent): void {
    this.networkEvents.push(event);

    // Keep only last 1000 events
    if (this.networkEvents.length > 1000) {
      this.networkEvents.shift();
    }

    this.emit('network_event', event);
  }

  private generateOptimizationRecommendations(metrics: NetworkMetrics): void {
    const recommendations: NetworkOptimizationRecommendation[] = [];

    // QUIC optimization recommendations
    if (metrics.quicSynchronization.performanceMetrics.synchronizationLatency > this.thresholds.synchronizationLatency.warning) {
      recommendations.push({
        id: `quic_opt_${Date.now()}`,
        timestamp: new Date(),
        category: 'quic_optimization',
        priority: 'high',
        title: 'QUIC Protocol Optimization',
        description: 'Optimize QUIC parameters to reduce synchronization latency',
        currentPerformance: {
          latency: metrics.quicSynchronization.performanceMetrics.synchronizationLatency,
          throughput: metrics.quicSynchronization.performanceMetrics.throughput
        },
        expectedImprovement: {
          latencyReduction: 30,
          throughputIncrease: 15,
          reliabilityImprovement: 10
        },
        configuration: {
          parameters: {
            maxIdleTimeout: 60000,
            maxDatagramSize: 1400,
            initialMaxData: 2097152,
            enableMultipath: true
          },
          estimatedRisk: 0.2,
          rollbackPlan: 'Revert to previous QUIC configuration'
        },
        validation: {
          successCriteria: [
            'Synchronization latency < 3ms',
            'No increase in packet loss',
            'Connection success rate > 95%'
          ],
          monitoringPeriod: 15,
          rollbackConditions: [
            'Latency increases by >20%',
            'Connection success rate drops <90%',
            'Packet loss rate >2%'
          ]
        }
      });
    }

    // Compression tuning recommendations
    if (metrics.optimizationMetrics.compressionEffectiveness.compressionRatio < 0.7) {
      recommendations.push({
        id: `compression_tuning_${Date.now()}`,
        timestamp: new Date(),
        category: 'compression_tuning',
        priority: 'medium',
        title: 'Compression Algorithm Tuning',
        description: 'Optimize compression settings for better data transfer efficiency',
        currentPerformance: {
          compressionRatio: metrics.optimizationMetrics.compressionEffectiveness.compressionRatio,
          overhead: metrics.optimizationMetrics.compressionEffectiveness.compressionOverhead
        },
        expectedImprovement: {
          latencyReduction: 20,
          throughputIncrease: 25,
          reliabilityImprovement: 5
        },
        configuration: {
          parameters: {
            compressionAlgorithm: 'zstd',
            compressionLevel: 6,
            enableAdaptiveCompression: true,
            compressionThreshold: 1024
          },
          estimatedRisk: 0.15,
          rollbackPlan: 'Revert to previous compression settings'
        },
        validation: {
          successCriteria: [
            'Compression ratio > 75%',
            'Compression overhead < 5ms',
            'No CPU usage increase >10%'
          ],
          monitoringPeriod: 10,
          rollbackConditions: [
            'Compression ratio drops <60%',
            'CPU usage increases >20%',
            'Latency increases >10%'
          ]
        }
      });
    }

    // Load balancing recommendations
    if (metrics.optimizationMetrics.loadBalancingEffectiveness.distributionBalance < 0.8) {
      recommendations.push({
        id: `load_balancing_${Date.now()}`,
        timestamp: new Date(),
        category: 'load_balancing',
        priority: 'medium',
        title: 'Load Balancing Optimization',
        description: 'Optimize network load distribution across agents',
        currentPerformance: {
          distributionBalance: metrics.optimizationMetrics.loadBalancingEffectiveness.distributionBalance,
          averageLoad: metrics.optimizationMetrics.loadBalancingEffectiveness.averageNodeLoad
        },
        expectedImprovement: {
          latencyReduction: 15,
          throughputIncrease: 20,
          reliabilityImprovement: 15
        },
        configuration: {
          parameters: {
            loadBalancingAlgorithm: 'weighted_round_robin',
            rebalanceInterval: 30,
            loadThreshold: 0.8,
            enableDynamicRebalancing: true
          },
          estimatedRisk: 0.25,
          rollbackPlan: 'Revert to previous load balancing configuration'
        },
        validation: {
          successCriteria: [
            'Distribution balance > 85%',
            'No overloaded nodes',
            'Rebalancing events < 5 per hour'
          ],
          monitoringPeriod: 20,
          rollbackConditions: [
            'Distribution balance drops <70%',
            'Overloaded nodes increase',
            'Rebalancing frequency > 10 per hour'
          ]
        }
      });
    }

    // Add new recommendations (avoid duplicates)
    recommendations.forEach(rec => {
      const exists = this.optimizationRecommendations.some(r =>
        r.category === rec.category && r.title === rec.title
      );
      if (!exists) {
        this.optimizationRecommendations.push(rec);
        this.emit('optimization_recommendation', rec);
      }
    });

    // Keep only recent recommendations (last 24 hours)
    const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000);
    this.optimizationRecommendations = this.optimizationRecommendations.filter(
      rec => rec.timestamp > cutoffTime
    );
  }

  public getCurrentMetrics(): NetworkMetrics | null {
    return this.metricsHistory.length > 0
      ? this.metricsHistory[this.metricsHistory.length - 1]
      : null;
  }

  public getMetricsHistory(limit?: number): NetworkMetrics[] {
    return limit ? this.metricsHistory.slice(-limit) : this.metricsHistory;
  }

  public getActiveConnections(): ConnectionAnalysis[] {
    return Array.from(this.activeConnections.values());
  }

  public getNetworkEvents(limit?: number): NetworkEvent[] {
    return limit ? this.networkEvents.slice(-limit) : this.networkEvents;
  }

  public getOptimizationRecommendations(): NetworkOptimizationRecommendation[] {
    return this.optimizationRecommendations;
  }

  public getNetworkHealthScore(): number {
    const metrics = this.getCurrentMetrics();
    return metrics ? metrics.networkHealth.overallHealthScore : 0;
  }

  public getBandwidthUtilization(): number {
    const metrics = this.getCurrentMetrics();
    return metrics ? metrics.interAgentCommunication.bandwidthMetrics.bandwidthUtilization : 0;
  }

  public getAverageLatency(): number {
    const metrics = this.getCurrentMetrics();
    return metrics ? metrics.quicSynchronization.performanceMetrics.synchronizationLatency : 0;
  }

  public getPacketLossRate(): number {
    const metrics = this.getCurrentMetrics();
    return metrics ? metrics.quicSynchronization.performanceMetrics.packetLossRate : 0;
  }

  public getThroughputTrend(): {
    trend: 'increasing' | 'stable' | 'decreasing';
    rate: number; // Mbps per minute
    prediction: number; // Mbps in 1 hour
  } {
    if (this.metricsHistory.length < 10) {
      return {
        trend: 'stable',
        rate: 0,
        prediction: 0
      };
    }

    const recent = this.metricsHistory.slice(-10);
    const throughputs = recent.map(m => m.quicSynchronization.performanceMetrics.throughput);

    // Calculate trend using linear regression
    const n = throughputs.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = throughputs.reduce((sum, val) => sum + val, 0);
    const sumXY = throughputs.reduce((sum, val, index) => sum + index * val, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    let trend: 'increasing' | 'stable' | 'decreasing' = 'stable';
    if (slope > 0.5) {
      trend = 'increasing';
    } else if (slope < -0.5) {
      trend = 'decreasing';
    }

    // Convert slope from per-interval to per-minute (assuming 5-second intervals)
    const rate = slope * 12; // Mbps per minute
    const currentThroughput = throughputs[throughputs.length - 1];
    const prediction = currentThroughput + (rate * 60); // Predict 1 hour ahead

    return {
      trend,
      rate: Math.round(rate * 100) / 100,
      prediction: Math.round(prediction * 100) / 100
    };
  }

  public async executeNetworkOptimization(recommendationId: string): Promise<boolean> {
    const recommendation = this.optimizationRecommendations.find(r => r.id === recommendationId);
    if (!recommendation) {
      throw new Error(`Optimization recommendation ${recommendationId} not found`);
    }

    console.log(`Executing network optimization: ${recommendation.title}`);

    // Simulate optimization execution
    console.log(`Applying configuration:`, recommendation.configuration.parameters);

    // Simulate execution time based on complexity
    const executionTime = 2000 + Math.random() * 3000; // 2-5 seconds
    await new Promise(resolve => setTimeout(resolve, executionTime));

    // Simulate success/failure
    const success = Math.random() > 0.1; // 90% success rate

    if (success) {
      console.log(`Network optimization completed successfully: ${recommendation.title}`);

      // Remove executed recommendation
      this.optimizationRecommendations = this.optimizationRecommendations.filter(r => r.id !== recommendationId);

      this.emit('optimization_executed', recommendation);
      return true;
    } else {
      console.error(`Network optimization failed: ${recommendation.title}`);
      return false;
    }
  }

  public exportNetworkData(): any {
    return {
      timestamp: new Date(),
      currentMetrics: this.getCurrentMetrics(),
      metricsHistory: this.getMetricsHistory(100), // Last 100 entries
      activeConnections: this.getActiveConnections(),
      networkEvents: this.getNetworkEvents(50), // Last 50 events
      optimizationRecommendations: this.getOptimizationRecommendations(),
      networkHealth: {
        healthScore: this.getNetworkHealthScore(),
        bandwidthUtilization: this.getBandwidthUtilization(),
        averageLatency: this.getAverageLatency(),
        packetLossRate: this.getPacketLossRate(),
        throughputTrend: this.getThroughputTrend()
      }
    };
  }

  public clearHistory(): void {
    this.metricsHistory = [];
    this.networkEvents = [];
    this.activeConnections.clear();
    this.optimizationRecommendations = [];
  }
}