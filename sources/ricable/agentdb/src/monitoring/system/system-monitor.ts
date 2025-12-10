/**
 * System Performance Monitoring System
 * CPU, memory, network, and storage monitoring with bottleneck detection and optimization
 */

import { EventEmitter } from 'events';
import { createLogger, Logger } from '../../utils/logger';
import { MetricsCollector } from '../deployment/metrics-collector';
import { AlertManager } from '../deployment/alert-manager';
import { DashboardManager } from '../deployment/dashboard-manager';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface SystemMetrics {
  timestamp: Date;
  cpu: {
    usage: number;
    loadAverage: number[];
    cores: number;
    processes: number;
  };
  memory: {
    total: number;
    used: number;
    free: number;
    cached: number;
    swap: {
      total: number;
      used: number;
      free: number;
    };
    usagePercentage: number;
  };
  disk: {
    total: number;
    used: number;
    free: number;
    usagePercentage: number;
    iops: {
      reads: number;
      writes: number;
    };
    throughput: {
      readBytes: number;
      writeBytes: number;
    };
  }[];
  network: {
    interfaces: NetworkInterface[];
    total: {
      bytesReceived: number;
      bytesSent: number;
      packetsReceived: number;
      packetsSent: number;
      errorsIn: number;
      errorsOut: number;
    };
  };
  processes: ProcessInfo[];
  temperature?: {
    cpu: number;
    gpu?: number;
    system?: number;
  };
}

export interface NetworkInterface {
  name: string;
  bytesReceived: number;
  bytesSent: number;
  packetsReceived: number;
  packetsSent: number;
  errorsIn: number;
  errorsOut: number;
  speed: number;
  duplex: string;
  mtu: number;
}

export interface ProcessInfo {
  pid: number;
  name: string;
  cpu: number;
  memory: number;
  status: string;
  user: string;
  startTime: Date;
  command: string;
}

export interface BottleneckDetection {
  type: 'cpu' | 'memory' | 'disk' | 'network' | 'process';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  metric: string;
  value: number;
  threshold: number;
  recommendations: string[];
  timestamp: Date;
}

export interface PerformanceBaseline {
  cpu: { min: number; max: number; avg: number };
  memory: { min: number; max: number; avg: number };
  disk: { min: number; max: number; avg: number }[];
  network: { min: number; max: number; avg: number };
  timestamp: Date;
  duration: number;
}

export class SystemMonitor extends EventEmitter {
  private logger: Logger;
  private metricsCollector: MetricsCollector;
  private alertManager: AlertManager;
  private dashboardManager: DashboardManager;
  private currentMetrics: SystemMetrics | null = null;
  private historicalMetrics: SystemMetrics[] = [];
  private baseline: PerformanceBaseline | null = null;
  private bottlenecks: BottleneckDetection[] = [];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isCollectingMetrics = false;

  constructor() {
    super();
    this.logger = createLogger('SystemMonitor');
    this.metricsCollector = new MetricsCollector('system');
    this.alertManager = new AlertManager();
    this.dashboardManager = new DashboardManager();
    this.initializeMonitoring();
  }

  private initializeMonitoring(): void {
    // Start metrics collection
    this.startMetricsCollection();

    // Setup monitoring intervals
    setInterval(() => this.detectBottlenecks(), 10000); // Every 10 seconds
    setInterval(() => this.analyzeTrends(), 60000); // Every minute
    setInterval(() => this.generateSystemReport(), 300000); // Every 5 minutes

    this.logger.info('System monitoring initialized');
    this.emit('monitoring:initialized');
  }

  /**
   * Start metrics collection
   */
  private startMetricsCollection(): void {
    if (this.isCollectingMetrics) {
      this.logger.warn('Metrics collection already running');
      return;
    }

    this.isCollectingMetrics = true;
    this.monitoringInterval = setInterval(async () => {
      try {
        const metrics = await this.collectSystemMetrics();
        this.currentMetrics = metrics;
        this.historicalMetrics.push(metrics);

        // Keep only last 1000 metrics (about 8.3 hours at 30-second intervals)
        if (this.historicalMetrics.length > 1000) {
          this.historicalMetrics.shift();
        }

        // Store metrics for historical analysis
        await this.storeMetrics(metrics);

        this.emit('metrics:collected', metrics);

      } catch (error) {
        this.logger.error('Failed to collect system metrics:', error);
        this.emit('metrics:error', error);
      }
    }, 5000); // Every 5 seconds

    this.logger.info('System metrics collection started');
  }

  /**
   * Collect comprehensive system metrics
   */
  async collectSystemMetrics(): Promise<SystemMetrics> {
    const timestamp = new Date();

    try {
      // Collect CPU metrics
      const cpuMetrics = await this.collectCpuMetrics();

      // Collect memory metrics
      const memoryMetrics = await this.collectMemoryMetrics();

      // Collect disk metrics
      const diskMetrics = await this.collectDiskMetrics();

      // Collect network metrics
      const networkMetrics = await this.collectNetworkMetrics();

      // Collect process information
      const processMetrics = await this.collectProcessMetrics();

      // Collect temperature metrics (if available)
      const temperatureMetrics = await this.collectTemperatureMetrics();

      return {
        timestamp,
        cpu: cpuMetrics,
        memory: memoryMetrics,
        disk: diskMetrics,
        network: networkMetrics,
        processes: processMetrics,
        temperature: temperatureMetrics
      };

    } catch (error) {
      this.logger.error('Failed to collect system metrics:', error);
      throw error;
    }
  }

  /**
   * Collect CPU metrics
   */
  private async collectCpuMetrics(): Promise<SystemMetrics['cpu']> {
    try {
      // Get CPU usage
      const { stdout: cpuUsageOutput } = await execAsync("top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'");
      const usage = parseFloat(cpuUsageOutput.trim()) || 0;

      // Get load average
      const { stdout: loadAvgOutput } = await execAsync("cat /proc/loadavg");
      const loadAverage = loadAvgOutput.trim().split(' ').slice(0, 3).map(Number);

      // Get CPU info
      const { stdout: cpuInfoOutput } = await execAsync("nproc");
      const cores = parseInt(cpuInfoOutput.trim()) || 1;

      // Get process count
      const { stdout: processCountOutput } = await execAsync("ps aux | wc -l");
      const processes = parseInt(processCountOutput.trim()) || 0;

      return {
        usage,
        loadAverage,
        cores,
        processes
      };

    } catch (error) {
      this.logger.error('Failed to collect CPU metrics:', error);
      return {
        usage: 0,
        loadAverage: [0, 0, 0],
        cores: 1,
        processes: 0
      };
    }
  }

  /**
   * Collect memory metrics
   */
  private async collectMemoryMetrics(): Promise<SystemMetrics['memory']> {
    try {
      const { stdout: memInfoOutput } = await execAsync("cat /proc/meminfo");
      const lines = memInfoOutput.trim().split('\n');

      const parseMemInfo = (key: string): number => {
        const line = lines.find(l => l.startsWith(key));
        if (line) {
          const match = line.match(/\d+/);
          return match ? parseInt(match[0]) * 1024 : 0; // Convert KB to bytes
        }
        return 0;
      };

      const total = parseMemInfo('MemTotal:');
      const free = parseMemInfo('MemFree:');
      const available = parseMemInfo('MemAvailable:');
      const buffers = parseMemInfo('Buffers:');
      const cached = parseMemInfo('Cached:');
      const swapTotal = parseMemInfo('SwapTotal:');
      const swapFree = parseMemInfo('SwapFree:');

      const used = total - free - buffers - cached;
      const swapUsed = swapTotal - swapFree;
      const usagePercentage = (used / total) * 100;

      return {
        total,
        used,
        free,
        cached,
        swap: {
          total: swapTotal,
          used: swapUsed,
          free: swapFree
        },
        usagePercentage
      };

    } catch (error) {
      this.logger.error('Failed to collect memory metrics:', error);
      return {
        total: 0,
        used: 0,
        free: 0,
        cached: 0,
        swap: {
          total: 0,
          used: 0,
          free: 0
        },
        usagePercentage: 0
      };
    }
  }

  /**
   * Collect disk metrics
   */
  private async collectDiskMetrics(): Promise<SystemMetrics['disk']> {
    try {
      const { stdout: dfOutput } = await execAsync("df -h --output=source,size,used,avail,pcent,target | tail -n +2");
      const lines = dfOutput.trim().split('\n');

      const diskMetrics = [];

      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 6) {
          const parseSize = (sizeStr: string): number => {
            const match = sizeStr.match(/^(\d+(?:\.\d+)?)(K|M|G|T)?$/);
            if (match) {
              const value = parseFloat(match[1]);
              const unit = match[2] || '';
              const multipliers = { K: 1024, M: 1024 * 1024, G: 1024 * 1024 * 1024, T: 1024 * 1024 * 1024 * 1024 };
              return Math.round(value * (multipliers[unit as keyof typeof multipliers] || 1));
            }
            return 0;
          };

          const total = parseSize(parts[1]);
          const used = parseSize(parts[2]);
          const free = parseSize(parts[3]);
          const usagePercentage = parseFloat(parts[4].replace('%', '')) || 0;

          // Get I/O stats for this disk
          const iops = await this.collectDiskIops(parts[0]);
          const throughput = await this.collectDiskThroughput(parts[0]);

          diskMetrics.push({
            total,
            used,
            free,
            usagePercentage,
            iops,
            throughput
          });
        }
      }

      return diskMetrics;

    } catch (error) {
      this.logger.error('Failed to collect disk metrics:', error);
      return [];
    }
  }

  /**
   * Collect disk IOPS
   */
  private async collectDiskIops(device: string): Promise<{ reads: number; writes: number }> {
    try {
      // Simplified IOPS collection - in production, you'd use /proc/diskstats
      return { reads: 0, writes: 0 };
    } catch (error) {
      return { reads: 0, writes: 0 };
    }
  }

  /**
   * Collect disk throughput
   */
  private async collectDiskThroughput(device: string): Promise<{ readBytes: number; writeBytes: number }> {
    try {
      // Simplified throughput collection - in production, you'd use /proc/diskstats
      return { readBytes: 0, writeBytes: 0 };
    } catch (error) {
      return { readBytes: 0, writeBytes: 0 };
    }
  }

  /**
   * Collect network metrics
   */
  private async collectNetworkMetrics(): Promise<SystemMetrics['network']> {
    try {
      const { stdout: networkOutput } = await execAsync("cat /proc/net/dev");
      const lines = networkOutput.trim().split('\n').slice(2); // Skip header lines

      const interfaces: NetworkInterface[] = [];
      let totalBytesReceived = 0;
      let totalBytesSent = 0;
      let totalPacketsReceived = 0;
      let totalPacketsSent = 0;
      let totalErrorsIn = 0;
      let totalErrorsOut = 0;

      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 17) {
          const interfaceName = parts[0].replace(':', '');
          const bytesReceived = parseInt(parts[1]) || 0;
          const packetsReceived = parseInt(parts[2]) || 0;
          const errorsIn = parseInt(parts[3]) || 0;
          const bytesSent = parseInt(parts[9]) || 0;
          const packetsSent = parseInt(parts[10]) || 0;
          const errorsOut = parseInt(parts[11]) || 0;

          // Skip loopback interface
          if (interfaceName === 'lo') continue;

          interfaces.push({
            name: interfaceName,
            bytesReceived,
            bytesSent,
            packetsReceived,
            packetsSent,
            errorsIn,
            errorsOut,
            speed: 0, // Would need additional commands to get speed
            duplex: 'unknown',
            mtu: 1500 // Default MTU
          });

          totalBytesReceived += bytesReceived;
          totalBytesSent += bytesSent;
          totalPacketsReceived += packetsReceived;
          totalPacketsSent += packetsSent;
          totalErrorsIn += errorsIn;
          totalErrorsOut += errorsOut;
        }
      }

      return {
        interfaces,
        total: {
          bytesReceived: totalBytesReceived,
          bytesSent: totalBytesSent,
          packetsReceived: totalPacketsReceived,
          packetsSent: totalPacketsSent,
          errorsIn: totalErrorsIn,
          errorsOut: totalErrorsOut
        }
      };

    } catch (error) {
      this.logger.error('Failed to collect network metrics:', error);
      return {
        interfaces: [],
        total: {
          bytesReceived: 0,
          bytesSent: 0,
          packetsReceived: 0,
          packetsSent: 0,
          errorsIn: 0,
          errorsOut: 0
        }
      };
    }
  }

  /**
   * Collect process metrics
   */
  private async collectProcessMetrics(): Promise<ProcessInfo[]> {
    try {
      const { stdout: psOutput } = await execAsync("ps aux --sort=-%cpu | head -20");
      const lines = psOutput.trim().split('\n').slice(1); // Skip header

      const processes: ProcessInfo[] = [];

      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 11) {
          processes.push({
            pid: parseInt(parts[1]) || 0,
            name: parts[10] || '',
            cpu: parseFloat(parts[2]) || 0,
            memory: parseFloat(parts[3]) || 0,
            status: parts[7] || '',
            user: parts[0] || '',
            startTime: new Date(), // Would need more complex parsing
            command: parts.slice(10).join(' ') || ''
          });
        }
      }

      return processes;

    } catch (error) {
      this.logger.error('Failed to collect process metrics:', error);
      return [];
    }
  }

  /**
   * Collect temperature metrics
   */
  private async collectTemperatureMetrics(): Promise<SystemMetrics['temperature']> {
    try {
      // Try to get temperature from various sources
      try {
        const { stdout: tempOutput } = await execAsync("sensors 2>/dev/null | grep -E '(Core|Package|temp)' | awk '{print $3}' | head -5");
        const temps = tempOutput.trim().split('\n').map(t => parseFloat(t.replace('°C', ''))).filter(t => !isNaN(t));

        if (temps.length > 0) {
          return {
            cpu: temps[0],
            system: temps[temps.length > 1 ? 1 : 0]
          };
        }
      } catch {
        // sensors command not available
      }

      return {
        cpu: 0,
        system: 0
      };

    } catch (error) {
      this.logger.debug('Temperature metrics not available');
      return undefined;
    }
  }

  /**
   * Store metrics for historical analysis
   */
  private async storeMetrics(metrics: SystemMetrics): Promise<void> {
    try {
      await this.metricsCollector.recordMetric('cpu_usage', metrics.cpu.usage);
      await this.metricsCollector.recordMetric('memory_usage', metrics.memory.usagePercentage);
      await this.metricsCollector.recordMetric('disk_usage', metrics.disk[0]?.usagePercentage || 0);

      // Store network metrics
      await this.metricsCollector.recordMetric('network_bytes_in', metrics.network.total.bytesReceived);
      await this.metricsCollector.recordMetric('network_bytes_out', metrics.network.total.bytesSent);

      // Store temperature if available
      if (metrics.temperature) {
        await this.metricsCollector.recordMetric('cpu_temperature', metrics.temperature.cpu);
        await this.metricsCollector.recordMetric('system_temperature', metrics.temperature.system || 0);
      }

    } catch (error) {
      this.logger.error('Failed to store metrics:', error);
    }
  }

  /**
   * Detect system bottlenecks
   */
  private async detectBottlenecks(): Promise<void> {
    if (!this.currentMetrics) return;

    const newBottlenecks: BottleneckDetection[] = [];

    // CPU bottleneck detection
    if (this.currentMetrics.cpu.usage > 90) {
      newBottlenecks.push({
        type: 'cpu',
        severity: this.currentMetrics.cpu.usage > 95 ? 'critical' : 'high',
        description: `High CPU usage detected: ${this.currentMetrics.cpu.usage.toFixed(1)}%`,
        metric: 'cpu_usage',
        value: this.currentMetrics.cpu.usage,
        threshold: 90,
        recommendations: [
          'Consider scaling up CPU resources',
          'Optimize CPU-intensive processes',
          'Check for runaway processes'
        ],
        timestamp: new Date()
      });
    }

    // Memory bottleneck detection
    if (this.currentMetrics.memory.usagePercentage > 85) {
      newBottlenecks.push({
        type: 'memory',
        severity: this.currentMetrics.memory.usagePercentage > 95 ? 'critical' : 'high',
        description: `High memory usage detected: ${this.currentMetrics.memory.usagePercentage.toFixed(1)}%`,
        metric: 'memory_usage',
        value: this.currentMetrics.memory.usagePercentage,
        threshold: 85,
        recommendations: [
          'Consider adding more memory',
          'Optimize memory usage in applications',
          'Check for memory leaks'
        ],
        timestamp: new Date()
      });
    }

    // Disk bottleneck detection
    for (const disk of this.currentMetrics.disk) {
      if (disk.usagePercentage > 90) {
        newBottlenecks.push({
          type: 'disk',
          severity: disk.usagePercentage > 95 ? 'critical' : 'high',
          description: `High disk usage on ${disk.usagePercentage.toFixed(1)}%`,
          metric: 'disk_usage',
          value: disk.usagePercentage,
          threshold: 90,
          recommendations: [
            'Clean up unnecessary files',
            'Consider disk space expansion',
            'Archive old data'
          ],
          timestamp: new Date()
        });
      }
    }

    // Network bottleneck detection
    const networkErrors = this.currentMetrics.network.total.errorsIn + this.currentMetrics.network.total.errorsOut;
    if (networkErrors > 1000) {
      newBottlenecks.push({
        type: 'network',
        severity: networkErrors > 10000 ? 'critical' : 'medium',
        description: `High network error rate detected: ${networkErrors} errors`,
        metric: 'network_errors',
        value: networkErrors,
        threshold: 1000,
        recommendations: [
          'Check network cable connections',
          'Verify network interface health',
          'Monitor network traffic patterns'
        ],
        timestamp: new Date()
      });
    }

    // Process bottleneck detection
    const highCpuProcesses = this.currentMetrics.processes.filter(p => p.cpu > 80);
    if (highCpuProcesses.length > 0) {
      const topProcess = highCpuProcesses[0];
      newBottlenecks.push({
        type: 'process',
        severity: topProcess.cpu > 95 ? 'critical' : 'high',
        description: `High CPU process detected: ${topProcess.name} (${topProcess.cpu.toFixed(1)}%)`,
        metric: 'process_cpu',
        value: topProcess.cpu,
        threshold: 80,
        recommendations: [
          `Investigate process ${topProcess.name} (PID: ${topProcess.pid})`,
          'Check if process is responsive',
          'Consider restarting the process if necessary'
        ],
        timestamp: new Date()
      });
    }

    // Temperature bottleneck detection
    if (this.currentMetrics.temperature) {
      if (this.currentMetrics.temperature.cpu > 80) {
        newBottlenecks.push({
          type: 'cpu',
          severity: this.currentMetrics.temperature.cpu > 90 ? 'critical' : 'high',
          description: `High CPU temperature: ${this.currentMetrics.temperature.cpu.toFixed(1)}°C`,
          metric: 'cpu_temperature',
          value: this.currentMetrics.temperature.cpu,
          threshold: 80,
          recommendations: [
            'Check cooling system',
            'Ensure proper ventilation',
            'Consider thermal throttling if necessary'
          ],
          timestamp: new Date()
        });
      }
    }

    // Update bottlenecks list
    this.bottlenecks = [...this.bottlenecks.filter(b => Date.now() - b.timestamp.getTime() < 300000), ...newBottlenecks]; // Keep last 5 minutes

    // Send alerts for critical bottlenecks
    for (const bottleneck of newBottlenecks) {
      if (bottleneck.severity === 'critical') {
        await this.alertManager.sendAlert({
          level: 'critical',
          title: `Critical System Bottleneck: ${bottleneck.type.toUpperCase()}`,
          message: bottleneck.description,
          metric: bottleneck.metric,
          value: bottleneck.value,
          timestamp: bottleneck.timestamp
        });
      }
    }

    if (newBottlenecks.length > 0) {
      this.emit('bottlenecks:detected', newBottlenecks);
    }
  }

  /**
   * Analyze system trends
   */
  private async analyzeTrends(): Promise<void> {
    if (this.historicalMetrics.length < 10) return;

    try {
      const recentMetrics = this.historicalMetrics.slice(-60); // Last 5 minutes (at 5-second intervals)

      // Analyze CPU trend
      const cpuTrend = this.calculateTrend(recentMetrics.map(m => m.cpu.usage));

      // Analyze memory trend
      const memoryTrend = this.calculateTrend(recentMetrics.map(m => m.memory.usagePercentage));

      // Detect abnormal patterns
      await this.detectAbnormalPatterns(cpuTrend, memoryTrend);

    } catch (error) {
      this.logger.error('Failed to analyze trends:', error);
    }
  }

  /**
   * Calculate trend from array of values
   */
  private calculateTrend(values: number[]): { slope: number; direction: 'increasing' | 'decreasing' | 'stable' } {
    if (values.length < 2) return { slope: 0, direction: 'stable' };

    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = values.reduce((sum, val, index) => sum + val * index, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    const direction = Math.abs(slope) < 0.1 ? 'stable' : slope > 0 ? 'increasing' : 'decreasing';

    return { slope, direction };
  }

  /**
   * Detect abnormal system patterns
   */
  private async detectAbnormalPatterns(cpuTrend: any, memoryTrend: any): Promise<void> {
    // Detect sudden CPU spikes
    if (cpuTrend.direction === 'increasing' && cpuTrend.slope > 5) {
      await this.alertManager.sendAlert({
        level: 'warning',
        title: 'Sudden CPU Usage Increase',
        message: `CPU usage is increasing rapidly (slope: ${cpuTrend.slope.toFixed(2)})`,
        timestamp: new Date()
      });
    }

    // Detect memory leaks
    if (memoryTrend.direction === 'increasing' && memoryTrend.slope > 2) {
      await this.alertManager.sendAlert({
        level: 'warning',
        title: 'Potential Memory Leak Detected',
        message: `Memory usage is steadily increasing (slope: ${memoryTrend.slope.toFixed(2)})`,
        timestamp: new Date()
      });
    }
  }

  /**
   * Generate system report
   */
  private async generateSystemReport(): Promise<void> {
    if (!this.currentMetrics) return;

    try {
      const report = {
        timestamp: new Date(),
        current: this.currentMetrics,
        bottlenecks: this.bottlenecks,
        trends: await this.analyzeLongTermTrends(),
        recommendations: await this.generateSystemRecommendations(),
        performance: await this.calculatePerformanceScore()
      };

      await this.dashboardManager.updateDashboard('system', report);
      this.emit('system:report_generated', report);

    } catch (error) {
      this.logger.error('Failed to generate system report:', error);
    }
  }

  /**
   * Analyze long-term trends
   */
  private async analyzeLongTermTrends(): Promise<any> {
    if (this.historicalMetrics.length < 100) return null;

    const lastHour = this.historicalMetrics.slice(-720); // Last hour at 5-second intervals

    return {
      cpu: this.calculateTrend(lastHour.map(m => m.cpu.usage)),
      memory: this.calculateTrend(lastHour.map(m => m.memory.usagePercentage)),
      network: this.calculateTrend(lastHour.map(m => m.network.total.bytesReceived + m.network.total.bytesSent))
    };
  }

  /**
   * Generate system recommendations
   */
  private async generateSystemRecommendations(): Promise<string[]> {
    const recommendations: string[] = [];

    if (!this.currentMetrics) return recommendations;

    // CPU recommendations
    if (this.currentMetrics.cpu.usage > 80) {
      recommendations.push('Consider CPU optimization or scaling up');
    }

    // Memory recommendations
    if (this.currentMetrics.memory.usagePercentage > 75) {
      recommendations.push('Monitor memory usage for potential leaks');
    }

    // Disk recommendations
    const highUsageDisks = this.currentMetrics.disk.filter(d => d.usagePercentage > 80);
    if (highUsageDisks.length > 0) {
      recommendations.push('Consider disk cleanup or expansion');
    }

    // Temperature recommendations
    if (this.currentMetrics.temperature && this.currentMetrics.temperature.cpu > 70) {
      recommendations.push('Monitor system cooling');
    }

    return recommendations;
  }

  /**
   * Calculate overall performance score
   */
  private async calculatePerformanceScore(): Promise<number> {
    if (!this.currentMetrics) return 0;

    let score = 100;

    // CPU impact
    score -= Math.max(0, this.currentMetrics.cpu.usage - 50) * 0.5;

    // Memory impact
    score -= Math.max(0, this.currentMetrics.memory.usagePercentage - 70) * 0.3;

    // Disk impact
    const maxDiskUsage = Math.max(...this.currentMetrics.disk.map(d => d.usagePercentage));
    score -= Math.max(0, maxDiskUsage - 80) * 0.2;

    // Bottleneck impact
    const criticalBottlenecks = this.bottlenecks.filter(b => b.severity === 'critical').length;
    score -= criticalBottlenecks * 10;

    return Math.max(0, Math.min(100, Math.round(score)));
  }

  /**
   * Get current system metrics
   */
  getCurrentMetrics(): SystemMetrics | null {
    return this.currentMetrics;
  }

  /**
   * Get detected bottlenecks
   */
  getBottlenecks(): BottleneckDetection[] {
    return this.bottlenecks;
  }

  /**
   * Get historical metrics
   */
  getHistoricalMetrics(limit?: number): SystemMetrics[] {
    if (limit) {
      return this.historicalMetrics.slice(-limit);
    }
    return this.historicalMetrics;
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.isCollectingMetrics = false;
    this.logger.info('System monitoring stopped');
  }

  /**
   * Start monitoring (if stopped)
   */
  startMonitoring(): void {
    if (!this.isCollectingMetrics) {
      this.startMetricsCollection();
    }
  }
}