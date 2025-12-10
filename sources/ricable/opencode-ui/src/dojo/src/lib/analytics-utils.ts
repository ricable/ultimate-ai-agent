/**
 * Analytics Utilities - Data processing and export functions for usage dashboard
 */

import { type Session, type ProviderMetrics, type Message } from './opencode-client';

export interface TimeSeriesDataPoint {
  timestamp: number;
  date: string;
  tokens: number;
  cost: number;
  sessions: number;
  inputTokens: number;
  outputTokens: number;
  cacheTokens: number;
}

export interface ProviderBreakdown {
  provider: string;
  model: string;
  sessions: number;
  tokens: number;
  cost: number;
  percentage: number;
  color: string;
}

export interface ProjectAnalytics {
  name: string;
  path: string;
  sessions: number;
  totalTokens: number;
  totalCost: number;
  avgSessionDuration: number;
  lastUsed: number;
}

export interface CacheAnalytics {
  hitRate: number;
  missRate: number;
  totalReads: number;
  totalWrites: number;
  savings: number;
}

export interface UsageTimeframe {
  label: string;
  days: number;
  getValue: (date: Date) => Date;
}

export const TIME_FRAMES: UsageTimeframe[] = [
  {
    label: "Last 7 Days",
    days: 7,
    getValue: (date: Date) => new Date(date.getTime() - 7 * 24 * 60 * 60 * 1000)
  },
  {
    label: "Last 30 Days", 
    days: 30,
    getValue: (date: Date) => new Date(date.getTime() - 30 * 24 * 60 * 60 * 1000)
  },
  {
    label: "Last 90 Days",
    days: 90,
    getValue: (date: Date) => new Date(date.getTime() - 90 * 24 * 60 * 60 * 1000)
  },
  {
    label: "All Time",
    days: 365,
    getValue: () => new Date(0)
  }
];

// Provider color mapping for consistent visualization
export const PROVIDER_COLORS: Record<string, string> = {
  'anthropic': '#FF6B35',
  'openai': '#10A37F', 
  'google': '#4285F4',
  'groq': '#FF8C00',
  'cohere': '#39C5BB',
  'mistral': '#FF7139',
  'perplexity': '#1FB6FF',
  'replicate': '#000000',
  'huggingface': '#FFD21E',
  'local': '#8B5CF6',
  'other': '#6B7280'
};

export const MODEL_COLORS: Record<string, string> = {
  'claude-3.5-sonnet': '#3B82F6',
  'claude-3.5-haiku': '#10B981',
  'claude-3-opus': '#8B5CF6',
  'gpt-4o': '#F59E0B',
  'gpt-4-turbo': '#EF4444',
  'gpt-3.5-turbo': '#06B6D4',
  'gemini-pro': '#84CC16',
  'llama-2': '#F97316',
  'default': '#6B7280'
};

/**
 * Process sessions into time series data for trend analysis
 */
export function processTimeSeriesData(
  sessions: Session[],
  timeframe: UsageTimeframe
): TimeSeriesDataPoint[] {
  const now = new Date();
  const startDate = timeframe.getValue(now);
  
  // Filter sessions within timeframe
  const filteredSessions = sessions.filter(session => {
    const sessionDate = new Date(session.created_at || 0);
    return sessionDate >= startDate;
  });

  // Group by day
  const dailyData = new Map<string, {
    tokens: number;
    cost: number;
    sessions: number;
    inputTokens: number;
    outputTokens: number;
    cacheTokens: number;
  }>();

  filteredSessions.forEach(session => {
    const date = new Date(session.created_at || 0);
    const dateKey = date.toISOString().split('T')[0];
    
    const existing = dailyData.get(dateKey) || {
      tokens: 0,
      cost: 0,
      sessions: 0,
      inputTokens: 0,
      outputTokens: 0,
      cacheTokens: 0
    };

    const tokenUsage = session.token_usage || { input_tokens: 0, output_tokens: 0, cache_tokens: 0 };
    const sessionCost = calculateSessionCost(tokenUsage, session.model || '');

    dailyData.set(dateKey, {
      tokens: existing.tokens + tokenUsage.input_tokens + tokenUsage.output_tokens,
      cost: existing.cost + sessionCost,
      sessions: existing.sessions + 1,
      inputTokens: existing.inputTokens + tokenUsage.input_tokens,
      outputTokens: existing.outputTokens + tokenUsage.output_tokens,
      cacheTokens: existing.cacheTokens + ((tokenUsage as any).cache_tokens || 0)
    });
  });

  // Convert to array and fill missing days
  const result: TimeSeriesDataPoint[] = [];
  const currentDate = new Date(startDate);
  
  while (currentDate <= now) {
    const dateKey = currentDate.toISOString().split('T')[0];
    const data = dailyData.get(dateKey) || {
      tokens: 0,
      cost: 0,
      sessions: 0,
      inputTokens: 0,
      outputTokens: 0,
      cacheTokens: 0
    };

    result.push({
      timestamp: currentDate.getTime(),
      date: dateKey,
      ...data
    });

    currentDate.setDate(currentDate.getDate() + 1);
  }

  return result.sort((a, b) => a.timestamp - b.timestamp);
}

/**
 * Calculate cost breakdown by provider and model
 */
export function processProviderBreakdown(
  sessions: Session[],
  providerMetrics: ProviderMetrics[]
): ProviderBreakdown[] {
  const breakdown = new Map<string, {
    provider: string;
    model: string;
    sessions: number;
    tokens: number;
    cost: number;
  }>();

  sessions.forEach(session => {
    const provider = session.provider || 'unknown';
    const model = session.model || 'unknown';
    const key = `${provider}::${model}`;
    
    const existing = breakdown.get(key) || {
      provider,
      model,
      sessions: 0,
      tokens: 0,
      cost: 0
    };

    const tokenUsage = session.token_usage || { input_tokens: 0, output_tokens: 0, cache_tokens: 0 };
    const sessionCost = calculateSessionCost(tokenUsage, model);
    const sessionTokens = tokenUsage.input_tokens + tokenUsage.output_tokens;

    breakdown.set(key, {
      ...existing,
      sessions: existing.sessions + 1,
      tokens: existing.tokens + sessionTokens,
      cost: existing.cost + sessionCost
    });
  });

  const totalCost = Array.from(breakdown.values()).reduce((sum, item) => sum + item.cost, 0);
  
  return Array.from(breakdown.values())
    .map(item => ({
      ...item,
      percentage: totalCost > 0 ? (item.cost / totalCost) * 100 : 0,
      color: getProviderColor(item.provider, item.model)
    }))
    .sort((a, b) => b.cost - a.cost);
}

/**
 * Analyze project usage patterns
 */
export function processProjectAnalytics(sessions: Session[]): ProjectAnalytics[] {
  const projectData = new Map<string, {
    name: string;
    path: string;
    sessions: Session[];
    totalTokens: number;
    totalCost: number;
    lastUsed: number;
  }>();

  sessions.forEach(session => {
    const path = session.project_path || 'Unknown Project';
    const name = path.split('/').pop() || path;
    
    const existing = projectData.get(path) || {
      name,
      path,
      sessions: [],
      totalTokens: 0,
      totalCost: 0,
      lastUsed: 0
    };

    const tokenUsage = session.token_usage || { input_tokens: 0, output_tokens: 0, cache_tokens: 0 };
    const sessionCost = calculateSessionCost(tokenUsage, session.model || '');
    const sessionTokens = tokenUsage.input_tokens + tokenUsage.output_tokens;

    projectData.set(path, {
      ...existing,
      sessions: [...existing.sessions, session],
      totalTokens: existing.totalTokens + sessionTokens,
      totalCost: existing.totalCost + sessionCost,
      lastUsed: Math.max(existing.lastUsed, session.created_at || 0)
    });
  });

  return Array.from(projectData.values())
    .map(project => ({
      name: project.name,
      path: project.path,
      sessions: project.sessions.length,
      totalTokens: project.totalTokens,
      totalCost: project.totalCost,
      avgSessionDuration: calculateAvgSessionDuration(project.sessions),
      lastUsed: project.lastUsed
    }))
    .sort((a, b) => b.totalCost - a.totalCost);
}

/**
 * Calculate cache analytics
 */
export function processCacheAnalytics(sessions: Session[]): CacheAnalytics {
  let totalCacheReads = 0;
  let totalCacheWrites = 0;
  let totalTokensWithoutCache = 0;
  let totalCacheTokens = 0;

  sessions.forEach(session => {
    const tokenUsage = session.token_usage;
    if (tokenUsage) {
      const cacheTokens = (tokenUsage as any).cache_tokens || 0;
      totalCacheTokens += cacheTokens;
      totalCacheReads += cacheTokens * 0.6; // Estimate 60% are reads
      totalCacheWrites += cacheTokens * 0.4; // Estimate 40% are writes
      totalTokensWithoutCache += tokenUsage.input_tokens + tokenUsage.output_tokens;
    }
  });

  const totalTokens = totalTokensWithoutCache + totalCacheTokens;
  const hitRate = totalTokens > 0 ? (totalCacheTokens / totalTokens) * 100 : 0;
  const missRate = 100 - hitRate;
  
  // Estimate savings (cache tokens typically cost 1/10th of regular tokens)
  const savings = totalCacheTokens * 0.00001 * 0.9; // 90% savings on cache hits

  return {
    hitRate,
    missRate,
    totalReads: totalCacheReads,
    totalWrites: totalCacheWrites,
    savings
  };
}

/**
 * Export data to CSV format
 */
export function exportToCSV(data: any[], filename: string): void {
  if (data.length === 0) return;

  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        if (typeof value === 'string' && value.includes(',')) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',')
    )
  ].join('\n');

  downloadFile(csvContent, `${filename}.csv`, 'text/csv');
}

/**
 * Export data to JSON format
 */
export function exportToJSON(data: any, filename: string): void {
  const jsonContent = JSON.stringify(data, null, 2);
  downloadFile(jsonContent, `${filename}.json`, 'application/json');
}

/**
 * Generate PDF report (basic implementation)
 */
export function exportToPDF(
  summary: any,
  charts: { title: string; data: any[] }[],
  filename: string
): void {
  // For a complete PDF implementation, you'd use a library like jsPDF
  // This is a simplified version that creates a readable text report
  const content = [
    'OpenCode Usage Analytics Report',
    '='.repeat(50),
    '',
    'Summary:',
    JSON.stringify(summary, null, 2),
    '',
    'Charts Data:',
    ...charts.map(chart => [
      `\n${chart.title}:`,
      '-'.repeat(chart.title.length + 1),
      JSON.stringify(chart.data, null, 2)
    ].join('\n'))
  ].join('\n');

  downloadFile(content, `${filename}.txt`, 'text/plain');
}

// Helper functions

function calculateSessionCost(
  tokenUsage: { input_tokens: number; output_tokens: number; cache_tokens?: number },
  model: string
): number {
  // Simplified cost calculation - in reality this would use actual pricing
  const rates = getModelRates(model);
  const inputCost = (tokenUsage.input_tokens / 1000) * rates.input;
  const outputCost = (tokenUsage.output_tokens / 1000) * rates.output;
  const cacheCost = ((tokenUsage.cache_tokens || 0) / 1000) * rates.cache;
  
  return inputCost + outputCost + cacheCost;
}

function getModelRates(model: string): { input: number; output: number; cache: number } {
  // Simplified rate mapping - actual implementation would have comprehensive pricing
  if (model.includes('claude-3.5-sonnet')) {
    return { input: 0.003, output: 0.015, cache: 0.00003 };
  } else if (model.includes('claude-3-opus')) {
    return { input: 0.015, output: 0.075, cache: 0.00015 };
  } else if (model.includes('gpt-4')) {
    return { input: 0.03, output: 0.06, cache: 0.0003 };
  } else if (model.includes('gpt-3.5')) {
    return { input: 0.001, output: 0.002, cache: 0.00001 };
  }
  
  return { input: 0.01, output: 0.02, cache: 0.0001 }; // Default rates
}

function getProviderColor(provider: string, model: string): string {
  return MODEL_COLORS[model] || PROVIDER_COLORS[provider] || PROVIDER_COLORS.other;
}

function calculateAvgSessionDuration(sessions: Session[]): number {
  if (sessions.length === 0) return 0;
  
  const durations = sessions
    .filter(s => s.created_at && s.updated_at)
    .map(s => (s.updated_at! - s.created_at!) / 1000 / 60); // Duration in minutes
    
  return durations.length > 0 
    ? durations.reduce((sum, d) => sum + d, 0) / durations.length 
    : 0;
}

function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Format numbers for display
 */
export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('en-US').format(num);
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 4
  }).format(amount);
};

export const formatTokens = (num: number): string => {
  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(2)}M`;
  } else if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}K`;
  }
  return formatNumber(num);
};

export const formatPercentage = (num: number): string => {
  return `${num.toFixed(1)}%`;
};

export const formatDuration = (minutes: number): string => {
  if (minutes < 60) {
    return `${Math.round(minutes)}m`;
  } else if (minutes < 1440) {
    return `${Math.round(minutes / 60)}h ${Math.round(minutes % 60)}m`;
  } else {
    return `${Math.round(minutes / 1440)}d ${Math.round((minutes % 1440) / 60)}h`;
  }
};