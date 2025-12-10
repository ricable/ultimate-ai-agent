/**
 * Provider Cost Management - Real-time cost tracking, budgets, and spending analytics
 * Provides comprehensive cost analysis across all providers with alerts and forecasting
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  Bell,
  Calculator,
  CreditCard,
  PieChart,
  BarChart3,
  Calendar,
  Target,
  Zap,
  FileText,
  Download,
  Settings
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';

interface CostAlert {
  id: string;
  type: 'budget' | 'spike' | 'forecast';
  severity: 'low' | 'medium' | 'high';
  message: string;
  provider?: string;
  timestamp: number;
  threshold: number;
  current: number;
}

interface BudgetConfig {
  providerId: string;
  daily: number;
  weekly: number;
  monthly: number;
  alertThreshold: number;
  enforceLimit: boolean;
}

const PROVIDER_LOGOS = {
  'anthropic': '🧠',
  'openai': '🤖',
  'google': '🔍',
  'groq': '⚡'
};

export const ProviderCosts: React.FC = () => {
  const { providers, providerMetrics } = useSessionStore();
  const [budgets, setBudgets] = useState<Record<string, BudgetConfig>>({});
  const [alerts, setAlerts] = useState<CostAlert[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState<'today' | 'week' | 'month' | 'year'>('month');
  const [showBudgetConfig, setShowBudgetConfig] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>('all');

  useEffect(() => {
    // Generate mock alerts
    const mockAlerts: CostAlert[] = [
      {
        id: '1',
        type: 'budget',
        severity: 'medium',
        message: 'OpenAI spending is at 80% of monthly budget',
        provider: 'openai',
        timestamp: Date.now() - 1800000,
        threshold: 100,
        current: 80
      },
      {
        id: '2',
        type: 'spike',
        severity: 'high',
        message: 'Anthropic costs increased 150% from yesterday',
        provider: 'anthropic',
        timestamp: Date.now() - 3600000,
        threshold: 120,
        current: 250
      }
    ];
    setAlerts(mockAlerts);

    // Generate default budgets
    const defaultBudgets: Record<string, BudgetConfig> = {};
    providers.forEach(provider => {
      defaultBudgets[provider.id] = {
        providerId: provider.id,
        daily: 10,
        weekly: 50,
        monthly: 200,
        alertThreshold: 80,
        enforceLimit: false
      };
    });
    setBudgets(defaultBudgets);
  }, [providers]);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const getTotalCost = (period: string) => {
    const total = providerMetrics.reduce((sum, m) => {
      switch (period) {
        case 'today':
          return sum + (m.last_24h?.cost || 0);
        case 'week':
          return sum + (m.last_24h?.cost || 0) * 7; // Mock weekly
        case 'month':
          return sum + m.total_cost;
        case 'year':
          return sum + m.total_cost * 12; // Mock yearly
        default:
          return sum + m.total_cost;
      }
    }, 0);
    return total;
  };

  const getCostTrend = (providerId?: string) => {
    const metrics = providerId 
      ? providerMetrics.filter(m => m.provider_id === providerId)
      : providerMetrics;
    
    // Mock trend calculation
    const currentCost = metrics.reduce((sum, m) => sum + (m.last_24h?.cost || 0), 0);
    const previousCost = currentCost * (0.8 + Math.random() * 0.4); // Mock previous period
    const change = ((currentCost - previousCost) / previousCost) * 100;
    
    return {
      change,
      isIncreasing: change > 0,
      amount: Math.abs(change)
    };
  };

  const getBudgetUsage = (providerId: string, period: 'daily' | 'weekly' | 'monthly') => {
    const budget = budgets[providerId];
    if (!budget) return { used: 0, total: 0, percentage: 0 };
    
    const metrics = providerMetrics.find(m => m.provider_id === providerId);
    if (!metrics) return { used: 0, total: budget[period], percentage: 0 };
    
    let used = 0;
    switch (period) {
      case 'daily':
        used = metrics.last_24h?.cost || 0;
        break;
      case 'weekly':
        used = (metrics.last_24h?.cost || 0) * 7; // Mock weekly
        break;
      case 'monthly':
        used = metrics.total_cost;
        break;
    }
    
    const total = budget[period];
    const percentage = (used / total) * 100;
    
    return { used, total, percentage };
  };

  const getAlertColor = (severity: 'low' | 'medium' | 'high') => {
    switch (severity) {
      case 'high':
        return 'bg-red-50 border-red-200 text-red-800 dark:bg-red-950 dark:border-red-800 dark:text-red-200';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-950 dark:border-yellow-800 dark:text-yellow-200';
      case 'low':
        return 'bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-200';
    }
  };

  const generateCostForecast = () => {
    const currentMonthly = getTotalCost('month');
    const trend = getCostTrend();
    const forecastMultiplier = 1 + (trend.change / 100);
    
    return {
      nextMonth: currentMonthly * forecastMultiplier,
      nextQuarter: currentMonthly * forecastMultiplier * 3,
      nextYear: currentMonthly * forecastMultiplier * 12
    };
  };

  const renderCostChart = () => {
    const chartData = providers.map(provider => {
      const metrics = providerMetrics.find(m => m.provider_id === provider.id);
      return {
        provider: provider.name,
        cost: metrics?.total_cost || 0,
        logo: PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'
      };
    }).sort((a, b) => b.cost - a.cost);

    const maxCost = Math.max(...chartData.map(d => d.cost));

    return (
      <div className="space-y-4">
        {chartData.map((item, index) => (
          <motion.div
            key={item.provider}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="space-y-2"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-lg">{item.logo}</span>
                <span className="font-medium">{item.provider}</span>
              </div>
              <span className="font-mono text-sm">{formatCurrency(item.cost)}</span>
            </div>
            <Progress 
              value={(item.cost / maxCost) * 100} 
              className="h-2"
            />
          </motion.div>
        ))}
      </div>
    );
  };

  const totalCost = getTotalCost(selectedPeriod);
  const trend = getCostTrend();
  const forecast = generateCostForecast();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <DollarSign className="h-6 w-6" />
            <span>Cost Management</span>
          </h2>
          <p className="text-muted-foreground">Track spending, manage budgets, and optimize costs</p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Select value={selectedPeriod} onValueChange={(value: any) => setSelectedPeriod(value)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">Today</SelectItem>
              <SelectItem value="week">This Week</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
              <SelectItem value="year">This Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button size="sm" onClick={() => setShowBudgetConfig(true)}>
            <Settings className="h-4 w-4 mr-2" />
            Budgets
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 border-green-200 dark:border-green-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-green-700 dark:text-green-300">Total Spent</p>
                <p className="text-2xl font-bold text-green-900 dark:text-green-100">{formatCurrency(totalCost)}</p>
                <p className="text-xs text-green-600 dark:text-green-400">
                  {selectedPeriod === 'today' ? 'Today' : 
                   selectedPeriod === 'week' ? 'This week' :
                   selectedPeriod === 'month' ? 'This month' : 'This year'}
                </p>
              </div>
              <CreditCard className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className={cn(
          "bg-gradient-to-br border-200 dark:border-800",
          trend.isIncreasing 
            ? "from-red-50 to-red-100 dark:from-red-950 dark:to-red-900 border-red-200 dark:border-red-800"
            : "from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 border-blue-200 dark:border-blue-800"
        )}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className={cn(
                  "text-sm font-medium",
                  trend.isIncreasing ? "text-red-700 dark:text-red-300" : "text-blue-700 dark:text-blue-300"
                )}>
                  Change
                </p>
                <p className={cn(
                  "text-2xl font-bold",
                  trend.isIncreasing ? "text-red-900 dark:text-red-100" : "text-blue-900 dark:text-blue-100"
                )}>
                  {trend.isIncreasing ? '+' : ''}{trend.amount.toFixed(1)}%
                </p>
                <p className={cn(
                  "text-xs",
                  trend.isIncreasing ? "text-red-600 dark:text-red-400" : "text-blue-600 dark:text-blue-400"
                )}>
                  vs last period
                </p>
              </div>
              {trend.isIncreasing ? (
                <TrendingUp className="h-8 w-8 text-red-500" />
              ) : (
                <TrendingDown className="h-8 w-8 text-blue-500" />
              )}
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 border-purple-200 dark:border-purple-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Forecast</p>
                <p className="text-2xl font-bold text-purple-900 dark:text-purple-100">
                  {formatCurrency(forecast.nextMonth)}
                </p>
                <p className="text-xs text-purple-600 dark:text-purple-400">Next month</p>
              </div>
              <Calculator className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-950 dark:to-orange-900 border-orange-200 dark:border-orange-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-orange-700 dark:text-orange-300">Active Alerts</p>
                <p className="text-2xl font-bold text-orange-900 dark:text-orange-100">{alerts.length}</p>
                <p className="text-xs text-orange-600 dark:text-orange-400">Budget & usage</p>
              </div>
              <Bell className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Alerts */}
      {alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Active Cost Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {alerts.map(alert => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn("p-4 rounded-lg border flex items-center justify-between", getAlertColor(alert.severity))}
              >
                <div className="flex items-center space-x-3">
                  {alert.provider && (
                    <span className="text-xl">
                      {PROVIDER_LOGOS[alert.provider as keyof typeof PROVIDER_LOGOS] || '🤖'}
                    </span>
                  )}
                  <div>
                    <div className="font-medium">{alert.message}</div>
                    <div className="text-sm opacity-75">
                      {alert.type === 'budget' && `${alert.current}% of budget used`}
                      {alert.type === 'spike' && `${alert.current}% increase detected`}
                      {alert.type === 'forecast' && `Projected to exceed by ${alert.current}%`}
                    </div>
                  </div>
                </div>
                <div className="text-sm opacity-75">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </div>
              </motion.div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <PieChart className="h-5 w-5" />
              <span>Cost Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {renderCostChart()}
          </CardContent>
        </Card>

        {/* Budget Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Budget Overview</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {providers.map(provider => {
              const monthlyUsage = getBudgetUsage(provider.id, 'monthly');
              
              return (
                <div key={provider.id} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                      </span>
                      <span className="font-medium">{provider.name}</span>
                    </div>
                    <div className="text-sm">
                      {formatCurrency(monthlyUsage.used)} / {formatCurrency(monthlyUsage.total)}
                    </div>
                  </div>
                  <Progress 
                    value={monthlyUsage.percentage} 
                    className={cn(
                      "h-2",
                      monthlyUsage.percentage > 90 && "bg-red-100",
                      monthlyUsage.percentage > 75 && monthlyUsage.percentage <= 90 && "bg-yellow-100"
                    )}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{monthlyUsage.percentage.toFixed(1)}% used</span>
                    <span>{formatCurrency(monthlyUsage.total - monthlyUsage.used)} remaining</span>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      </div>

      {/* Budget Configuration Modal */}
      {showBudgetConfig && (
        <Card className="fixed inset-4 z-50 bg-background border shadow-lg overflow-y-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Budget Configuration</span>
              </CardTitle>
              <Button variant="ghost" onClick={() => setShowBudgetConfig(false)}>
                ✕
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {providers.map(provider => {
              const budget = budgets[provider.id];
              if (!budget) return null;

              return (
                <Card key={provider.id}>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <span className="text-xl">
                        {PROVIDER_LOGOS[provider.id as keyof typeof PROVIDER_LOGOS] || '🤖'}
                      </span>
                      <span>{provider.name}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <Label>Daily Budget</Label>
                        <Input
                          type="number"
                          value={budget.daily}
                          onChange={(e) => setBudgets(prev => ({
                            ...prev,
                            [provider.id]: { ...budget, daily: parseFloat(e.target.value) }
                          }))}
                        />
                      </div>
                      <div>
                        <Label>Weekly Budget</Label>
                        <Input
                          type="number"
                          value={budget.weekly}
                          onChange={(e) => setBudgets(prev => ({
                            ...prev,
                            [provider.id]: { ...budget, weekly: parseFloat(e.target.value) }
                          }))}
                        />
                      </div>
                      <div>
                        <Label>Monthly Budget</Label>
                        <Input
                          type="number"
                          value={budget.monthly}
                          onChange={(e) => setBudgets(prev => ({
                            ...prev,
                            [provider.id]: { ...budget, monthly: parseFloat(e.target.value) }
                          }))}
                        />
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="space-y-2">
                        <Label>Alert Threshold (%)</Label>
                        <Input
                          type="number"
                          min="1"
                          max="100"
                          value={budget.alertThreshold}
                          onChange={(e) => setBudgets(prev => ({
                            ...prev,
                            [provider.id]: { ...budget, alertThreshold: parseInt(e.target.value) }
                          }))}
                          className="w-24"
                        />
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Switch
                          checked={budget.enforceLimit}
                          onCheckedChange={(checked) => setBudgets(prev => ({
                            ...prev,
                            [provider.id]: { ...budget, enforceLimit: checked }
                          }))}
                        />
                        <Label>Enforce hard limits</Label>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
            
            <div className="flex justify-end space-x-2 pt-4 border-t border-border">
              <Button variant="outline" onClick={() => setShowBudgetConfig(false)}>
                Cancel
              </Button>
              <Button onClick={() => setShowBudgetConfig(false)}>
                Save Budgets
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};