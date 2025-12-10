/**
 * Provider Dashboard - Comprehensive multi-provider management interface
 */

"use client";

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Server, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  TrendingUp,
  DollarSign,
  Clock,
  Activity,
  Settings,
  Plus,
  RefreshCw,
  Zap,
  Shield,
  Target,
  BarChart3,
  Globe,
  Cpu,
  Brain,
  Lightbulb,
  Bot,
  Code,
  Image,
  FileText,
  MessageSquare,
  Search,
  Filter,
  SortAsc,
  Eye,
  Key,
  Wifi,
  WifiOff,
  AlertCircle,
  Command,
  ExternalLink
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useSessionStore } from '@/lib/session-store';

export const ProviderDashboard = () => {
  const {
    providers,
    providerHealth,
    providerMetrics,
    isLoadingProviders,
    activeProvider,
    actions
  } = useSessionStore();

  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'performance'>('grid');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    actions.loadProviders();
    actions.loadProviderHealth();
    actions.loadProviderMetrics();
  }, []);

  if (isLoadingProviders) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  const filteredProviders = providers.filter(provider => 
    provider.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Provider Dashboard</h1>
            <p className="text-muted-foreground">
              Monitor and manage your AI provider connections
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm" onClick={() => actions.loadProviders()}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button size="sm">
              <Plus className="h-4 w-4 mr-2" />
              Add Provider
            </Button>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center space-x-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search providers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="grid">Grid</SelectItem>
              <SelectItem value="list">List</SelectItem>
              <SelectItem value="performance">Performance</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-6">
        <Tabs value={viewMode} onValueChange={(value: any) => setViewMode(value)}>
          <TabsList className="mb-6">
            <TabsTrigger value="grid">Grid View</TabsTrigger>
            <TabsTrigger value="list">List View</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
          </TabsList>

          <TabsContent value="grid" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredProviders.map((provider) => {
                const health = providerHealth.find(h => h.provider_id === provider.id);
                const metrics = providerMetrics.find(m => m.provider_id === provider.id);
                
                return (
                  <Card key={provider.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{provider.name}</CardTitle>
                        <Badge variant={health?.status === 'online' ? 'default' : 'destructive'}>
                          {health?.status === 'online' ? (
                            <CheckCircle className="h-3 w-3 mr-1" />
                          ) : (
                            <XCircle className="h-3 w-3 mr-1" />
                          )}
                          {health?.status || 'Unknown'}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Models:</span>
                          <span>{provider.models.length}</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Uptime:</span>
                          <span>{health?.uptime || 0}%</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Avg Response:</span>
                          <span>{provider.avg_response_time}ms</span>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Cost/1K:</span>
                          <span>${provider.cost_per_1k_tokens}</span>
                        </div>
                      </div>
                      <div className="mt-4 flex space-x-2">
                        <Button variant="outline" size="sm" className="flex-1">
                          <Settings className="h-4 w-4 mr-1" />
                          Configure
                        </Button>
                        <Button 
                          size="sm" 
                          className="flex-1"
                          variant={activeProvider === provider.id ? "default" : "outline"}
                        >
                          {activeProvider === provider.id ? "Active" : "Select"}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          <TabsContent value="list" className="space-y-4">
            <div className="space-y-2">
              {filteredProviders.map((provider) => {
                const health = providerHealth.find(h => h.provider_id === provider.id);
                
                return (
                  <Card key={provider.id}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="flex items-center space-x-2">
                            <Server className="h-5 w-5" />
                            <div>
                              <h3 className="font-medium">{provider.name}</h3>
                              <p className="text-sm text-muted-foreground">
                                {provider.models.length} models
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <Badge variant={health?.status === 'online' ? 'default' : 'destructive'}>
                            {health?.status || 'Unknown'}
                          </Badge>
                          <div className="text-sm text-muted-foreground">
                            {provider.avg_response_time}ms
                          </div>
                          <div className="text-sm font-medium">
                            ${provider.cost_per_1k_tokens}/1K
                          </div>
                          <Button 
                            size="sm"
                            variant={activeProvider === provider.id ? "default" : "outline"}
                          >
                            {activeProvider === provider.id ? "Active" : "Select"}
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Server className="h-5 w-5 text-blue-500" />
                    <div>
                      <p className="text-sm font-medium">Total Providers</p>
                      <p className="text-2xl font-bold">{providers.length}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <div>
                      <p className="text-sm font-medium">Online</p>
                      <p className="text-2xl font-bold">
                        {providerHealth.filter(h => h.status === 'online').length}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-5 w-5 text-yellow-500" />
                    <div>
                      <p className="text-sm font-medium">Avg Response</p>
                      <p className="text-2xl font-bold">
                        {Math.round(providers.reduce((sum, p) => sum + p.avg_response_time, 0) / providers.length)}ms
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center space-x-2">
                    <DollarSign className="h-5 w-5 text-green-500" />
                    <div>
                      <p className="text-sm font-medium">Total Cost</p>
                      <p className="text-2xl font-bold">
                        ${providerMetrics.reduce((sum, m) => sum + m.total_cost, 0).toFixed(2)}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};