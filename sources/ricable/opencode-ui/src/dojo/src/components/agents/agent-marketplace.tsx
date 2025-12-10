"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Filter,
  Star,
  Download,
  Heart,
  Eye,
  TrendingUp,
  Award,
  Clock,
  Users,
  Code,
  Database,
  Zap,
  TestTube,
  Globe,
  BookOpen,
  Shield,
  Settings,
  ArrowLeft,
  Plus,
  ExternalLink,
  Share2,
  Flag,
  ThumbsUp,
  MessageSquare,
  Tag,
  Calendar
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";
import { openCodeClient } from "@/lib/opencode-client";
import type { 
  Agent,
  AgentMarketplace,
  AgentTemplate,
  AgentCategory,
  AgentFeedback
} from "@/types/opencode";

interface AgentMarketplaceProps {
  onAgentInstall?: (agent: Agent) => void;
  onClose?: () => void;
}

type ViewType = "featured" | "categories" | "trending" | "new" | "top-rated";

export function AgentMarketplace({ onAgentInstall, onClose }: AgentMarketplaceProps) {
  const [marketplace, setMarketplace] = useState<AgentMarketplace | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [agentFeedback, setAgentFeedback] = useState<AgentFeedback[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<AgentCategory | "all">("all");
  const [selectedView, setSelectedView] = useState<ViewType>("featured");
  const [sortBy, setSortBy] = useState<"rating" | "downloads" | "recent" | "name">("rating");
  const [showDetails, setShowDetails] = useState(false);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    loadMarketplace();
  }, []);

  const loadMarketplace = async () => {
    try {
      setLoading(true);
      const marketplaceData = await openCodeClient.getAgentMarketplace();
      setMarketplace(marketplaceData || getMockMarketplace());
    } catch (error) {
      console.error("Failed to load marketplace:", error);
      setMarketplace(getMockMarketplace());
    } finally {
      setLoading(false);
    }
  };

  const loadAgentDetails = async (agent: Agent) => {
    try {
      const feedback = await openCodeClient.getAgentFeedback(agent.id);
      setAgentFeedback(feedback);
    } catch (error) {
      console.error("Failed to load agent feedback:", error);
      setAgentFeedback(getMockFeedback());
    }
  };

  const handleInstallAgent = async (agent: Agent) => {
    try {
      // In a real implementation, this would download and install the agent
      // For now, we'll just call the callback
      if (onAgentInstall) {
        onAgentInstall(agent);
      }
      
      toast({
        title: "Success",
        description: `Agent "${agent.name}" installed successfully`
      });
    } catch (error) {
      console.error("Failed to install agent:", error);
      toast({
        title: "Error",
        description: "Failed to install agent",
        variant: "destructive"
      });
    }
  };

  const handleViewAgent = (agent: Agent) => {
    setSelectedAgent(agent);
    setShowDetails(true);
    loadAgentDetails(agent);
  };

  const getAgentsByView = (view: ViewType): Agent[] => {
    if (!marketplace) return [];
    
    switch (view) {
      case "featured":
        return marketplace.featured_agents;
      case "trending":
        return marketplace.trending_agents;
      case "new":
        return marketplace.new_agents;
      case "top-rated":
        return marketplace.top_rated_agents;
      case "categories":
        return marketplace.featured_agents; // Default to featured for categories view
      default:
        return marketplace.featured_agents;
    }
  };

  const filteredAgents = getAgentsByView(selectedView).filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      agent.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = selectedCategory === "all" || agent.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  }).sort((a, b) => {
    switch (sortBy) {
      case "rating":
        return b.rating - a.rating;
      case "downloads":
        return b.usage_count - a.usage_count;
      case "recent":
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      case "name":
        return a.name.localeCompare(b.name);
      default:
        return 0;
    }
  });

  const getCategoryIcon = (category: AgentCategory) => {
    const icons = {
      coding: Code,
      "data-analysis": Database,
      writing: BookOpen,
      research: Search,
      automation: Zap,
      testing: TestTube,
      deployment: Globe,
      documentation: BookOpen,
      security: Shield,
      custom: Settings
    };
    return icons[category] || Code;
  };

  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">Agent Marketplace</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(9)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-4">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-16 bg-gray-200 rounded mb-2"></div>
                <div className="h-4 bg-gray-200 rounded"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
          )}
          <div>
            <h2 className="text-2xl font-bold">Agent Marketplace</h2>
            <p className="text-muted-foreground">Discover and install AI agents from the community</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Publish Agent
          </Button>
          <Button>
            <Heart className="h-4 w-4 mr-2" />
            My Favorites
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search agents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Select value={selectedCategory} onValueChange={(value) => setSelectedCategory(value as AgentCategory | "all")}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="All Categories" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            <SelectItem value="coding">Coding</SelectItem>
            <SelectItem value="data-analysis">Data Analysis</SelectItem>
            <SelectItem value="writing">Writing</SelectItem>
            <SelectItem value="research">Research</SelectItem>
            <SelectItem value="automation">Automation</SelectItem>
            <SelectItem value="testing">Testing</SelectItem>
            <SelectItem value="deployment">Deployment</SelectItem>
            <SelectItem value="documentation">Documentation</SelectItem>
            <SelectItem value="security">Security</SelectItem>
            <SelectItem value="custom">Custom</SelectItem>
          </SelectContent>
        </Select>
        <Select value={sortBy} onValueChange={(value) => setSortBy(value as typeof sortBy)}>
          <SelectTrigger className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="rating">Top Rated</SelectItem>
            <SelectItem value="downloads">Most Downloaded</SelectItem>
            <SelectItem value="recent">Recently Updated</SelectItem>
            <SelectItem value="name">Name</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Tabs value={selectedView} onValueChange={(value) => setSelectedView(value as ViewType)} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="featured">Featured</TabsTrigger>
          <TabsTrigger value="trending">Trending</TabsTrigger>
          <TabsTrigger value="new">New</TabsTrigger>
          <TabsTrigger value="top-rated">Top Rated</TabsTrigger>
          <TabsTrigger value="categories">Categories</TabsTrigger>
        </TabsList>

        <TabsContent value="categories" className="space-y-6">
          {/* Categories Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {marketplace?.categories.map((category) => {
              const IconComponent = getCategoryIcon(category.id);
              return (
                <motion.div
                  key={category.id}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={() => {
                    setSelectedCategory(category.id);
                    setSelectedView("featured");
                  }}>
                    <CardContent className="p-4 text-center">
                      <IconComponent className="h-8 w-8 mx-auto mb-2 text-primary" />
                      <h3 className="font-medium">{category.name}</h3>
                      <p className="text-xs text-muted-foreground">{category.agent_count} agents</p>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="featured" className="space-y-6">
          <AgentGrid agents={filteredAgents} onViewAgent={handleViewAgent} onInstallAgent={handleInstallAgent} />
        </TabsContent>

        <TabsContent value="trending" className="space-y-6">
          <AgentGrid agents={filteredAgents} onViewAgent={handleViewAgent} onInstallAgent={handleInstallAgent} />
        </TabsContent>

        <TabsContent value="new" className="space-y-6">
          <AgentGrid agents={filteredAgents} onViewAgent={handleViewAgent} onInstallAgent={handleInstallAgent} />
        </TabsContent>

        <TabsContent value="top-rated" className="space-y-6">
          <AgentGrid agents={filteredAgents} onViewAgent={handleViewAgent} onInstallAgent={handleInstallAgent} />
        </TabsContent>
      </Tabs>

      {/* Agent Details Modal */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          {selectedAgent && (
            <div className="space-y-6">
              <DialogHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="text-4xl">{selectedAgent.icon}</div>
                    <div>
                      <DialogTitle className="text-2xl">{selectedAgent.name}</DialogTitle>
                      <p className="text-muted-foreground">by {selectedAgent.created_by}</p>
                      <div className="flex items-center space-x-4 mt-2">
                        <div className="flex items-center space-x-1">
                          <Star className="h-4 w-4 text-yellow-500" />
                          <span className="font-medium">{selectedAgent.rating.toFixed(1)}</span>
                          <span className="text-muted-foreground">({agentFeedback.length} reviews)</span>
                        </div>
                        <Badge variant="secondary">{selectedAgent.category}</Badge>
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Download className="h-4 w-4" />
                          <span>{selectedAgent.usage_count.toLocaleString()} downloads</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="icon">
                      <Heart className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="icon">
                      <Share2 className="h-4 w-4" />
                    </Button>
                    <Button onClick={() => handleInstallAgent(selectedAgent)}>
                      <Download className="h-4 w-4 mr-2" />
                      Install
                    </Button>
                  </div>
                </div>
              </DialogHeader>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                  {/* Description */}
                  <div>
                    <h3 className="font-semibold mb-2">Description</h3>
                    <p className="text-muted-foreground">{selectedAgent.description}</p>
                  </div>

                  {/* System Prompt Preview */}
                  <div>
                    <h3 className="font-semibold mb-2">System Prompt</h3>
                    <div className="bg-muted rounded-lg p-4">
                      <pre className="text-sm whitespace-pre-wrap">{selectedAgent.system_prompt}</pre>
                    </div>
                  </div>

                  {/* Tools and Capabilities */}
                  <div>
                    <h3 className="font-semibold mb-2">Tools & Capabilities</h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedAgent.tools.map(tool => (
                        <Badge key={tool} variant="outline">{tool}</Badge>
                      ))}
                    </div>
                  </div>

                  {/* Tags */}
                  {selectedAgent.tags.length > 0 && (
                    <div>
                      <h3 className="font-semibold mb-2">Tags</h3>
                      <div className="flex flex-wrap gap-2">
                        {selectedAgent.tags.map(tag => (
                          <Badge key={tag} variant="secondary">
                            <Tag className="h-3 w-3 mr-1" />
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Reviews */}
                  <div>
                    <h3 className="font-semibold mb-4">Reviews ({agentFeedback.length})</h3>
                    <div className="space-y-4">
                      {agentFeedback.slice(0, 3).map((feedback) => (
                        <div key={feedback.id} className="border rounded-lg p-4">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <Avatar className="h-6 w-6">
                                <AvatarFallback>{feedback.user_id.charAt(0).toUpperCase()}</AvatarFallback>
                              </Avatar>
                              <span className="font-medium">{feedback.user_id}</span>
                              <div className="flex items-center">
                                {[...Array(5)].map((_, i) => (
                                  <Star
                                    key={i}
                                    className={`h-3 w-3 ${i < feedback.rating ? "text-yellow-500 fill-current" : "text-gray-300"}`}
                                  />
                                ))}
                              </div>
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {new Date(feedback.created_at).toLocaleDateString()}
                            </span>
                          </div>
                          {feedback.comment && (
                            <p className="text-sm text-muted-foreground">{feedback.comment}</p>
                          )}
                          <div className="flex items-center space-x-4 mt-2">
                            <Button variant="ghost" size="sm">
                              <ThumbsUp className="h-3 w-3 mr-1" />
                              {feedback.helpful_count}
                            </Button>
                            <Button variant="ghost" size="sm">
                              <MessageSquare className="h-3 w-3 mr-1" />
                              Reply
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  {/* Agent Info */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Agent Information</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Version</span>
                        <span className="text-sm font-medium">{selectedAgent.version}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Provider</span>
                        <Badge variant="outline">{selectedAgent.provider}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Model</span>
                        <span className="text-sm font-medium">{selectedAgent.model}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Created</span>
                        <span className="text-sm">{new Date(selectedAgent.created_at).toLocaleDateString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Updated</span>
                        <span className="text-sm">{new Date(selectedAgent.updated_at).toLocaleDateString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Sandbox</span>
                        <Badge variant={selectedAgent.sandbox_enabled ? "default" : "destructive"}>
                          {selectedAgent.sandbox_enabled ? "Enabled" : "Disabled"}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Permissions */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Permissions</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {Object.entries(selectedAgent.permissions).map(([key, value]) => (
                          <div key={key} className="flex items-center justify-between">
                            <span className="text-sm capitalize">{key.replace(/_/g, " ")}</span>
                            <Badge variant={value ? "default" : "secondary"}>
                              {value ? "Allowed" : "Denied"}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Actions */}
                  <div className="space-y-2">
                    <Button className="w-full" onClick={() => handleInstallAgent(selectedAgent)}>
                      <Download className="h-4 w-4 mr-2" />
                      Install Agent
                    </Button>
                    <Button variant="outline" className="w-full">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      View Source
                    </Button>
                    <Button variant="outline" className="w-full">
                      <Flag className="h-4 w-4 mr-2" />
                      Report Issue
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Agent Grid Component
interface AgentGridProps {
  agents: Agent[];
  onViewAgent: (agent: Agent) => void;
  onInstallAgent: (agent: Agent) => void;
}

function AgentGrid({ agents, onViewAgent, onInstallAgent }: AgentGridProps) {
  if (agents.length === 0) {
    return (
      <div className="text-center py-12">
        <Search className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-medium mb-2">No agents found</h3>
        <p className="text-muted-foreground">Try adjusting your search or filters</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <AnimatePresence>
        {agents.map((agent) => (
          <motion.div
            key={agent.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">{agent.icon}</div>
                    <div>
                      <CardTitle className="text-base font-semibold">{agent.name}</CardTitle>
                      <p className="text-xs text-muted-foreground">by {agent.created_by}</p>
                    </div>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => onViewAgent(agent)}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                  {agent.description}
                </p>
                
                <div className="flex items-center space-x-4 mb-4 text-xs">
                  <div className="flex items-center space-x-1">
                    <Star className="h-3 w-3 text-yellow-500" />
                    <span>{agent.rating.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Download className="h-3 w-3 text-blue-500" />
                    <span>{agent.usage_count.toLocaleString()}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3 text-gray-500" />
                    <span>{new Date(agent.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1 mb-4">
                  <Badge variant="secondary" className="text-xs">{agent.category}</Badge>
                  <Badge variant="outline" className="text-xs">{agent.provider}</Badge>
                  {agent.sandbox_enabled && (
                    <Badge variant="outline" className="text-xs">
                      <Shield className="h-2 w-2 mr-1" />
                      Secure
                    </Badge>
                  )}
                </div>

                <div className="flex space-x-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1"
                    onClick={() => onViewAgent(agent)}
                  >
                    <Eye className="h-3 w-3 mr-1" />
                    View
                  </Button>
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={() => onInstallAgent(agent)}
                  >
                    <Download className="h-3 w-3 mr-1" />
                    Install
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// Mock data functions
function getMockMarketplace(): AgentMarketplace {
  const featuredAgents = [
    {
      id: "marketplace-1",
      name: "TypeScript Expert",
      description: "Advanced TypeScript development assistant with deep knowledge of modern patterns and best practices",
      icon: "🔧",
      system_prompt: "You are a TypeScript expert. Help users write type-safe, maintainable TypeScript code following modern best practices.",
      default_task: "Review TypeScript code and suggest improvements",
      model: "claude-3.5-sonnet",
      provider: "anthropic",
      temperature: 0.3,
      max_tokens: 4000,
      tools: ["file_read", "file_write", "typescript"],
      capabilities: [],
      sandbox_enabled: true,
      permissions: {
        file_read: true,
        file_write: true,
        network_access: false,
        system_commands: false,
        environment_access: false,
        package_management: true,
        custom_tools: []
      },
      created_at: "2024-01-10T10:00:00Z",
      updated_at: "2024-01-20T15:30:00Z",
      created_by: "typescript-guru",
      tags: ["typescript", "javascript", "development", "types"],
      category: "coding" as AgentCategory,
      is_public: true,
      usage_count: 1250,
      rating: 4.9,
      version: "2.1.0"
    }
  ];

  return {
    featured_agents: featuredAgents,
    trending_agents: featuredAgents,
    new_agents: featuredAgents,
    top_rated_agents: featuredAgents,
    categories: [
      {
        id: "coding" as AgentCategory,
        name: "Coding",
        description: "Programming and development assistance",
        agent_count: 45
      },
      {
        id: "data-analysis" as AgentCategory,
        name: "Data Analysis",
        description: "Data processing and analytics",
        agent_count: 23
      },
      {
        id: "writing" as AgentCategory,
        name: "Writing",
        description: "Content creation and editing",
        agent_count: 18
      },
      {
        id: "automation" as AgentCategory,
        name: "Automation",
        description: "Task automation and workflows",
        agent_count: 12
      },
      {
        id: "testing" as AgentCategory,
        name: "Testing",
        description: "Quality assurance and testing",
        agent_count: 8
      }
    ]
  };
}

function getMockFeedback(): AgentFeedback[] {
  return [
    {
      id: "feedback-1",
      agent_id: "marketplace-1",
      user_id: "developer123",
      rating: 5,
      comment: "Excellent TypeScript assistant! Helped me refactor a large codebase with perfect type safety.",
      categories: ["accuracy", "usefulness"],
      helpful_count: 12,
      created_at: "2024-01-18T14:30:00Z"
    },
    {
      id: "feedback-2",
      agent_id: "marketplace-1",
      user_id: "codemaster",
      rating: 4,
      comment: "Great for TypeScript development. Sometimes responses are a bit verbose but very helpful overall.",
      categories: ["accuracy", "ease_of_use"],
      helpful_count: 8,
      created_at: "2024-01-16T09:15:00Z"
    }
  ];
}