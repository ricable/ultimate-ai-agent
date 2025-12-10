"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Search,
  Filter,
  Star,
  Download,
  Globe,
  Terminal,
  Code,
  Database,
  Bot,
  Zap,
  Shield,
  Paintbrush,
  MessageSquare,
  FileText,
  BarChart3,
  Settings,
  ExternalLink,
  CheckCircle,
  Plus,
  ArrowRight,
  Tag,
  User,
  Calendar,
  TrendingUp,
  Heart,
  BookOpen
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { 
  Tooltip, 
  TooltipContent, 
  TooltipTrigger,
  TooltipProvider 
} from "@/components/ui/tooltip";

import { 
  MCPServerTemplate, 
  MCPServerConfig,
  MCP_SERVER_CATEGORIES 
} from "@/lib/types/mcp";
import { cn } from "@/lib/utils";

// Mock template data
const mockTemplates: MCPServerTemplate[] = [
  {
    id: "puppeteer-automation",
    name: "Puppeteer Automation",
    description: "Browser automation and web scraping with Puppeteer. Perfect for testing, screenshots, and data extraction.",
    category: "automation",
    tags: ["browser", "automation", "testing", "screenshots"],
    author: "OpenCode Team",
    version: "2.1.0",
    documentation: "https://github.com/opencode/puppeteer-mcp",
    icon: "🎭",
    popularity: 95,
    rating: 4.8,
    downloads: 15420,
    config: {
      name: "puppeteer-server",
      type: "local",
      command: ["npx", "puppeteer-mcp-server"],
      enabled: true,
      category: "automation",
      description: "Browser automation with Puppeteer"
    },
    requirements: {
      dependencies: ["puppeteer", "puppeteer-mcp-server"],
      environment: ["PUPPETEER_HEADLESS"],
      platforms: ["node"]
    },
    examples: [
      {
        name: "Basic Setup",
        description: "Standard Puppeteer server configuration",
        config: {
          name: "puppeteer-basic",
          type: "local",
          command: ["npx", "puppeteer-mcp-server"],
          environment: { "PUPPETEER_HEADLESS": "true" }
        }
      }
    ]
  },
  {
    id: "filesystem-manager",
    name: "File System Manager",
    description: "Comprehensive file system operations with advanced search, monitoring, and batch operations.",
    category: "file-management",
    tags: ["files", "filesystem", "search", "monitoring"],
    author: "Community",
    version: "1.8.2",
    documentation: "https://docs.opencode.dev/mcp/filesystem",
    icon: "📁",
    popularity: 87,
    rating: 4.6,
    downloads: 8930,
    config: {
      name: "filesystem-server",
      type: "local",
      command: ["python", "-m", "filesystem_mcp"],
      enabled: true,
      category: "file-management"
    },
    requirements: {
      dependencies: ["python>=3.8", "filesystem-mcp"],
      platforms: ["python"]
    }
  },
  {
    id: "slack-integration",
    name: "Slack Integration",
    description: "Send messages, manage channels, and interact with Slack workspaces directly from your AI sessions.",
    category: "communication",
    tags: ["slack", "messaging", "collaboration", "notifications"],
    author: "Slack",
    version: "3.0.1",
    documentation: "https://api.slack.com/mcp",
    icon: "💬",
    popularity: 76,
    rating: 4.4,
    downloads: 5240,
    config: {
      name: "slack-server",
      type: "remote",
      url: "https://mcp.slack.com/api",
      enabled: true,
      category: "communication"
    },
    requirements: {
      environment: ["SLACK_BOT_TOKEN", "SLACK_SIGNING_SECRET"]
    }
  },
  {
    id: "database-connector",
    name: "Universal Database Connector",
    description: "Connect to PostgreSQL, MySQL, SQLite, and MongoDB with unified query interface and schema introspection.",
    category: "integration",
    tags: ["database", "sql", "mongodb", "queries"],
    author: "DB Tools Inc",
    version: "4.2.0",
    documentation: "https://dbtools.dev/mcp-connector",
    icon: "🗄️",
    popularity: 82,
    rating: 4.7,
    downloads: 12100,
    config: {
      name: "database-server",
      type: "local",
      command: ["db-mcp-server"],
      enabled: true,
      category: "integration"
    },
    requirements: {
      dependencies: ["db-mcp-server", "database-drivers"],
      environment: ["DB_CONNECTION_STRING"]
    }
  },
  {
    id: "code-analysis",
    name: "Code Analysis Suite",
    description: "Static code analysis, complexity metrics, and code quality assessments for multiple languages.",
    category: "development",
    tags: ["code", "analysis", "quality", "metrics"],
    author: "DevTools Corp",
    version: "1.5.0",
    documentation: "https://devtools.corp/code-analysis-mcp",
    icon: "🔍",
    popularity: 69,
    rating: 4.3,
    downloads: 3850,
    config: {
      name: "code-analysis-server",
      type: "local",
      command: ["code-analysis-mcp"],
      enabled: true,
      category: "development"
    }
  },
  {
    id: "ai-assistant",
    name: "AI Assistant Bridge",
    description: "Connect to additional AI models and services for specialized tasks and multi-model workflows.",
    category: "custom",
    tags: ["ai", "models", "bridge", "workflow"],
    author: "AI Labs",
    version: "2.3.1",
    documentation: "https://ai-labs.io/mcp-bridge",
    icon: "🤖",
    popularity: 91,
    rating: 4.9,
    downloads: 18750,
    config: {
      name: "ai-assistant-bridge",
      type: "remote",
      url: "https://api.ai-labs.io/mcp",
      enabled: true,
      category: "custom"
    }
  }
];

interface ServerTemplatesProps {
  onInstallTemplate: (template: MCPServerTemplate) => void;
  installedServers?: string[];
}

export function ServerTemplates({ 
  onInstallTemplate,
  installedServers = []
}: ServerTemplatesProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"popularity" | "rating" | "downloads" | "name">("popularity");
  const [selectedTemplate, setSelectedTemplate] = useState<MCPServerTemplate | null>(null);

  // Filter and sort templates
  const filteredTemplates = useMemo(() => {
    let filtered = mockTemplates.filter(template => {
      const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      
      const matchesCategory = selectedCategory === "all" || template.category === selectedCategory;
      
      return matchesSearch && matchesCategory;
    });

    // Sort templates
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "popularity":
          return (b.popularity || 0) - (a.popularity || 0);
        case "rating":
          return (b.rating || 0) - (a.rating || 0);
        case "downloads":
          return (b.downloads || 0) - (a.downloads || 0);
        case "name":
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    return filtered;
  }, [searchQuery, selectedCategory, sortBy]);

  const getCategoryIcon = (category: string) => {
    const icons: Record<string, any> = {
      development: Code,
      productivity: Zap,
      communication: MessageSquare,
      "file-management": FileText,
      automation: Bot,
      monitoring: BarChart3,
      analytics: TrendingUp,
      security: Shield,
      integration: Settings,
      custom: Paintbrush
    };
    
    return icons[category] || Settings;
  };

  const isInstalled = (templateId: string) => {
    return installedServers.includes(templateId);
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h2 className="text-2xl font-bold">MCP Server Templates</h2>
          <p className="text-muted-foreground">
            Discover and install pre-configured MCP servers from the community
          </p>
        </div>

        {/* Filters and Search */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
          
          <div className="flex gap-2">
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {MCP_SERVER_CATEGORIES.map(category => (
                  <SelectItem key={category} value={category}>
                    {category.charAt(0).toUpperCase() + category.slice(1).replace("-", " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="popularity">Popular</SelectItem>
                <SelectItem value="rating">Rating</SelectItem>
                <SelectItem value="downloads">Downloads</SelectItem>
                <SelectItem value="name">Name</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{mockTemplates.length}</div>
              <div className="text-sm text-muted-foreground">Total Templates</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">{installedServers.length}</div>
              <div className="text-sm text-muted-foreground">Installed</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">
                {new Set(mockTemplates.map(t => t.category)).size}
              </div>
              <div className="text-sm text-muted-foreground">Categories</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-2xl font-bold">
                {Math.round(mockTemplates.reduce((acc, t) => acc + (t.rating || 0), 0) / mockTemplates.length * 10) / 10}
              </div>
              <div className="text-sm text-muted-foreground">Avg Rating</div>
            </CardContent>
          </Card>
        </div>

        {/* Templates Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTemplates.map((template) => {
            const CategoryIcon = getCategoryIcon(template.category);
            const installed = isInstalled(template.id);
            
            return (
              <motion.div
                key={template.id}
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white text-xl">
                          {template.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3 className="font-semibold truncate">{template.name}</h3>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              <CategoryIcon className="h-3 w-3 mr-1" />
                              {template.category.replace("-", " ")}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              v{template.version}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      
                      {installed && (
                        <CheckCircle className="h-5 w-5 text-green-500" />
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground line-clamp-3">
                      {template.description}
                    </p>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      {template.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {template.tags.length > 3 && (
                        <Badge variant="secondary" className="text-xs">
                          +{template.tags.length - 3}
                        </Badge>
                      )}
                    </div>

                    {/* Stats */}
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <div className="flex items-center space-x-3">
                        <Tooltip>
                          <TooltipTrigger>
                            <div className="flex items-center space-x-1">
                              <Star className="h-3 w-3 fill-current text-yellow-500" />
                              <span>{template.rating}</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Average rating</p>
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger>
                            <div className="flex items-center space-x-1">
                              <Download className="h-3 w-3" />
                              <span>{template.downloads?.toLocaleString()}</span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Total downloads</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>

                      <div className="flex items-center space-x-1">
                        <User className="h-3 w-3" />
                        <span className="text-xs">{template.author}</span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex space-x-2">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="outline" size="sm" className="flex-1">
                            <BookOpen className="h-4 w-4 mr-2" />
                            Details
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-3xl">
                          <DialogHeader>
                            <DialogTitle className="flex items-center space-x-3">
                              <span className="text-2xl">{template.icon}</span>
                              <span>{template.name}</span>
                            </DialogTitle>
                          </DialogHeader>
                          <TemplateDetailsView 
                            template={template} 
                            onInstall={onInstallTemplate}
                            installed={installed}
                          />
                        </DialogContent>
                      </Dialog>

                      <Button 
                        size="sm" 
                        className="flex-1"
                        onClick={() => onInstallTemplate(template)}
                        disabled={installed}
                      >
                        {installed ? (
                          <>
                            <CheckCircle className="h-4 w-4 mr-2" />
                            Installed
                          </>
                        ) : (
                          <>
                            <Plus className="h-4 w-4 mr-2" />
                            Install
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>

        {filteredTemplates.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <Search className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">No templates found</h3>
              <p className="text-muted-foreground">
                Try adjusting your search criteria or browse all categories
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </TooltipProvider>
  );
}

// Template Details View Component
function TemplateDetailsView({ 
  template, 
  onInstall,
  installed 
}: { 
  template: MCPServerTemplate;
  onInstall: (template: MCPServerTemplate) => void;
  installed: boolean;
}) {
  const [activeTab, setActiveTab] = useState<"overview" | "config" | "examples">("overview");
  const CategoryIcon = getCategoryIcon(template.category);

  return (
    <div className="space-y-6">
      {/* Template Header */}
      <div className="flex items-start space-x-4">
        <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white text-2xl">
          {template.icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Badge variant="outline">
              <CategoryIcon className="h-3 w-3 mr-1" />
              {template.category.replace("-", " ")}
            </Badge>
            <Badge variant="outline">v{template.version}</Badge>
            <Badge variant="outline">by {template.author}</Badge>
          </div>
          <p className="text-muted-foreground">{template.description}</p>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-500">{template.rating}</div>
          <div className="text-sm text-muted-foreground">Rating</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold">{template.downloads?.toLocaleString()}</div>
          <div className="text-sm text-muted-foreground">Downloads</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold">{template.popularity}%</div>
          <div className="text-sm text-muted-foreground">Popularity</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold">{template.tags.length}</div>
          <div className="text-sm text-muted-foreground">Tags</div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex space-x-1 bg-muted p-1 rounded-lg">
        {["overview", "config", "examples"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={cn(
              "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
              activeTab === tab
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === "overview" && (
          <div className="space-y-4">
            {/* Tags */}
            <div>
              <h4 className="font-medium mb-2">Tags</h4>
              <div className="flex flex-wrap gap-2">
                {template.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    <Tag className="h-3 w-3 mr-1" />
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Requirements */}
            {template.requirements && (
              <div>
                <h4 className="font-medium mb-2">Requirements</h4>
                <Card>
                  <CardContent className="p-4 space-y-3">
                    {template.requirements.dependencies && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">Dependencies</div>
                        <div className="mt-1 space-y-1">
                          {template.requirements.dependencies.map((dep, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs mr-1">
                              {dep}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {template.requirements.environment && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">Environment Variables</div>
                        <div className="mt-1 space-y-1">
                          {template.requirements.environment.map((env, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs mr-1">
                              {env}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {template.requirements.platforms && (
                      <div>
                        <div className="text-sm font-medium text-muted-foreground">Platforms</div>
                        <div className="mt-1 space-y-1">
                          {template.requirements.platforms.map((platform, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs mr-1">
                              {platform}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Documentation Link */}
            {template.documentation && (
              <div>
                <h4 className="font-medium mb-2">Documentation</h4>
                <Button variant="outline" asChild>
                  <a href={template.documentation} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View Documentation
                  </a>
                </Button>
              </div>
            )}
          </div>
        )}

        {activeTab === "config" && (
          <div className="space-y-4">
            <h4 className="font-medium">Default Configuration</h4>
            <Card>
              <CardContent className="p-4">
                <pre className="text-sm bg-muted p-3 rounded overflow-auto">
                  {JSON.stringify(template.config, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </div>
        )}

        {activeTab === "examples" && (
          <div className="space-y-4">
            {template.examples && template.examples.length > 0 ? (
              template.examples.map((example, idx) => (
                <Card key={idx}>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-base">{example.name}</CardTitle>
                    <p className="text-sm text-muted-foreground">{example.description}</p>
                  </CardHeader>
                  <CardContent>
                    <pre className="text-sm bg-muted p-3 rounded overflow-auto">
                      {JSON.stringify(example.config, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card>
                <CardContent className="p-8 text-center">
                  <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No examples available</h3>
                  <p className="text-muted-foreground">
                    This template doesn&apos;t include configuration examples
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </div>

      {/* Install Button */}
      <div className="flex justify-end space-x-2">
        <Button variant="outline">
          <Heart className="h-4 w-4 mr-2" />
          Add to Favorites
        </Button>
        <Button 
          onClick={() => onInstall(template)}
          disabled={installed}
        >
          {installed ? (
            <>
              <CheckCircle className="h-4 w-4 mr-2" />
              Already Installed
            </>
          ) : (
            <>
              <Download className="h-4 w-4 mr-2" />
              Install Template
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

function getCategoryIcon(category: string) {
  const icons: Record<string, any> = {
    development: Code,
    productivity: Zap,
    communication: MessageSquare,
    "file-management": FileText,
    automation: Bot,
    monitoring: BarChart3,
    analytics: TrendingUp,
    security: Shield,
    integration: Settings,
    custom: Paintbrush
  };
  
  return icons[category] || Settings;
}