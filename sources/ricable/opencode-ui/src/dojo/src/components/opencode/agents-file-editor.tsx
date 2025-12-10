import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Save, Loader2, FileText, Brain, Globe, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

interface AgentsFile {
  path: string;
  relative_path: string;
  absolute_path: string;
  scope: "global" | "project";
  exists: boolean;
}

interface AgentsFileEditorProps {
  /**
   * The AGENTS.md file to edit
   */
  file: AgentsFile;
  /**
   * Callback to go back to the previous view
   */
  onBack: () => void;
  /**
   * Optional className for styling
   */
  className?: string;
}

// Mock API functions - replace with actual OpenCode API calls
const mockApi = {
  async readAgentsFile(path: string): Promise<string> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Return default AGENTS.md content if file doesn't exist
    return `# Agent Configuration

## Overview
This file configures how OpenCode agents behave in this project.

## Rules

### Code Style
- Use TypeScript for all new code
- Follow existing naming conventions
- Add JSDoc comments for public APIs

### Development Workflow
- Always run tests before committing
- Use meaningful commit messages
- Create feature branches for new functionality

### OpenCode Specific
- Prefer local providers when available for privacy
- Use cost-effective models for simple tasks
- Cache expensive operations when possible

## Context

This is a TypeScript/React project using OpenCode for AI assistance.
Key technologies: React, TypeScript, Tailwind CSS, shadcn/ui.

## Preferences

- **Preferred Providers**: Ollama (local), Anthropic (cloud)
- **Model Selection**: Use local models for code completion, cloud models for complex reasoning
- **Cost Optimization**: Enable caching, use appropriate model sizes`;
  },

  async saveAgentsFile(path: string, content: string): Promise<void> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Simulate success
    console.log('Saving AGENTS.md to:', path, 'with content length:', content.length);
  }
};

/**
 * AGENTS.md file template sections
 */
const TEMPLATE_SECTIONS = {
  overview: `# Agent Configuration

## Overview
This file configures how OpenCode agents behave in this project.`,
  
  rules: `## Rules

### Code Style
- Use TypeScript for all new code
- Follow existing naming conventions
- Add JSDoc comments for public APIs

### Development Workflow
- Always run tests before committing
- Use meaningful commit messages
- Create feature branches for new functionality`,

  context: `## Context

This is a [PROJECT_TYPE] project using OpenCode for AI assistance.
Key technologies: [LIST_TECHNOLOGIES_HERE].`,

  preferences: `## Preferences

- **Preferred Providers**: [LIST_PROVIDERS]
- **Model Selection**: [DESCRIBE_MODEL_PREFERENCES]
- **Cost Optimization**: [LIST_COST_STRATEGIES]`
};

/**
 * AgentsFileEditor component for editing project-specific AGENTS.md files
 * Adapted from ClaudeFileEditor for OpenCode's multi-provider system
 * 
 * @example
 * <AgentsFileEditor 
 *   file={agentsFile} 
 *   onBack={() => setEditingFile(null)} 
 * />
 */
export const AgentsFileEditor: React.FC<AgentsFileEditorProps> = ({
  file,
  onBack,
  className,
}) => {
  const [content, setContent] = useState<string>("");
  const [originalContent, setOriginalContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"edit" | "preview">("edit");
  
  const { toast } = useToast();
  const hasChanges = content !== originalContent;
  
  // Load the file content on mount
  useEffect(() => {
    loadFileContent();
  }, [file.absolute_path]);
  
  const loadFileContent = async () => {
    try {
      setLoading(true);
      setError(null);
      const fileContent = await mockApi.readAgentsFile(file.absolute_path);
      setContent(fileContent);
      setOriginalContent(fileContent);
    } catch (err) {
      console.error("Failed to load file:", err);
      setError("Failed to load AGENTS.md file");
    } finally {
      setLoading(false);
    }
  };
  
  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      await mockApi.saveAgentsFile(file.absolute_path, content);
      setOriginalContent(content);
      toast({
        title: "File saved successfully",
        description: "AGENTS.md has been updated",
      });
    } catch (err) {
      console.error("Failed to save file:", err);
      setError("Failed to save AGENTS.md file");
      toast({
        title: "Failed to save file",
        description: "There was an error saving AGENTS.md",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };
  
  const handleBack = () => {
    if (hasChanges) {
      const confirmLeave = window.confirm(
        "You have unsaved changes. Are you sure you want to leave?"
      );
      if (!confirmLeave) return;
    }
    onBack();
  };

  const insertTemplate = (section: keyof typeof TEMPLATE_SECTIONS) => {
    const template = TEMPLATE_SECTIONS[section];
    const newContent = content + (content.endsWith('\n\n') ? '' : '\n\n') + template;
    setContent(newContent);
  };

  const renderPreview = () => {
    // Simple markdown rendering - in a real implementation, use a proper markdown renderer
    return (
      <div className="prose prose-sm max-w-none p-4">
        <pre className="whitespace-pre-wrap text-sm font-mono">{content}</pre>
      </div>
    );
  };

  return (
    <div className={cn("flex flex-col h-full bg-background", className)}>
      <div className="w-full max-w-5xl mx-auto flex flex-col h-full">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex items-center justify-between p-4 border-b border-border"
        >
          <div className="flex items-center space-x-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={handleBack}
              className="h-8 w-8"
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-blue-500" />
                <h2 className="text-lg font-semibold truncate">{file.relative_path}</h2>
                <Badge variant={file.scope === "global" ? "default" : "secondary"} className="text-xs">
                  {file.scope === "global" ? (
                    <><Globe className="h-3 w-3 mr-1" /> Global</>
                  ) : (
                    <><User className="h-3 w-3 mr-1" /> Project</>
                  )}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Configure OpenCode agent behavior and rules
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              onClick={handleSave}
              disabled={!hasChanges || saving}
              size="sm"
            >
              {saving ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              {saving ? "Saving..." : "Save"}
            </Button>
          </div>
        </motion.div>
        
        {/* Error display */}
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mx-4 mt-4 rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-xs text-destructive"
          >
            {error}
          </motion.div>
        )}
        
        {/* Template Helper Bar */}
        {!loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="p-3 border-b border-border bg-muted/30"
          >
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-muted-foreground mr-2">Quick templates:</span>
              {Object.keys(TEMPLATE_SECTIONS).map((section) => (
                <Button
                  key={section}
                  variant="outline"
                  size="sm"
                  onClick={() => insertTemplate(section as keyof typeof TEMPLATE_SECTIONS)}
                  className="text-xs h-6 px-2"
                >
                  {section.charAt(0).toUpperCase() + section.slice(1)}
                </Button>
              ))}
            </div>
          </motion.div>
        )}
        
        {/* Editor with Tabs */}
        <div className="flex-1 p-4 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "edit" | "preview")} className="h-full flex flex-col">
              <TabsList className="grid w-full grid-cols-2 mb-4">
                <TabsTrigger value="edit" className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Edit
                </TabsTrigger>
                <TabsTrigger value="preview" className="flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Preview
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="edit" className="flex-1 overflow-hidden">
                <Textarea
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Enter your AGENTS.md content here..."
                  className="h-full resize-none font-mono text-sm"
                />
              </TabsContent>
              
              <TabsContent value="preview" className="flex-1 overflow-auto border border-border rounded-lg">
                {renderPreview()}
              </TabsContent>
            </Tabs>
          )}
        </div>

        {/* Help Footer */}
        <div className="p-3 border-t border-border bg-muted/30">
          <div className="text-xs text-muted-foreground">
            <strong>AGENTS.md</strong> configures how OpenCode agents behave in your project. 
            Include rules, context, and preferences to guide AI assistance.
            {file.scope === "global" && " This is your global configuration applied to all projects."}
            {file.scope === "project" && " This is project-specific and overrides global settings."}
          </div>
        </div>
      </div>
    </div>
  );
};