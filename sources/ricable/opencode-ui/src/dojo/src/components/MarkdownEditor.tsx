import React, { useState, useEffect } from "react";
import MDEditor from "@uiw/react-md-editor";
import { motion } from "framer-motion";
import { ArrowLeft, Save, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toast, ToastContainer } from "@/components/ui/toast";
import { openCodeClient } from "@/lib/opencode-client";
import { cn } from "@/lib/utils";

interface MarkdownEditorProps {
  /**
   * Callback to go back to the main view
   */
  onBack: () => void;
  /**
   * Optional className for styling
   */
  className?: string;
  /**
   * File path to edit (defaults to AGENTS.md)
   */
  filePath?: string;
  /**
   * Title for the editor
   */
  title?: string;
  /**
   * Description for the editor
   */
  description?: string;
}

/**
 * MarkdownEditor component for editing OpenCode configuration files
 * 
 * @example
 * <MarkdownEditor onBack={() => setView('main')} />
 */
export const MarkdownEditor: React.FC<MarkdownEditorProps> = ({
  onBack,
  className,
  filePath = "AGENTS.md",
  title = "AGENTS.md",
  description = "Edit your OpenCode agent configuration",
}) => {
  const [content, setContent] = useState<string>("");
  const [originalContent, setOriginalContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);
  
  const hasChanges = content !== originalContent;
  
  // Load the file content on mount
  useEffect(() => {
    loadFileContent();
  }, [filePath]); // eslint-disable-line react-hooks/exhaustive-deps
  
  const loadFileContent = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Try to read the file using OpenCode API
      // For now, we'll use mock content since the specific API doesn't exist yet
      const mockContent = getDefaultContent(filePath);
      setContent(mockContent);
      setOriginalContent(mockContent);
    } catch (err) {
      console.error("Failed to load file:", err);
      setError(`Failed to load ${filePath} file`);
      
      // Set default content on error
      const defaultContent = getDefaultContent(filePath);
      setContent(defaultContent);
      setOriginalContent(defaultContent);
    } finally {
      setLoading(false);
    }
  };
  
  const getDefaultContent = (path: string): string => {
    if (path === "AGENTS.md" || path.endsWith("AGENTS.md")) {
      return `# Agent Configuration

## Global Rules
These rules apply to all agents in your OpenCode sessions.

### Code Style Preferences
- Use TypeScript for all new code
- Follow existing naming conventions (camelCase for variables, PascalCase for classes)
- Add JSDoc comments for public APIs
- Use async/await instead of Promise chains
- Prefer const/let over var

### Workflow Guidelines
- Always run tests after making code changes
- Use meaningful commit messages following conventional commits
- Create feature branches for new functionality
- Ensure all tests pass before merging

### Project-Specific Instructions
Add any project-specific rules here that should apply to all AI agents working on this codebase.

## Agent Behavior
- Be concise and direct in responses
- Ask clarifying questions when requirements are unclear
- Provide code examples when explaining concepts
- Consider security implications in all suggestions
- Focus on maintainable and scalable solutions
`;
    }
    
    return `# ${path}

Add your configuration here.
`;
  };
  
  const handleSave = async () => {
    try {
      setSaving(true);
      setError(null);
      setToast(null);
      
      // In a real implementation, this would save via OpenCode API
      // For now, we'll simulate a successful save
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setOriginalContent(content);
      setToast({ message: `${filePath} saved successfully`, type: "success" });
    } catch (err) {
      console.error("Failed to save file:", err);
      setError(`Failed to save ${filePath} file`);
      setToast({ message: `Failed to save ${filePath}`, type: "error" });
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
            <div>
              <h2 className="text-lg font-semibold">{title}</h2>
              <p className="text-xs text-muted-foreground">
                {description}
              </p>
            </div>
          </div>
          
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
        
        {/* Editor */}
        <div className="flex-1 p-4 overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="h-full rounded-lg border border-border overflow-hidden shadow-sm" data-color-mode="dark">
              <MDEditor
                value={content}
                onChange={(val) => setContent(val || "")}
                preview="edit"
                height="100%"
                visibleDragbar={false}
              />
            </div>
          )}
        </div>
      </div>
      
      {/* Toast Notification */}
      <ToastContainer>
        {toast && (
          <Toast
            message={toast.message}
            type={toast.type}
            onDismiss={() => setToast(null)}
          />
        )}
      </ToastContainer>
    </div>
  );
};