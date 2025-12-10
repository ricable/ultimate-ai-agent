/**
 * Checkpoint Demo Page - Showcase advanced checkpoint features
 */

"use client";

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  GitBranch, 
  GitCompare, 
  GitFork, 
  Download, 
  Upload,
  Clock,
  Bookmark,
  CheckCircle
} from 'lucide-react';
import { 
  CheckpointDiffViewer,
  CheckpointForkManager,
  AdvancedCheckpointManager,
  CheckpointExportImport
} from '@/components/opencode';
import { 
  Checkpoint, 
  CheckpointDiff, 
  CheckpointExportFormat 
} from '@/types/opencode';

const mockCheckpoint1: Checkpoint = {
  id: 'checkpoint-1',
  session_id: 'demo-session',
  name: 'Authentication System Setup',
  description: 'Initial implementation of user authentication with JWT tokens',
  message_index: 5,
  timestamp: Date.now() - 3600000,
  created_at: Date.now() - 3600000,
  type: 'manual',
  metadata: {
    userPrompt: 'Help me implement user authentication',
    model: 'claude-3.7-sonnet',
    provider: 'anthropic',
    totalTokens: 2500,
    inputTokens: 1200,
    outputTokens: 1300,
    cost: 0.045,
    fileChanges: 6,
    toolsUsed: ['file_write', 'bash', 'grep'],
    filesModified: ['auth.js', 'middleware.js', 'routes/auth.js', 'models/user.js', 'config/jwt.js', 'tests/auth.test.js']
  }
};

const mockCheckpoint2: Checkpoint = {
  id: 'checkpoint-2',
  session_id: 'demo-session',
  name: 'Database Integration Complete',
  description: 'Added PostgreSQL integration with user tables and migrations',
  message_index: 12,
  timestamp: Date.now() - 1800000,
  created_at: Date.now() - 1800000,
  type: 'milestone',
  metadata: {
    userPrompt: 'Integrate PostgreSQL database for user management',
    model: 'claude-3.7-sonnet',
    provider: 'anthropic',
    totalTokens: 3200,
    inputTokens: 1800,
    outputTokens: 1400,
    cost: 0.067,
    fileChanges: 8,
    toolsUsed: ['file_write', 'bash', 'grep', 'sql_executor'],
    filesModified: ['config/database.js', 'migrations/001_create_users.sql', 'models/user.js', 'seeds/users.js']
  }
};

const mockDiff: CheckpointDiff = {
  from_checkpoint: mockCheckpoint1,
  to_checkpoint: mockCheckpoint2,
  modified_files: [
    {
      path: 'models/user.js',
      old_content: '// Simple user model\nconst users = [];\n\nmodule.exports = { users };',
      new_content: '// PostgreSQL user model\nconst { Pool } = require("pg");\nconst pool = new Pool();\n\nclass User {\n  static async findById(id) {\n    const result = await pool.query("SELECT * FROM users WHERE id = $1", [id]);\n    return result.rows[0];\n  }\n}\n\nmodule.exports = User;',
      additions: 12,
      deletions: 3,
      changes: [
        {
          old_start: 1,
          old_count: 3,
          new_start: 1,
          new_count: 12,
          lines: [
            { type: 'removed', content: '// Simple user model', old_line_number: 1 },
            { type: 'removed', content: 'const users = [];', old_line_number: 2 },
            { type: 'removed', content: 'module.exports = { users };', old_line_number: 3 },
            { type: 'added', content: '// PostgreSQL user model', new_line_number: 1 },
            { type: 'added', content: 'const { Pool } = require("pg");', new_line_number: 2 },
            { type: 'added', content: 'const pool = new Pool();', new_line_number: 3 },
            { type: 'added', content: '', new_line_number: 4 },
            { type: 'added', content: 'class User {', new_line_number: 5 },
            { type: 'added', content: '  static async findById(id) {', new_line_number: 6 },
            { type: 'added', content: '    const result = await pool.query("SELECT * FROM users WHERE id = $1", [id]);', new_line_number: 7 },
            { type: 'added', content: '    return result.rows[0];', new_line_number: 8 },
            { type: 'added', content: '  }', new_line_number: 9 },
            { type: 'added', content: '}', new_line_number: 10 },
            { type: 'added', content: '', new_line_number: 11 },
            { type: 'added', content: 'module.exports = User;', new_line_number: 12 }
          ]
        }
      ]
    }
  ],
  added_files: ['config/database.js', 'migrations/001_create_users.sql'],
  deleted_files: ['data/users.json'],
  token_delta: 700,
  cost_delta: 0.022,
  total_changes: 15
};

export default function CheckpointDemoPage() {
  const [activeDemo, setActiveDemo] = useState<'manager' | 'diff' | 'fork' | 'export'>('manager');
  const [showDiffDialog, setShowDiffDialog] = useState(false);
  const [showForkDialog, setShowForkDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold">Advanced Checkpoint System Demo</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Experience the comprehensive checkpoint management system with diff comparison, 
          forking capabilities, advanced metadata tracking, and export functionality.
        </p>
      </div>

      {/* Feature Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <GitBranch className="h-8 w-8 mx-auto mb-2 text-primary" />
            <h3 className="font-medium">Advanced Timeline</h3>
            <p className="text-sm text-muted-foreground">Visual timeline with branching support</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <GitCompare className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <h3 className="font-medium">Diff Comparison</h3>
            <p className="text-sm text-muted-foreground">Side-by-side code differences</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <GitFork className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <h3 className="font-medium">Checkpoint Forking</h3>
            <p className="text-sm text-muted-foreground">Create branches from any checkpoint</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <Download className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <h3 className="font-medium">Export/Import</h3>
            <p className="text-sm text-muted-foreground">Multiple export formats supported</p>
          </CardContent>
        </Card>
      </div>

      {/* Demo Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Interactive Demos</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
            <Button
              variant={activeDemo === 'manager' ? 'default' : 'outline'}
              onClick={() => setActiveDemo('manager')}
              className="h-auto flex-col p-4"
            >
              <GitBranch className="h-6 w-6 mb-2" />
              <span>Checkpoint Manager</span>
            </Button>
            <Button
              variant={activeDemo === 'diff' ? 'default' : 'outline'}
              onClick={() => setShowDiffDialog(true)}
              className="h-auto flex-col p-4"
            >
              <GitCompare className="h-6 w-6 mb-2" />
              <span>Diff Viewer</span>
            </Button>
            <Button
              variant={activeDemo === 'fork' ? 'default' : 'outline'}
              onClick={() => setShowForkDialog(true)}
              className="h-auto flex-col p-4"
            >
              <GitFork className="h-6 w-6 mb-2" />
              <span>Fork Manager</span>
            </Button>
            <Button
              variant={activeDemo === 'export' ? 'default' : 'outline'}
              onClick={() => setShowExportDialog(true)}
              className="h-auto flex-col p-4"
            >
              <Download className="h-6 w-6 mb-2" />
              <span>Export/Import</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Demo Content */}
      <Card className="min-h-[600px]">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <GitBranch className="h-5 w-5" />
            <span>Advanced Checkpoint Manager</span>
            <Badge variant="secondary">Live Demo</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-[600px]">
            <AdvancedCheckpointManager
              sessionId="demo-session"
              compact={false}
              onCheckpointSelect={(checkpoint) => {
                console.log('Selected checkpoint:', checkpoint);
              }}
            />
          </div>
        </CardContent>
      </Card>

      {/* Feature Highlights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Key Features</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start space-x-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <div className="font-medium">Enhanced Metadata Tracking</div>
                <div className="text-sm text-muted-foreground">
                  Track provider, model, costs, tokens, and tool usage
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <div className="font-medium">Advanced Search & Filtering</div>
                <div className="text-sm text-muted-foreground">
                  Find checkpoints by type, provider, cost, or content
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <div className="font-medium">Visual Timeline with Branching</div>
                <div className="text-sm text-muted-foreground">
                  Git-like visualization of checkpoint relationships
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <div className="font-medium">Comprehensive Statistics</div>
                <div className="text-sm text-muted-foreground">
                  Usage analytics, cost breakdowns, and performance metrics
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Technical Capabilities</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start space-x-3">
              <Clock className="h-5 w-5 text-blue-500 mt-0.5" />
              <div>
                <div className="font-medium">Real-time Collaboration</div>
                <div className="text-sm text-muted-foreground">
                  Share checkpoints and collaborate on session branches
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <Bookmark className="h-5 w-5 text-purple-500 mt-0.5" />
              <div>
                <div className="font-medium">Multiple Export Formats</div>
                <div className="text-sm text-muted-foreground">
                  JSON, Markdown, PDF, and ZIP archive formats
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <GitFork className="h-5 w-5 text-orange-500 mt-0.5" />
              <div>
                <div className="font-medium">Advanced Forking</div>
                <div className="text-sm text-muted-foreground">
                  Create branches with configurable data preservation
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <GitCompare className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <div className="font-medium">Syntax-Highlighted Diffs</div>
                <div className="text-sm text-muted-foreground">
                  Side-by-side and unified diff views with search
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dialogs */}
      {showDiffDialog && (
        <div className="fixed inset-0 z-50 bg-background">
          <div className="h-full flex flex-col">
            <div className="p-4 border-b flex items-center justify-between">
              <h2 className="text-lg font-semibold">Checkpoint Diff Viewer Demo</h2>
              <Button variant="outline" onClick={() => setShowDiffDialog(false)}>
                Close Demo
              </Button>
            </div>
            <div className="flex-1">
              <CheckpointDiffViewer
                diff={mockDiff}
                fullscreen={true}
                onFullscreenChange={() => {}}
                onExport={(format) => console.log('Export diff:', format)}
              />
            </div>
          </div>
        </div>
      )}

      {showForkDialog && (
        <CheckpointForkManager
          sourceCheckpoint={mockCheckpoint1}
          timeline={[]}
          isDialog={true}
          onForkCreated={(forkId, options) => {
            console.log('Fork created:', forkId, options);
            setShowForkDialog(false);
          }}
          onCancel={() => setShowForkDialog(false)}
        />
      )}

      {showExportDialog && (
        <CheckpointExportImport
          checkpoints={[mockCheckpoint1, mockCheckpoint2]}
          open={showExportDialog}
          onOpenChange={setShowExportDialog}
          defaultTab="export"
          onExportComplete={(result) => {
            console.log('Export complete:', result);
          }}
          onImportComplete={(checkpoints) => {
            console.log('Import complete:', checkpoints);
          }}
        />
      )}
    </div>
  );
}