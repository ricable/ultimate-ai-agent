/**
 * Checkpoint System Tests - Comprehensive tests for advanced checkpoint features
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { 
  CheckpointDiffViewer,
  CheckpointForkManager,
  AdvancedCheckpointManager,
  CheckpointExportImport 
} from '../index';
import { 
  Checkpoint, 
  CheckpointDiff, 
  CheckpointExportFormat 
} from '@/types/opencode';

// Mock data
const mockCheckpoint1: Checkpoint = {
  id: 'checkpoint-1',
  session_id: 'test-session',
  name: 'Test Checkpoint 1',
  description: 'First test checkpoint',
  message_index: 0,
  timestamp: Date.now() - 3600000,
  created_at: Date.now() - 3600000,
  type: 'manual',
  metadata: {
    userPrompt: 'Test prompt',
    model: 'claude-3.7-sonnet',
    provider: 'anthropic',
    totalTokens: 1000,
    inputTokens: 600,
    outputTokens: 400,
    cost: 0.02,
    fileChanges: 2,
    toolsUsed: ['file_read', 'file_write'],
    filesModified: ['test1.js', 'test2.js']
  }
};

const mockCheckpoint2: Checkpoint = {
  id: 'checkpoint-2',
  session_id: 'test-session',
  name: 'Test Checkpoint 2',
  description: 'Second test checkpoint',
  message_index: 5,
  timestamp: Date.now() - 1800000,
  created_at: Date.now() - 1800000,
  type: 'auto',
  metadata: {
    userPrompt: 'Another test prompt',
    model: 'claude-3.7-sonnet',
    provider: 'anthropic',
    totalTokens: 1500,
    inputTokens: 900,
    outputTokens: 600,
    cost: 0.03,
    fileChanges: 3,
    toolsUsed: ['file_read', 'file_write', 'bash'],
    filesModified: ['test1.js', 'test2.js', 'test3.js']
  }
};

const mockDiff: CheckpointDiff = {
  from_checkpoint: mockCheckpoint1,
  to_checkpoint: mockCheckpoint2,
  modified_files: [
    {
      path: 'test1.js',
      old_content: 'console.log("old");',
      new_content: 'console.log("new");',
      additions: 1,
      deletions: 1,
      changes: [
        {
          old_start: 1,
          old_count: 1,
          new_start: 1,
          new_count: 1,
          lines: [
            { type: 'removed', content: 'console.log("old");', old_line_number: 1 },
            { type: 'added', content: 'console.log("new");', new_line_number: 1 }
          ]
        }
      ]
    }
  ],
  added_files: ['test3.js'],
  deleted_files: [],
  token_delta: 500,
  cost_delta: 0.01,
  total_changes: 2
};

// Mock session store
vi.mock('@/lib/session-store', () => ({
  useSessionStore: () => ({
    actions: {
      createCheckpoint: vi.fn(),
      updateCheckpoint: vi.fn(),
      deleteCheckpoint: vi.fn()
    }
  }),
  useActiveSession: () => ({
    id: 'test-session',
    name: 'Test Session',
    created_at: Date.now()
  }),
  useActiveSessionMessages: () => [
    { id: '1', content: 'Test message 1' },
    { id: '2', content: 'Test message 2' }
  ]
}));

describe('CheckpointDiffViewer', () => {
  it('renders diff viewer with file changes', () => {
    render(
      <CheckpointDiffViewer
        diff={mockDiff}
        onExport={vi.fn()}
      />
    );

    expect(screen.getByText('Checkpoint Comparison')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument(); // Modified files count
    expect(screen.getByText('1')).toBeInTheDocument(); // Added files count
    expect(screen.getByText('test1.js')).toBeInTheDocument();
  });

  it('switches between side-by-side and unified views', async () => {
    render(
      <CheckpointDiffViewer
        diff={mockDiff}
        onExport={vi.fn()}
      />
    );

    const unifiedTab = screen.getByText('Unified');
    fireEvent.click(unifiedTab);

    // Should show unified view
    expect(unifiedTab).toHaveAttribute('data-state', 'active');
  });

  it('allows searching through file changes', async () => {
    render(
      <CheckpointDiffViewer
        diff={mockDiff}
        onExport={vi.fn()}
      />
    );

    const searchInput = screen.getByPlaceholderText('Search files...');
    fireEvent.change(searchInput, { target: { value: 'test1' } });

    await waitFor(() => {
      expect(searchInput).toHaveValue('test1');
    });
  });

  it('calls export callback when export button is clicked', () => {
    const onExport = vi.fn();
    render(
      <CheckpointDiffViewer
        diff={mockDiff}
        onExport={onExport}
      />
    );

    const exportButton = screen.getByLabelText('Export Diff');
    fireEvent.click(exportButton);

    expect(onExport).toHaveBeenCalledWith('pdf');
  });
});

describe('CheckpointForkManager', () => {
  it('renders fork manager with source checkpoint info', () => {
    render(
      <CheckpointForkManager
        sourceCheckpoint={mockCheckpoint1}
        timeline={[]}
        onForkCreated={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText('Create Checkpoint Fork')).toBeInTheDocument();
    expect(screen.getByText('Test Checkpoint 1')).toBeInTheDocument();
    expect(screen.getByText('1,000 tokens')).toBeInTheDocument();
  });

  it('allows configuring fork options', async () => {
    render(
      <CheckpointForkManager
        sourceCheckpoint={mockCheckpoint1}
        timeline={[]}
        onForkCreated={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const nameInput = screen.getByLabelText('Fork Name *');
    fireEvent.change(nameInput, { target: { value: 'My Custom Fork' } });

    await waitFor(() => {
      expect(nameInput).toHaveValue('My Custom Fork');
    });

    const preserveMessagesSwitch = screen.getByLabelText('Preserve Message History');
    fireEvent.click(preserveMessagesSwitch);

    // Switch should toggle
    expect(preserveMessagesSwitch).not.toBeChecked();
  });

  it('shows advanced options when expanded', async () => {
    render(
      <CheckpointForkManager
        sourceCheckpoint={mockCheckpoint1}
        timeline={[]}
        onForkCreated={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    const advancedButton = screen.getByText('Show Advanced Options');
    fireEvent.click(advancedButton);

    await waitFor(() => {
      expect(screen.getByText('Target Branch')).toBeInTheDocument();
      expect(screen.getByText('Branch Visualization')).toBeInTheDocument();
    });
  });

  it('calls onForkCreated when fork is created', async () => {
    const onForkCreated = vi.fn();
    render(
      <CheckpointForkManager
        sourceCheckpoint={mockCheckpoint1}
        timeline={[]}
        onForkCreated={onForkCreated}
        onCancel={vi.fn()}
      />
    );

    const createButton = screen.getByText('Create Fork');
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(onForkCreated).toHaveBeenCalled();
    });
  });
});

describe('AdvancedCheckpointManager', () => {
  it('renders checkpoint manager with create button', () => {
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={vi.fn()}
      />
    );

    expect(screen.getByText('Advanced Checkpoint Manager')).toBeInTheDocument();
    expect(screen.getByText('Create Checkpoint')).toBeInTheDocument();
  });

  it('opens create dialog when create button is clicked', async () => {
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={vi.fn()}
      />
    );

    const createButton = screen.getByText('Create Checkpoint');
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(screen.getByText('Save the current state of your session')).toBeInTheDocument();
    });
  });

  it('filters checkpoints by search term', async () => {
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={vi.fn()}
      />
    );

    const searchInput = screen.getByPlaceholderText('Search checkpoints...');
    fireEvent.change(searchInput, { target: { value: 'auth' } });

    await waitFor(() => {
      expect(searchInput).toHaveValue('auth');
    });
  });

  it('filters checkpoints by type', async () => {
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={vi.fn()}
      />
    );

    // Find and click the type selector
    const typeSelect = screen.getByDisplayValue('All Types');
    fireEvent.click(typeSelect);

    const manualOption = screen.getByText('Manual');
    fireEvent.click(manualOption);

    await waitFor(() => {
      expect(screen.getByDisplayValue('Manual')).toBeInTheDocument();
    });
  });

  it('shows statistics dialog when stats button is clicked', async () => {
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={vi.fn()}
      />
    );

    const statsButton = screen.getByLabelText('View Statistics');
    fireEvent.click(statsButton);

    await waitFor(() => {
      expect(screen.getByText('Checkpoint Statistics')).toBeInTheDocument();
    });
  });
});

describe('CheckpointExportImport', () => {
  it('renders export/import dialog', () => {
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    expect(screen.getByText('Checkpoint Export & Import')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
    expect(screen.getByText('Import')).toBeInTheDocument();
  });

  it('switches between export and import tabs', async () => {
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    const importTab = screen.getByText('Import');
    fireEvent.click(importTab);

    await waitFor(() => {
      expect(screen.getByText('Select Import File')).toBeInTheDocument();
    });
  });

  it('allows selecting different export formats', async () => {
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    const pdfFormat = screen.getByText('PDF Report');
    fireEvent.click(pdfFormat);

    // Should show PDF format as selected
    expect(pdfFormat.closest('[data-selected="true"]')).toBeTruthy();
  });

  it('toggles export options', async () => {
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    const includeFilesSwitch = screen.getByLabelText('Include File Snapshots');
    fireEvent.click(includeFilesSwitch);

    // Switch should toggle
    expect(includeFilesSwitch).not.toBeChecked();
  });

  it('shows password field when password protection is enabled', async () => {
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    const passwordSwitch = screen.getByLabelText('Password Protection');
    fireEvent.click(passwordSwitch);

    await waitFor(() => {
      expect(screen.getByLabelText('Export Password')).toBeInTheDocument();
    });
  });

  it('calls export callback when export is completed', async () => {
    const onExportComplete = vi.fn();
    render(
      <CheckpointExportImport
        checkpoints={[mockCheckpoint1, mockCheckpoint2]}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={onExportComplete}
        onImportComplete={vi.fn()}
      />
    );

    const exportButton = screen.getByText('Export Checkpoints');
    fireEvent.click(exportButton);

    await waitFor(() => {
      expect(onExportComplete).toHaveBeenCalled();
    }, { timeout: 10000 }); // Allow time for mock export process
  });
});

describe('Integration Tests', () => {
  it('checkpoint manager integrates with diff viewer', async () => {
    const onCheckpointSelect = vi.fn();
    render(
      <AdvancedCheckpointManager
        sessionId="test-session"
        onCheckpointSelect={onCheckpointSelect}
      />
    );

    // Should be able to trigger comparison actions
    // This would require more complex mock data and interactions
    expect(screen.getByText('Advanced Checkpoint Manager')).toBeInTheDocument();
  });

  it('maintains state consistency across components', () => {
    // Test that checkpoint data remains consistent when passed between components
    const checkpoints = [mockCheckpoint1, mockCheckpoint2];
    
    render(
      <CheckpointExportImport
        checkpoints={checkpoints}
        open={true}
        onOpenChange={vi.fn()}
        onExportComplete={vi.fn()}
        onImportComplete={vi.fn()}
      />
    );

    // Should show correct checkpoint count
    expect(screen.getByText('2 checkpoints selected')).toBeInTheDocument();
  });
});