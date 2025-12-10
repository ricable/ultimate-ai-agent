/**
 * Tests for task command
 */

import { jest } from '@jest/globals';
import { taskCommand } from '../task.js';
import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';

jest.mock('fs-extra');
jest.mock('ora');
jest.mock('chalk', () => ({
  default: {
    blue: jest.fn(str => str),
    green: jest.fn(str => str),
    yellow: jest.fn(str => str),
    red: jest.fn(str => str),
    cyan: jest.fn(str => str),
    magenta: jest.fn(str => str),
    dim: jest.fn(str => str),
    bold: jest.fn(str => str),
  }
}));

describe('Task Command', () => {
  let consoleLogSpy;
  let consoleErrorSpy;
  let mockSpinner;

  beforeEach(() => {
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation();
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    
    mockSpinner = {
      start: jest.fn().mockReturnThis(),
      succeed: jest.fn().mockReturnThis(),
      fail: jest.fn().mockReturnThis(),
      info: jest.fn().mockReturnThis(),
      warn: jest.fn().mockReturnThis(),
      text: '',
    };
    ora.mockReturnValue(mockSpinner);
    
    jest.clearAllMocks();
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  describe('list subcommand', () => {
    test('should list all tasks', async () => {
      const mockTasks = {
        tasks: [
          {
            id: 'task-1',
            title: 'Implement authentication',
            description: 'Add JWT authentication to API',
            status: 'in-progress',
            priority: 'high',
            assignedTo: 'agent-1',
            created: new Date().toISOString(),
            tags: ['api', 'security']
          },
          {
            id: 'task-2',
            title: 'Write tests',
            description: 'Add unit tests for core functions',
            status: 'pending',
            priority: 'medium',
            assignedTo: null,
            created: new Date().toISOString(),
            tags: ['testing']
          }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['list'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Tasks (2)');
      expect(output).toContain('Implement authentication');
      expect(output).toContain('in-progress');
      expect(output).toContain('Write tests');
      expect(output).toContain('pending');
    });

    test('should filter tasks by status', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', status: 'completed', title: 'Done task' },
          { id: 'task-2', status: 'in-progress', title: 'Working task' },
          { id: 'task-3', status: 'pending', title: 'Todo task' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['list'], { status: 'in-progress' });
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Working task');
      expect(output).not.toContain('Done task');
      expect(output).not.toContain('Todo task');
    });

    test('should filter tasks by assignee', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', assignedTo: 'agent-1', title: 'Agent 1 task' },
          { id: 'task-2', assignedTo: 'agent-2', title: 'Agent 2 task' },
          { id: 'task-3', assignedTo: null, title: 'Unassigned task' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['list'], { assignee: 'agent-1' });
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Agent 1 task');
      expect(output).not.toContain('Agent 2 task');
      expect(output).not.toContain('Unassigned task');
    });

    test('should show empty list message', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });

      await taskCommand(['list'], {});
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('No tasks found')
      );
    });
  });

  describe('create subcommand', () => {
    test('should create a new task', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['create', 'New task title', 'Task description'], {});
      
      expect(mockSpinner.succeed).toHaveBeenCalledWith(
        expect.stringContaining('Task created')
      );
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks).toHaveLength(1);
      expect(writeCall[1].tasks[0].title).toBe('New task title');
      expect(writeCall[1].tasks[0].description).toBe('Task description');
      expect(writeCall[1].tasks[0].status).toBe('pending');
    });

    test('should create task with priority', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['create', 'High priority task'], { priority: 'high' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].priority).toBe('high');
    });

    test('should create task with tags', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['create', 'Tagged task'], { tags: 'api,security,urgent' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].tags).toEqual(['api', 'security', 'urgent']);
    });

    test('should create task with assignment', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['create', 'Assigned task'], { assign: 'agent-1' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].assignedTo).toBe('agent-1');
    });

    test('should create tasks file if not exists', async () => {
      fs.pathExists.mockResolvedValue(false);
      fs.ensureDir.mockResolvedValue(undefined);
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['create', 'First task'], {});
      
      expect(fs.ensureDir).toHaveBeenCalled();
      expect(fs.writeJson).toHaveBeenCalled();
    });
  });

  describe('update subcommand', () => {
    test('should update task status', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Test task', status: 'pending' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['update', 'task-1'], { status: 'in-progress' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].status).toBe('in-progress');
      expect(mockSpinner.succeed).toHaveBeenCalledWith(
        expect.stringContaining('Task updated')
      );
    });

    test('should update task assignment', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Test task', assignedTo: null }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['update', 'task-1'], { assign: 'agent-2' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].assignedTo).toBe('agent-2');
    });

    test('should update task priority', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Test task', priority: 'medium' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['update', 'task-1'], { priority: 'high' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks[0].priority).toBe('high');
    });

    test('should handle non-existent task', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });

      await taskCommand(['update', 'nonexistent'], { status: 'completed' });
      
      expect(mockSpinner.fail).toHaveBeenCalledWith(
        expect.stringContaining('Task nonexistent not found')
      );
    });
  });

  describe('show subcommand', () => {
    test('should show task details', async () => {
      const mockTasks = {
        tasks: [
          {
            id: 'task-1',
            title: 'Detailed task',
            description: 'Long description here',
            status: 'in-progress',
            priority: 'high',
            assignedTo: 'agent-1',
            created: new Date().toISOString(),
            updated: new Date().toISOString(),
            tags: ['api', 'security'],
            dependencies: ['task-0'],
            metadata: {
              estimatedHours: 8,
              actualHours: 4
            }
          }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['show', 'task-1'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Task Details: task-1');
      expect(output).toContain('Detailed task');
      expect(output).toContain('Long description here');
      expect(output).toContain('in-progress');
      expect(output).toContain('agent-1');
      expect(output).toContain('[api, security]');
    });

    test('should handle non-existent task', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });

      await taskCommand(['show', 'nonexistent'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Task nonexistent not found')
      );
    });
  });

  describe('delete subcommand', () => {
    test('should delete a task', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Task 1' },
          { id: 'task-2', title: 'Task 2' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);
      fs.writeJson.mockResolvedValue(undefined);

      await taskCommand(['delete', 'task-1'], { force: true });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].tasks).toHaveLength(1);
      expect(writeCall[1].tasks[0].id).toBe('task-2');
      
      expect(mockSpinner.succeed).toHaveBeenCalledWith(
        expect.stringContaining('Task deleted')
      );
    });

    test('should require force flag', async () => {
      const mockTasks = {
        tasks: [{ id: 'task-1', title: 'Task 1' }]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['delete', 'task-1'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Use --force to confirm')
      );
    });
  });

  describe('stats subcommand', () => {
    test('should show task statistics', async () => {
      const mockTasks = {
        tasks: [
          { status: 'completed', priority: 'high', assignedTo: 'agent-1' },
          { status: 'in-progress', priority: 'medium', assignedTo: 'agent-1' },
          { status: 'pending', priority: 'high', assignedTo: 'agent-2' },
          { status: 'pending', priority: 'low', assignedTo: null },
          { status: 'completed', priority: 'medium', assignedTo: 'agent-2' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['stats'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Task Statistics');
      expect(output).toContain('Total: 5');
      expect(output).toContain('Completed: 2 (40%)');
      expect(output).toContain('In Progress: 1 (20%)');
      expect(output).toContain('Pending: 2 (40%)');
      expect(output).toContain('High Priority: 2');
      expect(output).toContain('Assigned: 4');
      expect(output).toContain('Unassigned: 1');
    });
  });

  describe('search subcommand', () => {
    test('should search tasks by title', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Implement API authentication', description: 'JWT auth' },
          { id: 'task-2', title: 'Write unit tests', description: 'Test coverage' },
          { id: 'task-3', title: 'API documentation', description: 'Swagger docs' }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['search', 'API'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Implement API authentication');
      expect(output).toContain('API documentation');
      expect(output).not.toContain('Write unit tests');
    });

    test('should search tasks by tags', async () => {
      const mockTasks = {
        tasks: [
          { id: 'task-1', title: 'Task 1', tags: ['api', 'security'] },
          { id: 'task-2', title: 'Task 2', tags: ['testing', 'quality'] },
          { id: 'task-3', title: 'Task 3', tags: ['api', 'performance'] }
        ]
      };

      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['search'], { tags: 'api' });
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Task 1');
      expect(output).toContain('Task 3');
      expect(output).not.toContain('Task 2');
    });
  });

  describe('help subcommand', () => {
    test('should show help when no arguments', async () => {
      await taskCommand([], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Task Management');
      expect(output).toContain('USAGE:');
      expect(output).toContain('task <subcommand>');
    });
  });

  describe('error handling', () => {
    test('should handle file system errors', async () => {
      fs.pathExists.mockRejectedValue(new Error('Permission denied'));

      await taskCommand(['list'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Error:')
      );
    });

    test('should handle invalid priority', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ tasks: [] });

      await taskCommand(['create', 'Test task'], { priority: 'invalid' });
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid priority')
      );
    });

    test('should handle invalid status', async () => {
      const mockTasks = { tasks: [{ id: 'task-1', status: 'pending' }] };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockTasks);

      await taskCommand(['update', 'task-1'], { status: 'invalid-status' });
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Invalid status')
      );
    });
  });
});