/**
 * Tests for memory command
 */

import { jest } from '@jest/globals';
import { memoryCommand } from '../memory.js';
import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';

jest.mock('fs-extra');
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

describe('Memory Command', () => {
  let consoleLogSpy;
  let consoleErrorSpy;
  const memoryPath = path.join(process.cwd(), '.claude', 'memory', 'memory.json');

  beforeEach(() => {
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation();
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    jest.clearAllMocks();
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
  });

  describe('store subcommand', () => {
    test('should store memory entry', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['store', 'test-key', 'test-value'], {});
      
      expect(fs.writeJson).toHaveBeenCalled();
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries).toHaveLength(1);
      expect(writeCall[1].entries[0].key).toBe('test-key');
      expect(writeCall[1].entries[0].value).toBe('test-value');
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Memory stored successfully')
      );
    });

    test('should store memory with tags', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['store', 'key', 'value'], { tags: 'important,api' });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries[0].tags).toEqual(['important', 'api']);
    });

    test('should store memory with TTL', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['store', 'key', 'value'], { ttl: 3600 });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries[0].expiresAt).toBeDefined();
    });

    test('should handle JSON values', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });
      fs.writeJson.mockResolvedValue(undefined);

      const jsonValue = '{"name":"test","count":42}';
      await memoryCommand(['store', 'json-key', jsonValue], {});
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries[0].value).toEqual({ name: 'test', count: 42 });
    });

    test('should create memory file if not exists', async () => {
      fs.pathExists.mockResolvedValue(false);
      fs.ensureDir.mockResolvedValue(undefined);
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['store', 'key', 'value'], {});
      
      expect(fs.ensureDir).toHaveBeenCalledWith(path.dirname(memoryPath));
      expect(fs.writeJson).toHaveBeenCalled();
    });
  });

  describe('retrieve subcommand', () => {
    test('should retrieve memory by key', async () => {
      const mockMemory = {
        entries: [
          { key: 'test-key', value: 'test-value', timestamp: new Date().toISOString() }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['retrieve', 'test-key'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('test-value');
    });

    test('should show not found message', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });

      await memoryCommand(['retrieve', 'nonexistent'], {});
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('No memory found for key: nonexistent')
      );
    });

    test('should handle expired entries', async () => {
      const pastDate = new Date(Date.now() - 1000).toISOString();
      const mockMemory = {
        entries: [
          { key: 'expired', value: 'value', expiresAt: pastDate }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['retrieve', 'expired'], {});
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('No memory found for key: expired')
      );
    });
  });

  describe('list subcommand', () => {
    test('should list all memories', async () => {
      const mockMemory = {
        entries: [
          { key: 'key1', value: 'value1', timestamp: new Date().toISOString() },
          { key: 'key2', value: { complex: 'object' }, timestamp: new Date().toISOString() },
          { key: 'key3', value: 'value3', tags: ['important'], timestamp: new Date().toISOString() }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['list'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Memory Entries (3)');
      expect(output).toContain('key1');
      expect(output).toContain('key2');
      expect(output).toContain('key3');
      expect(output).toContain('[important]');
    });

    test('should filter by pattern', async () => {
      const mockMemory = {
        entries: [
          { key: 'api/user', value: 'data1', timestamp: new Date().toISOString() },
          { key: 'api/product', value: 'data2', timestamp: new Date().toISOString() },
          { key: 'config/settings', value: 'data3', timestamp: new Date().toISOString() }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['list'], { pattern: 'api/*' });
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('api/user');
      expect(output).toContain('api/product');
      expect(output).not.toContain('config/settings');
    });

    test('should filter by tags', async () => {
      const mockMemory = {
        entries: [
          { key: 'key1', value: 'v1', tags: ['important'], timestamp: new Date().toISOString() },
          { key: 'key2', value: 'v2', tags: ['temporary'], timestamp: new Date().toISOString() },
          { key: 'key3', value: 'v3', tags: ['important', 'api'], timestamp: new Date().toISOString() }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['list'], { tags: 'important' });
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('key1');
      expect(output).toContain('key3');
      expect(output).not.toContain('key2');
    });

    test('should show empty message', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });

      await memoryCommand(['list'], {});
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('No memories stored')
      );
    });
  });

  describe('delete subcommand', () => {
    test('should delete memory by key', async () => {
      const mockMemory = {
        entries: [
          { key: 'key1', value: 'value1' },
          { key: 'key2', value: 'value2' }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['delete', 'key1'], {});
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries).toHaveLength(1);
      expect(writeCall[1].entries[0].key).toBe('key2');
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Memory deleted: key1')
      );
    });

    test('should handle non-existent key', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue({ entries: [] });

      await memoryCommand(['delete', 'nonexistent'], {});
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Memory not found: nonexistent')
      );
    });
  });

  describe('clear subcommand', () => {
    test('should clear all memories with force flag', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['clear'], { force: true });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries).toEqual([]);
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('All memories cleared')
      );
    });

    test('should warn without force flag', async () => {
      await memoryCommand(['clear'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Use --force to confirm')
      );
    });
  });

  describe('export subcommand', () => {
    test('should export memories to file', async () => {
      const mockMemory = {
        entries: [
          { key: 'key1', value: 'value1' },
          { key: 'key2', value: 'value2' }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['export', 'backup.json'], {});
      
      expect(fs.writeJson).toHaveBeenCalledWith(
        'backup.json',
        mockMemory,
        { spaces: 2 }
      );
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Memories exported to backup.json')
      );
    });
  });

  describe('import subcommand', () => {
    test('should import memories from file', async () => {
      const importData = {
        entries: [
          { key: 'imported1', value: 'value1' },
          { key: 'imported2', value: 'value2' }
        ]
      };
      fs.pathExists.mockResolvedValueOnce(true) // import file exists
        .mockResolvedValueOnce(true); // memory file exists
      fs.readJson.mockResolvedValueOnce(importData) // import data
        .mockResolvedValueOnce({ entries: [] }); // existing memory
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['import', 'import.json'], {});
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries).toHaveLength(2);
      
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Imported 2 memory entries')
      );
    });

    test('should merge with existing memories', async () => {
      const importData = { entries: [{ key: 'new', value: 'imported' }] };
      const existingData = { entries: [{ key: 'existing', value: 'original' }] };
      
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValueOnce(importData)
        .mockResolvedValueOnce(existingData);
      fs.writeJson.mockResolvedValue(undefined);

      await memoryCommand(['import', 'import.json'], { merge: true });
      
      const writeCall = fs.writeJson.mock.calls[0];
      expect(writeCall[1].entries).toHaveLength(2);
    });
  });

  describe('stats subcommand', () => {
    test('should show memory statistics', async () => {
      const mockMemory = {
        entries: [
          { key: 'key1', value: 'short', tags: ['tag1'] },
          { key: 'key2', value: { complex: 'object' }, tags: ['tag1', 'tag2'] },
          { key: 'key3', value: 'medium length value', tags: ['tag2'] },
          { key: 'expired', value: 'value', expiresAt: new Date(Date.now() - 1000).toISOString() }
        ]
      };
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockResolvedValue(mockMemory);

      await memoryCommand(['stats'], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Memory Statistics');
      expect(output).toContain('Total entries: 4');
      expect(output).toContain('Active entries: 3');
      expect(output).toContain('Expired entries: 1');
      expect(output).toContain('Unique tags: 2');
    });
  });

  describe('help subcommand', () => {
    test('should show help when no arguments', async () => {
      await memoryCommand([], {});
      
      const output = consoleLogSpy.mock.calls.flat().join('\n');
      expect(output).toContain('Memory Management');
      expect(output).toContain('USAGE:');
      expect(output).toContain('memory <subcommand>');
    });
  });

  describe('error handling', () => {
    test('should handle file system errors', async () => {
      fs.pathExists.mockRejectedValue(new Error('Permission denied'));

      await memoryCommand(['list'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Error:')
      );
    });

    test('should handle invalid JSON in import', async () => {
      fs.pathExists.mockResolvedValue(true);
      fs.readJson.mockRejectedValue(new Error('Invalid JSON'));

      await memoryCommand(['import', 'bad.json'], {});
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Error:')
      );
    });
  });
});