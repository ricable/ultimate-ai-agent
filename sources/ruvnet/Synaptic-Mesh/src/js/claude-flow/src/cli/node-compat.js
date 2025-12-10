/**
 * Node.js compatibility layer for Deno APIs
 * This module provides Node.js equivalents for Deno-specific APIs
 */

import { readdir, stat, mkdir, readFile, writeFile, unlink, rmdir } from 'fs/promises';
import { existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import process from 'process';

// Process arguments (remove first two: node executable and script path)
export const args = process.argv.slice(2);

// Current working directory
export const cwd = () => process.cwd();

// File system operations
export const readDir = async (path) => {
  const entries = await readdir(path, { withFileTypes: true });
  return entries.map(entry => ({
    name: entry.name,
    isFile: entry.isFile(),
    isDirectory: entry.isDirectory(),
    isSymlink: entry.isSymbolicLink()
  }));
};

export const statFile = async (path) => {
  const stats = await stat(path);
  return {
    isFile: stats.isFile(),
    isDirectory: stats.isDirectory(),
    size: stats.size,
    mtime: stats.mtime,
    atime: stats.atime,
    birthtime: stats.birthtime
  };
};

export const readTextFile = async (path) => {
  return await readFile(path, 'utf-8');
};

export const writeTextFile = async (path, content) => {
  await writeFile(path, content, 'utf-8');
};

export const remove = async (path) => {
  const stats = await stat(path);
  if (stats.isDirectory()) {
    await rmdir(path, { recursive: true });
  } else {
    await unlink(path);
  }
};

export const mkdirSync = (path, options = {}) => {
  const fs = require('fs');
  fs.mkdirSync(path, { recursive: options.recursive });
};

export const mkdirAsync = async (path, options = {}) => {
  await mkdir(path, { recursive: options.recursive });
};

// Process operations
export const pid = process.pid;

export const kill = (pid, signal = 'SIGTERM') => {
  process.kill(pid, signal);
};

export const exit = (code = 0) => {
  process.exit(code);
};

// Deno.errors compatibility
export const errors = {
  NotFound: class NotFound extends Error {
    constructor(message) {
      super(message);
      this.name = 'NotFound';
    }
  },
  AlreadyExists: class AlreadyExists extends Error {
    constructor(message) {
      super(message);
      this.name = 'AlreadyExists';
    }
  },
  PermissionDenied: class PermissionDenied extends Error {
    constructor(message) {
      super(message);
      this.name = 'PermissionDenied';
    }
  }
};

// import.meta compatibility
export const getImportMetaUrl = () => {
  // This will be replaced by the actual import.meta.url in each file
  return import.meta.url;
};

export const getDirname = (importMetaUrl) => {
  const __filename = fileURLToPath(importMetaUrl);
  return dirname(__filename);
};

export const getFilename = (importMetaUrl) => {
  return fileURLToPath(importMetaUrl);
};

// Check if this is the main module (Node.js equivalent of import.meta.main)
export const isMainModule = (importMetaUrl) => {
  const __filename = fileURLToPath(importMetaUrl);
  return process.argv[1] === __filename;
};

// Helper to check file existence
export { existsSync };

// Build information (Node.js equivalent of Deno.build)
export const build = {
  os: process.platform === 'win32' ? 'windows' : 
      process.platform === 'darwin' ? 'darwin' :
      process.platform === 'linux' ? 'linux' : process.platform,
  arch: process.arch,
  target: `${process.arch}-${process.platform}`
};

// Export a Deno-like object for easier migration
export const Deno = {
  args,
  cwd,
  readDir,
  stat: statFile,
  readTextFile,
  writeTextFile,
  remove,
  mkdir: mkdirAsync,
  pid,
  kill,
  exit,
  errors,
  build
};

export default Deno;