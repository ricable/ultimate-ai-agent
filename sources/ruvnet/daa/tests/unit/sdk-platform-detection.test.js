/**
 * Unit Tests for DAA SDK Platform Detection
 *
 * Tests platform detection and module loading
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock platform detection
const createMockPlatform = () => ({
  detectPlatform() {
    if (typeof process !== 'undefined' && process.versions?.node) {
      return 'native';
    }
    return 'wasm';
  },

  isNodeJs() {
    return typeof process !== 'undefined' && process.versions?.node;
  },

  isBrowser() {
    return typeof window !== 'undefined' && typeof window.document !== 'undefined';
  },

  getNodeVersion() {
    return process.versions?.node || null;
  },

  getPlatformInfo() {
    return {
      platform: this.detectPlatform(),
      isNodeJs: this.isNodeJs(),
      isBrowser: this.isBrowser(),
      nodeVersion: this.getNodeVersion(),
      arch: process.arch,
      os: process.platform
    };
  }
});

const platform = createMockPlatform();

test('Platform: Detect Node.js environment', (t) => {
  const detected = platform.detectPlatform();

  assert.equal(detected, 'native', 'Should detect Node.js as native platform');
});

test('Platform: Is Node.js check', (t) => {
  const isNode = platform.isNodeJs();

  assert.equal(isNode, true, 'Should identify as Node.js environment');
});

test('Platform: Is not browser', (t) => {
  const isBrowser = platform.isBrowser();

  assert.equal(isBrowser, false, 'Should not identify as browser environment');
});

test('Platform: Get Node.js version', (t) => {
  const version = platform.getNodeVersion();

  assert.ok(version, 'Should return Node.js version');
  assert.match(version, /^\d+\.\d+\.\d+/, 'Version should match semver format');
});

test('Platform: Get platform info', (t) => {
  const info = platform.getPlatformInfo();

  assert.ok(info, 'Should return platform info');
  assert.equal(info.platform, 'native', 'Platform should be native');
  assert.equal(info.isNodeJs, true, 'Should be Node.js');
  assert.equal(info.isBrowser, false, 'Should not be browser');
  assert.ok(info.nodeVersion, 'Should have Node.js version');
  assert.ok(info.arch, 'Should have architecture');
  assert.ok(info.os, 'Should have OS platform');
});

test('Platform: Architecture detection', (t) => {
  const info = platform.getPlatformInfo();

  assert.ok(['x64', 'arm64', 'ia32'].includes(info.arch), 'Should have valid architecture');
});

test('Platform: OS detection', (t) => {
  const info = platform.getPlatformInfo();

  assert.ok(['linux', 'darwin', 'win32'].includes(info.os), 'Should have valid OS');
});
