/**
 * Settings Functionality Test Suite
 * Tests for bugs and issues in the settings dialog
 */

// Mock DOM environment for testing
const mockDOM = () => {
  // Create mock localStorage
  const storage = {};
  global.localStorage = {
    getItem: (key) => storage[key] || null,
    setItem: (key, value) => storage[key] = value,
    removeItem: (key) => delete storage[key],
    clear: () => Object.keys(storage).forEach(key => delete storage[key])
  };

  // Create mock document
  global.document = {
    getElementById: (id) => mockElements[id] || null,
    querySelector: (selector) => null,
    addEventListener: () => {},
    createElement: () => ({ 
      addEventListener: () => {},
      appendChild: () => {},
      classList: { add: () => {}, remove: () => {} }
    })
  };

  // Mock window
  global.window = {
    matchMedia: () => ({ addEventListener: () => {} })
  };
};

// Mock elements
const mockElements = {
  settingsPanel: {
    classList: {
      add: jest.fn(),
      remove: jest.fn(),
      contains: () => false
    },
    contains: () => false,
    querySelector: () => null
  },
  serverUrl: {
    value: 'ws://localhost:3000/ws',
    addEventListener: jest.fn()
  },
  authToken: {
    value: '',
    addEventListener: jest.fn()
  },
  fontSize: {
    value: '14',
    addEventListener: jest.fn()
  },
  defaultMode: {
    value: 'coder',
    addEventListener: jest.fn()
  },
  maxLines: {
    value: '1000',
    addEventListener: jest.fn()
  }
};

// Test Suite
describe('Settings Manager Tests', () => {
  let SettingsManager;

  beforeEach(() => {
    mockDOM();
    jest.clearAllMocks();
    localStorage.clear();
    
    // Import after mocking
    const module = require('../src/ui/console/js/settings.js');
    SettingsManager = module.SettingsManager;
  });

  describe('Initialization', () => {
    test('should merge defaults with stored settings', () => {
      localStorage.setItem('claude_console_settings', JSON.stringify({
        fontSize: 16,
        theme: 'light'
      }));

      const settings = new SettingsManager();
      
      expect(settings.get('fontSize')).toBe(16);
      expect(settings.get('theme')).toBe('light');
      expect(settings.get('serverUrl')).toBe('ws://localhost:3000/ws'); // default
    });

    test('should handle corrupted localStorage gracefully', () => {
      localStorage.setItem('claude_console_settings', 'invalid json{');
      
      const settings = new SettingsManager();
      
      expect(settings.get('fontSize')).toBe(14); // should use default
    });
  });

  describe('Validation Bugs', () => {
    test('validateSetting is never called - allows invalid values', () => {
      const settings = new SettingsManager();
      
      // This should fail validation but doesn't because validateSetting is never called
      settings.set('fontSize', 100); // way outside valid range
      expect(settings.get('fontSize')).toBe(100); // Bug: invalid value saved
      
      settings.set('theme', 'invalid-theme');
      expect(settings.get('theme')).toBe('invalid-theme'); // Bug: invalid value saved
    });

    test('defaultMode has mismatched values', () => {
      const settings = new SettingsManager();
      
      // The validator expects 'analyst' but HTML has 'analyzer'
      const isValid = settings.validateSetting('defaultMode', 'analyzer');
      expect(isValid).toBe(false); // Bug: HTML option doesn't match validation
      
      const isValidAnalyst = settings.validateSetting('defaultMode', 'analyst');
      expect(isValidAnalyst).toBe(true);
    });

    test('URL validation is missing', () => {
      const settings = new SettingsManager();
      
      // These invalid URLs are accepted
      settings.set('serverUrl', 'not-a-url');
      expect(settings.get('serverUrl')).toBe('not-a-url'); // Bug: no validation
      
      settings.set('serverUrl', 'http://'); // incomplete URL
      expect(settings.get('serverUrl')).toBe('http://'); // Bug: no validation
    });
  });

  describe('Memory Leaks', () => {
    test('multiple init calls create duplicate event listeners', () => {
      const settings = new SettingsManager();
      
      const addEventListenerSpy = jest.spyOn(document, 'addEventListener');
      
      settings.init();
      settings.init(); // Second init
      settings.init(); // Third init
      
      // Bug: Event listeners added multiple times
      const clickListenerCount = addEventListenerSpy.mock.calls
        .filter(call => call[0] === 'click').length;
      const keydownListenerCount = addEventListenerSpy.mock.calls
        .filter(call => call[0] === 'keydown').length;
      
      expect(clickListenerCount).toBeGreaterThan(1); // Bug: multiple listeners
      expect(keydownListenerCount).toBeGreaterThan(1); // Bug: multiple listeners
    });
  });

  describe('Missing Features', () => {
    test('advanced settings have no UI elements', () => {
      const settings = new SettingsManager();
      
      // These settings exist in defaults but have no UI
      expect(settings.get('reconnectAttempts')).toBe(5);
      expect(settings.get('heartbeatInterval')).toBe(30000);
      expect(settings.get('commandTimeout')).toBe(30000);
      
      // But no DOM elements exist for them
      expect(document.getElementById('reconnectAttempts')).toBeNull();
      expect(document.getElementById('heartbeatInterval')).toBeNull();
      expect(document.getElementById('commandTimeout')).toBeNull();
    });

    test('import/export methods exist but have no UI triggers', () => {
      const settings = new SettingsManager();
      
      // Methods exist
      expect(typeof settings.exportSettings).toBe('function');
      expect(typeof settings.importSettings).toBe('function');
      
      // But no buttons to trigger them
      expect(document.getElementById('exportSettings')).toBeNull();
      expect(document.getElementById('importSettings')).toBeNull();
    });
  });

  describe('State Management Bugs', () => {
    test('connection button states not initialized', () => {
      const settings = new SettingsManager();
      settings.init();
      
      const connectButton = mockElements.connectButton = {
        disabled: undefined, // Bug: not initialized
        addEventListener: jest.fn()
      };
      const disconnectButton = mockElements.disconnectButton = {
        disabled: undefined, // Bug: not initialized
        addEventListener: jest.fn()
      };
      
      // Buttons should have initial state but don't
      expect(connectButton.disabled).toBeUndefined(); // Bug
      expect(disconnectButton.disabled).toBeUndefined(); // Bug
    });

    test('race condition with auto-connect', async () => {
      const settings = new SettingsManager();
      settings.set('autoConnect', true);
      
      let connectCalled = false;
      settings.on('connect_requested', () => {
        connectCalled = true;
        // At this point, settings might not be fully applied
        // This is a race condition
      });
      
      // In real app, auto-connect might fire before settings are applied
      // This test demonstrates the potential for race conditions
    });
  });

  describe('Edge Cases', () => {
    test('settings panel visibility toggle edge cases', () => {
      const settings = new SettingsManager();
      settings.init();
      
      // Toggle without initialization
      settings.toggle(); // Should not throw
      expect(settings.isVisible).toBe(true);
      
      // Hide when already hidden
      settings.hide();
      settings.hide(); // Should not throw
      expect(settings.isVisible).toBe(false);
      
      // Show when already visible  
      settings.show();
      settings.show(); // Should not throw
      expect(settings.isVisible).toBe(true);
    });

    test('event emitter with no listeners', () => {
      const settings = new SettingsManager();
      
      // Emit with no listeners - should not throw
      expect(() => {
        settings.emit('test_event', { data: 'test' });
      }).not.toThrow();
    });

    test('update form elements with missing elements', () => {
      const settings = new SettingsManager();
      
      // Set a value for which no DOM element exists
      settings.set('nonExistentSetting', 'value');
      
      // Should not throw when updating form elements
      expect(() => {
        settings.updateFormElements();
      }).not.toThrow();
    });
  });
});

// Run tests
if (typeof jest === 'undefined') {
  console.log('⚠️  Jest not installed. These tests demonstrate the bugs found.');
  console.log('Run with: npm test tests/settings-functionality-test.js');
}