/**
 * Nova Medicina - Help System Tests
 *
 * Comprehensive tests for the CLI help system
 */

import helpSystem from '../../src/cli/help-system';

describe('Help System', () => {
  // Capture console output for testing
  let consoleOutput: string[] = [];
  const originalLog = console.log;

  beforeEach(() => {
    consoleOutput = [];
    console.log = (...args: any[]) => {
      consoleOutput.push(args.join(' '));
    };
  });

  afterEach(() => {
    console.log = originalLog;
  });

  describe('Command Suggestions', () => {
    test('suggests "analyze" for "analyz"', () => {
      const suggestion = helpSystem.suggestCommand('analyz');
      expect(suggestion).toBe('analyze');
    });

    test('suggests "verify" for "verifiy"', () => {
      const suggestion = helpSystem.suggestCommand('verifiy');
      expect(suggestion).toBe('verify');
    });

    test('suggests "provider" for "provder"', () => {
      const suggestion = helpSystem.suggestCommand('provder');
      expect(suggestion).toBe('provider');
    });

    test('suggests "config" for "confg"', () => {
      const suggestion = helpSystem.suggestCommand('confg');
      expect(suggestion).toBe('config');
    });

    test('suggests "history" for "histry"', () => {
      const suggestion = helpSystem.suggestCommand('histry');
      expect(suggestion).toBe('history');
    });

    test('suggests "export" for "exprt"', () => {
      const suggestion = helpSystem.suggestCommand('exprt');
      expect(suggestion).toBe('export');
    });

    test('suggests "help" for "hlp"', () => {
      const suggestion = helpSystem.suggestCommand('hlp');
      expect(suggestion).toBe('help');
    });

    test('returns null for completely unknown command', () => {
      const suggestion = helpSystem.suggestCommand('xyzabc123');
      expect(suggestion).toBeNull();
    });

    test('handles case insensitivity', () => {
      expect(helpSystem.suggestCommand('ANALYZ')).toBe('analyze');
      expect(helpSystem.suggestCommand('VeRiFy')).toBe('verify');
    });

    test('handles empty string', () => {
      const suggestion = helpSystem.suggestCommand('');
      expect(suggestion).toBeNull();
    });

    test('distance threshold works correctly', () => {
      // Distance > 3 should return null
      const suggestion = helpSystem.suggestCommand('zzzzz');
      expect(suggestion).toBeNull();
    });
  });

  describe('Main Help Display', () => {
    test('displays main help without errors', () => {
      expect(() => helpSystem.showMainHelp()).not.toThrow();
    });

    test('includes logo in output', () => {
      helpSystem.showMainHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('NOVA MEDICINA');
    });

    test('includes safety warning', () => {
      helpSystem.showMainHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('CRITICAL SAFETY WARNING');
      expect(output).toContain('NOT a substitute');
    });

    test('includes all main commands', () => {
      helpSystem.showMainHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('analyze');
      expect(output).toContain('verify');
      expect(output).toContain('provider');
      expect(output).toContain('config');
      expect(output).toContain('history');
      expect(output).toContain('export');
    });

    test('includes quick start guide', () => {
      helpSystem.showMainHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('QUICK START');
    });

    test('includes resource links', () => {
      helpSystem.showMainHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('Documentation:');
      expect(output).toContain('GitHub:');
      expect(output).toContain('Support:');
    });
  });

  describe('Analyze Help Display', () => {
    test('displays analyze help without errors', () => {
      expect(() => helpSystem.showAnalyzeHelp()).not.toThrow();
    });

    test('includes command description', () => {
      helpSystem.showAnalyzeHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('ANALYZE COMMAND');
      expect(output).toContain('symptoms');
    });

    test('includes all options', () => {
      helpSystem.showAnalyzeHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('--severity');
      expect(output).toContain('--age');
      expect(output).toContain('--conditions');
      expect(output).toContain('--emergency');
    });

    test('includes practical examples', () => {
      helpSystem.showAnalyzeHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('EXAMPLES');
      expect(output).toContain('nova-medicina analyze');
    });

    test('includes emergency indicators', () => {
      helpSystem.showAnalyzeHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('EMERGENCY INDICATORS');
      expect(output).toContain('chest pain');
      expect(output).toContain('CALL 911');
    });

    test('includes symptom categories', () => {
      helpSystem.showAnalyzeHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('SYMPTOM CATEGORIES');
      expect(output).toContain('Cardiovascular');
      expect(output).toContain('Neurological');
    });
  });

  describe('Verify Help Display', () => {
    test('displays verify help without errors', () => {
      expect(() => helpSystem.showVerifyHelp()).not.toThrow();
    });

    test('includes verification models', () => {
      helpSystem.showVerifyHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('VERIFICATION MODELS');
      expect(output).toContain('GPT-4o');
      expect(output).toContain('Claude');
      expect(output).toContain('Gemini');
    });

    test('includes confidence scoring explanation', () => {
      helpSystem.showVerifyHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('CONFIDENCE SCORING');
      expect(output).toContain('90-100%');
      expect(output).toContain('Very High');
    });

    test('includes verification process steps', () => {
      helpSystem.showVerifyHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('VERIFICATION PROCESS');
    });

    test('includes examples', () => {
      helpSystem.showVerifyHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('EXAMPLES');
      expect(output).toContain('nova-medicina verify');
    });
  });

  describe('Provider Help Display', () => {
    test('displays provider help without errors', () => {
      expect(() => helpSystem.showProviderHelp()).not.toThrow();
    });

    test('includes subcommands', () => {
      helpSystem.showProviderHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('add');
      expect(output).toContain('list');
      expect(output).toContain('remove');
      expect(output).toContain('notify');
      expect(output).toContain('status');
    });

    test('includes notification options', () => {
      helpSystem.showProviderHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('NOTIFICATION OPTIONS');
      expect(output).toContain('email');
      expect(output).toContain('sms');
    });

    test('includes privacy information', () => {
      helpSystem.showProviderHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('PRIVACY');
      expect(output).toContain('HIPAA');
      expect(output).toContain('encrypted');
    });
  });

  describe('Config Help Display', () => {
    test('displays config help without errors', () => {
      expect(() => helpSystem.showConfigHelp()).not.toThrow();
    });

    test('includes setup guide', () => {
      helpSystem.showConfigHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('SETUP GUIDE');
      expect(output).toContain('Step 1');
    });

    test('includes API key configuration', () => {
      helpSystem.showConfigHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('--api-key');
      expect(output).toContain('--openai-key');
      expect(output).toContain('--anthropic-key');
    });

    test('includes file locations', () => {
      helpSystem.showConfigHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('CONFIGURATION LOCATIONS');
      expect(output).toContain('.nova-medicina');
    });

    test('includes security best practices', () => {
      helpSystem.showConfigHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('SECURITY');
      expect(output).toContain('Never share');
    });

    test('includes environment variables', () => {
      helpSystem.showConfigHelp();
      const output = consoleOutput.join('\n');
      expect(output).toContain('ENVIRONMENT VARIABLES');
      expect(output).toContain('NOVA_MEDICINA_API_KEY');
    });
  });

  describe('Context-Sensitive Help', () => {
    test('shows main help for empty context', () => {
      helpSystem.showContextHelp([]);
      const output = consoleOutput.join('\n');
      expect(output).toContain('OVERVIEW');
    });

    test('shows analyze help for "analyze" context', () => {
      helpSystem.showContextHelp(['analyze']);
      const output = consoleOutput.join('\n');
      expect(output).toContain('ANALYZE COMMAND');
    });

    test('shows verify help for "verify" context', () => {
      helpSystem.showContextHelp(['verify']);
      const output = consoleOutput.join('\n');
      expect(output).toContain('VERIFY COMMAND');
    });

    test('shows provider help for "provider" context', () => {
      helpSystem.showContextHelp(['provider']);
      const output = consoleOutput.join('\n');
      expect(output).toContain('PROVIDER COMMAND');
    });

    test('shows config help for "config" context', () => {
      helpSystem.showContextHelp(['config']);
      const output = consoleOutput.join('\n');
      expect(output).toContain('CONFIG COMMAND');
    });

    test('suggests command for unknown context', () => {
      helpSystem.showContextHelp(['analyz']);
      const output = consoleOutput.join('\n');
      expect(output).toContain('Unknown command');
      expect(output).toContain('Did you mean');
      expect(output).toContain('analyze');
    });
  });

  describe('Provider Contacts Display', () => {
    test('displays empty state message', () => {
      helpSystem.showProviderContacts([]);
      const output = consoleOutput.join('\n');
      expect(output).toContain('No providers configured');
    });

    test('displays single provider', () => {
      const providers = [
        {
          name: 'Dr. Jane Smith',
          email: 'jsmith@clinic.com',
          phone: '555-0100',
          specialty: 'Family Medicine',
          primary: true,
          emergency: false
        }
      ];
      helpSystem.showProviderContacts(providers);
      const output = consoleOutput.join('\n');
      expect(output).toContain('Dr. Jane Smith');
      expect(output).toContain('jsmith@clinic.com');
      expect(output).toContain('555-0100');
    });

    test('displays multiple providers', () => {
      const providers = [
        {
          name: 'Dr. Jane Smith',
          email: 'jsmith@clinic.com',
          primary: true,
          emergency: false
        },
        {
          name: 'City Hospital ER',
          phone: '555-0911',
          primary: false,
          emergency: true
        }
      ];
      helpSystem.showProviderContacts(providers);
      const output = consoleOutput.join('\n');
      expect(output).toContain('Dr. Jane Smith');
      expect(output).toContain('City Hospital ER');
    });

    test('highlights primary provider', () => {
      const providers = [
        {
          name: 'Dr. Jane Smith',
          email: 'jsmith@clinic.com',
          primary: true,
          emergency: false
        }
      ];
      helpSystem.showProviderContacts(providers);
      const output = consoleOutput.join('\n');
      expect(output).toContain('PRIMARY');
    });

    test('highlights emergency contact', () => {
      const providers = [
        {
          name: 'City Hospital ER',
          phone: '555-0911',
          primary: false,
          emergency: true
        }
      ];
      helpSystem.showProviderContacts(providers);
      const output = consoleOutput.join('\n');
      expect(output).toContain('EMERGENCY');
    });
  });

  describe('Tutorial Mode', () => {
    test('runs tutorial without errors', async () => {
      await expect(helpSystem.runTutorial()).resolves.not.toThrow();
    });

    test('includes tutorial steps', async () => {
      await helpSystem.runTutorial();
      const output = consoleOutput.join('\n');
      expect(output).toContain('INTERACTIVE TUTORIAL');
      expect(output).toContain('Step 1');
      expect(output).toContain('Tutorial Complete');
    });

    test('includes safety information in tutorial', async () => {
      await helpSystem.runTutorial();
      const output = consoleOutput.join('\n');
      expect(output).toContain('Emergency');
      expect(output).toContain('911');
    });
  });

  describe('Constants', () => {
    test('exports LOGO constant', () => {
      expect(helpSystem.LOGO).toBeDefined();
      expect(typeof helpSystem.LOGO).toBe('string');
      expect(helpSystem.LOGO).toContain('NOVA');
    });

    test('exports SAFETY_WARNING constant', () => {
      expect(helpSystem.SAFETY_WARNING).toBeDefined();
      expect(typeof helpSystem.SAFETY_WARNING).toBe('string');
      expect(helpSystem.SAFETY_WARNING).toContain('WARNING');
    });
  });

  describe('Levenshtein Distance Edge Cases', () => {
    test('handles identical strings', () => {
      const suggestion = helpSystem.suggestCommand('analyze');
      expect(suggestion).toBe('analyze');
    });

    test('handles single character difference', () => {
      const suggestion = helpSystem.suggestCommand('analyzr');
      expect(suggestion).toBe('analyze');
    });

    test('handles transposed characters', () => {
      const suggestion = helpSystem.suggestCommand('analzye');
      expect(suggestion).toBe('analyze');
    });

    test('handles missing characters', () => {
      const suggestion = helpSystem.suggestCommand('analye');
      expect(suggestion).toBe('analyze');
    });

    test('handles extra characters', () => {
      const suggestion = helpSystem.suggestCommand('analyzee');
      expect(suggestion).toBe('analyze');
    });
  });

  describe('Error Handling', () => {
    test('handles malformed input gracefully', () => {
      expect(() => helpSystem.suggestCommand('!@#$%')).not.toThrow();
    });

    test('handles very long strings', () => {
      const longString = 'a'.repeat(1000);
      expect(() => helpSystem.suggestCommand(longString)).not.toThrow();
    });

    test('handles special characters', () => {
      expect(() => helpSystem.suggestCommand('anályze')).not.toThrow();
    });
  });
});
