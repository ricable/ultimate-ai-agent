# Nova Medicina Help System - Quick Start

## Installation

```bash
cd /home/user/agentic-flow/nova-medicina
npm install
```

## Basic Usage

### Show Main Help
```bash
npm start -- --help
# or
node src/cli/index.js --help
```

### Show Command-Specific Help
```bash
npm start -- analyze --help
npm start -- verify --help
npm start -- provider --help
npm start -- config --help
```

### Run Interactive Tutorial
```bash
npm start -- --tutorial
```

### Test Command Suggestions
```bash
# Typo: "analyz" → suggests "analyze"
npm start -- analyz

# Typo: "verifiy" → suggests "verify"
npm start -- verifiy

# Typo: "provder" → suggests "provider"
npm start -- provder
```

## From Code

```typescript
// Import help system
import helpSystem from './src/cli/help-system';

// Display help
helpSystem.showMainHelp();
helpSystem.showAnalyzeHelp();
helpSystem.showVerifyHelp();
helpSystem.showProviderHelp();
helpSystem.showConfigHelp();

// Get command suggestions
const suggestion = helpSystem.suggestCommand('analyz');
console.log(suggestion); // "analyze"

// Run interactive tutorial
await helpSystem.runTutorial();

// Show provider contacts
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

// Context-sensitive help
helpSystem.showContextHelp(['analyze']);
helpSystem.showContextHelp(['unknown-command']); // Shows suggestion
```

## Testing

```bash
# Run all tests
npm test tests/cli/help-system.test.ts

# Run specific test suite
npm test -- --testNamePattern="Command Suggestions"
npm test -- --testNamePattern="Help Display"
```

## File Locations

```
/home/user/agentic-flow/nova-medicina/
├── src/cli/
│   ├── help-system.ts     (818 lines - main implementation)
│   ├── index.ts           (283 lines - CLI integration)
│   └── README.md          (quick reference)
├── tests/cli/
│   └── help-system.test.ts (473 lines - test suite)
├── docs/
│   └── cli-help-system.md  (440 lines - full documentation)
└── IMPLEMENTATION_SUMMARY.md (complete summary)
```

## Key Features

- ASCII art logo
- Color-coded sections
- 5 help sections (main + 4 commands)
- 20+ practical examples
- Interactive tutorial (8 steps)
- Command suggestions (typo correction)
- Safety warnings
- Provider contact display
- 50+ test cases

## Dependencies

```json
{
  "chalk": "^5.3.0",    // Terminal colors
  "boxen": "^7.1.1",    // Boxed content
  "commander": "^11.1.0" // CLI framework
}
```

## Quick Examples

### Analyze Symptoms
```bash
npm start -- analyze "headache and fever" --age 45
npm start -- analyze "chest pain" --emergency
```

### Verify Analysis
```bash
npm start -- verify abc123def456
npm start -- verify abc123def456 --explain --min-confidence 85
```

### Manage Providers
```bash
npm start -- provider add --name "Dr. Smith" --email "dr@clinic.com" --primary
npm start -- provider list
npm start -- provider notify abc123 --urgent
```

### Configure CLI
```bash
npm start -- config --api-key "your-key-here"
npm start -- config set language es
npm start -- config list
```

## Next Steps

1. Review `/docs/cli-help-system.md` for full documentation
2. Run `npm test` to verify implementation
3. Try `npm start -- --tutorial` for guided walkthrough
4. Integrate with main CLI application
5. Add custom commands following the pattern

## Support

- Full docs: `/docs/cli-help-system.md`
- Implementation: `/src/cli/help-system.ts`
- Tests: `/tests/cli/help-system.test.ts`
- Summary: `/IMPLEMENTATION_SUMMARY.md`
