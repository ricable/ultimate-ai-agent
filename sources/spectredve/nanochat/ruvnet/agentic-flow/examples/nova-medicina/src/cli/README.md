# Nova Medicina CLI Help System

## Quick Start

```bash
# Install dependencies
npm install

# Run CLI
npm start

# Show help
npm start -- --help

# Run tutorial
npm start -- --tutorial

# Test command suggestion
npm start -- analyz  # Suggests "analyze"
```

## Files

### Core Implementation
- **`help-system.ts`** (818 lines) - Comprehensive help system with:
  - ASCII art logo
  - Color-coded sections
  - Command-specific help (analyze, verify, provider, config)
  - Interactive tutorial mode
  - Command suggestion for typos
  - Safety warnings and disclaimers
  - Provider contact display

### Integration
- **`index.ts`** - CLI entry point showing help system integration with Commander.js

## Features

### 1. Main Help (`showMainHelp()`)
- Overview of Nova Medicina
- Quick start guide
- Command reference
- Resource links

### 2. Command-Specific Help
- `showAnalyzeHelp()` - Symptom analysis with examples
- `showVerifyHelp()` - AI verification and confidence scoring
- `showProviderHelp()` - Provider management and notifications
- `showConfigHelp()` - Configuration and setup guide

### 3. Interactive Features
- `runTutorial()` - 8-step guided walkthrough
- `suggestCommand()` - Typo correction using Levenshtein distance
- `showContextHelp()` - Context-aware help display

### 4. Safety Features
- Prominent safety warnings
- Emergency symptom indicators
- 911 reminders
- HIPAA compliance information

### 5. Visual Design
- Color-coded sections (cyan, green, yellow, red)
- Boxed content using `boxen`
- ASCII art logo
- Emoji icons for visual cues
- Clear hierarchy and spacing

## Usage Examples

### Display Main Help
```typescript
import helpSystem from './help-system';

helpSystem.showMainHelp();
```

### Show Command Help
```typescript
helpSystem.showAnalyzeHelp();
helpSystem.showVerifyHelp();
helpSystem.showProviderHelp();
helpSystem.showConfigHelp();
```

### Command Suggestions
```typescript
const suggestion = helpSystem.suggestCommand('analyz');
console.log(suggestion); // "analyze"

const unknown = helpSystem.suggestCommand('xyzabc');
console.log(unknown); // null
```

### Run Tutorial
```typescript
await helpSystem.runTutorial();
```

### Display Provider Contacts
```typescript
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
```

### Context-Sensitive Help
```typescript
// Show help for specific command
helpSystem.showContextHelp(['analyze']);

// Unknown command with suggestion
helpSystem.showContextHelp(['analyz']);
```

## Integration with Commander.js

```typescript
import { Command } from 'commander';
import helpSystem from './help-system';

const program = new Command();

// Override default help
program
  .name('nova-medicina')
  .addHelpText('beforeAll', () => {
    helpSystem.showMainHelp();
    return '';
  });

// Add command with custom help
program
  .command('analyze <symptoms>')
  .addHelpText('beforeAll', () => {
    helpSystem.showAnalyzeHelp();
    return '';
  })
  .action((symptoms, options) => {
    // Implementation
  });

// Handle unknown commands
program.on('command:*', (operands) => {
  const suggestion = helpSystem.suggestCommand(operands[0]);
  if (suggestion) {
    console.log(`Did you mean: ${suggestion}?`);
  }
});
```

## Dependencies

```json
{
  "chalk": "^5.3.0",    // Terminal colors
  "boxen": "^7.1.1",    // Boxed content
  "commander": "^11.1.0" // CLI framework
}
```

## Testing

```typescript
import helpSystem from './help-system';

describe('Help System', () => {
  test('suggests correct command', () => {
    expect(helpSystem.suggestCommand('analyz')).toBe('analyze');
    expect(helpSystem.suggestCommand('verifiy')).toBe('verify');
    expect(helpSystem.suggestCommand('xyzabc')).toBeNull();
  });

  test('displays help without errors', () => {
    expect(() => helpSystem.showMainHelp()).not.toThrow();
    expect(() => helpSystem.showAnalyzeHelp()).not.toThrow();
  });
});
```

## Safety Compliance

### Medical Disclaimers
- Displayed in main help
- Repeated in command-specific help
- Prominent warning boxes
- Emergency indicators

### Emergency Handling
- 10 critical symptoms requiring immediate attention
- 911 reminders in multiple locations
- Emergency provider contacts
- Urgent notification options

### Privacy & Security
- HIPAA compliance notes
- Encryption information
- API key security practices
- Audit trail documentation

## Color Scheme

| Color | Usage |
|-------|-------|
| Cyan | Headings, primary commands |
| Green | Success, recommended actions |
| Yellow | Warnings, important notes |
| Red | Errors, emergencies |
| Blue | Links, references |
| Gray/Dim | Secondary information |

## Visual Elements

### ASCII Logo
```
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗     ███╗   ███╗███████╗██████╗ ██╗ ██████╗██╗███╗   ██╗ █████╗
...
```

### Safety Warning Box
```
╔════════════════════════════════════╗
║ ⚠️  CRITICAL SAFETY WARNING ⚠️     ║
║                                    ║
║ This tool is NOT a substitute...   ║
╚════════════════════════════════════╝
```

### Provider Contact Cards
```
╭──────────────────────────────────╮
│ Dr. Jane Smith ★ PRIMARY         │
│                                  │
│ Specialty: Family Medicine       │
│ Email: jsmith@clinic.com         │
│ Phone: 555-0100                  │
╰──────────────────────────────────╯
```

## Command Reference

### Main Commands
- `analyze` - Symptom analysis
- `verify` - AI verification
- `provider` - Provider management
- `config` - Configuration
- `history` - Analysis history
- `export` - Export data
- `tutorial` - Interactive tutorial
- `help` - Context-sensitive help

### Provider Subcommands
- `provider add` - Add provider
- `provider list` - List providers
- `provider remove` - Remove provider
- `provider notify` - Send notification
- `provider status` - Check notification status

### Config Subcommands
- `config set` - Set value
- `config get` - Get value
- `config list` - List all settings

## Content Structure

Each help section follows:
1. **Title/Logo** - Identification
2. **Description** - Brief overview
3. **Usage** - Syntax patterns
4. **Options** - Parameter reference
5. **Examples** - Practical scenarios (4+ examples)
6. **Warnings** - Safety notes
7. **Additional Info** - Context details

## Future Enhancements

- [ ] Search functionality
- [ ] Man page generation
- [ ] HTML export
- [ ] Video tutorials
- [ ] Contextual tips
- [ ] Version-specific help
- [ ] Plugin documentation
- [ ] Offline mode
- [ ] Internationalization (i18n)
- [ ] Screen reader support
- [ ] High contrast mode

## Maintenance

### Adding New Commands
1. Create help function (e.g., `showNewCommandHelp()`)
2. Add to `showContextHelp()` switch
3. Update command list in `suggestCommand()`
4. Add minimum 4 examples
5. Include safety warnings if relevant
6. Update main help command list

### Updating Content
- Keep examples realistic
- Test all command syntax
- Include edge cases
- Show error handling
- Follow style guide

## Performance

- Help text generated on-demand
- No heavy computations
- Minimal memory footprint
- Fast Levenshtein distance calculation

## Statistics

- **Total Lines:** 818
- **Functions:** 10 exported
- **Help Sections:** 5 (main + 4 commands)
- **Examples:** 20+
- **Safety Warnings:** 3 critical sections
- **Commands Covered:** 8 main + subcommands

## Support

- **Documentation:** Full docs in `/docs/cli-help-system.md`
- **GitHub:** https://github.com/ruvnet/nova-medicina
- **Issues:** https://github.com/ruvnet/nova-medicina/issues

---

**Version:** 1.0.0
**Last Updated:** 2025-11-08
**Status:** ✅ Production Ready
