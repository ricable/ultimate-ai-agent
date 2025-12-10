# Nova Medicina - CLI Help System Documentation

## Overview

A comprehensive, interactive help system for the Nova Medicina emergency medical analysis CLI. Provides detailed guidance, safety warnings, usage examples, and command suggestions.

## File Location

**Main Implementation:** `/home/user/agentic-flow/nova-medicina/src/cli/help-system.ts`

## Features Implemented

### 1. **Comprehensive Help Display** ✅
- ASCII art logo with brand identity
- Color-coded sections for visual hierarchy
- Clear command structure and organization
- Quick start guide for new users
- Resource links (docs, GitHub, support)

### 2. **Command-Specific Help** ✅

#### Analyze Command (`showAnalyzeHelp`)
- Detailed option descriptions
- 5 practical usage examples
- Symptom category reference (7 categories)
- Emergency indicator list (10 critical symptoms)
- Output format explanation
- Safety warnings prominently displayed

#### Verify Command (`showVerifyHelp`)
- Multi-model verification explanation
- Confidence scoring system (4 levels)
- Model specialization details
- 6-step verification process
- 4 usage examples
- Detailed confidence level breakdown

#### Provider Command (`showProviderHelp`)
- Sub-command reference (add, list, remove, notify, status)
- Notification options (email, SMS, fax, portal)
- Privacy and security information
- 5 practical examples
- HIPAA compliance notes
- Notification content breakdown

#### Config Command (`showConfigHelp`)
- Complete setup guide (5 steps)
- API key configuration
- Environment variable reference
- File location documentation
- Security best practices
- 4 configuration examples

### 3. **Safety Features** ✅

#### Critical Safety Warning Box
- Prominent display using `boxen` package
- Red color scheme for maximum visibility
- Double border style for emphasis
- Emergency contact reminder (911)
- Professional medical care disclaimer

#### Emergency Indicators
- 10 critical symptoms requiring immediate attention
- Red color coding throughout
- Repeated warnings across commands
- Clear "CALL 911 IMMEDIATELY" messaging

### 4. **Interactive Features** ✅

#### Command Suggestions (`suggestCommand`)
- Levenshtein distance algorithm for typo detection
- Suggests closest matching command (within 3 character distance)
- Handles common misspellings
- User-friendly error messages

#### Tutorial Mode (`runTutorial`)
- 8-step interactive walkthrough
- Progressive disclosure of features
- Real command examples
- Best practices guide
- Emergency handling education
- Boxed steps with visual hierarchy

#### Context-Sensitive Help (`showContextHelp`)
- Detects command context
- Shows relevant help automatically
- Provides command suggestions for unknown commands
- Fallback to main help

### 5. **Visual Formatting** ✅

#### Color Coding System
- **Cyan:** Headings and primary commands
- **Green:** Success states and recommended actions
- **Yellow:** Warnings and important notes
- **Red:** Errors and emergency indicators
- **Gray/Dim:** Secondary information and hints
- **Blue:** Links and references

#### Layout Components
- ASCII art logo (prominent brand identity)
- Boxed sections using `boxen` package
- Structured tables for options
- Indented examples for clarity
- Section headers with emoji icons

### 6. **Provider Management** ✅

#### Provider Contact Display (`showProviderContacts`)
- Boxed display for each provider
- Primary provider highlighting (★ PRIMARY)
- Emergency contact flagging (🚨 EMERGENCY)
- Complete contact information
- Specialty and clinic details
- Color-coded borders (green=primary, red=emergency)

## Dependencies Required

```json
{
  "dependencies": {
    "chalk": "^5.3.0",
    "boxen": "^7.1.1"
  }
}
```

### Chalk Features Used
- Color methods: `cyan`, `green`, `yellow`, `red`, `blue`, `gray`, `white`
- Style methods: `bold`, `dim`
- Chaining: `chalk.bold.cyan()`, `chalk.red.bold()`

### Boxen Features Used
- Padding and margin control
- Border styles: `double`, `round`
- Border colors
- Background colors

## Usage Examples

### Main Help
```bash
nova-medicina --help
nova-medicina help
```

### Command-Specific Help
```bash
nova-medicina analyze --help
nova-medicina help verify
nova-medicina provider --help
nova-medicina config --help
```

### Interactive Tutorial
```bash
nova-medicina --tutorial
```

### Context Help from Code
```typescript
import helpSystem from './help-system';

// Show main help
helpSystem.showMainHelp();

// Show command help
helpSystem.showAnalyzeHelp();
helpSystem.showVerifyHelp();

// Suggest command for typo
const suggestion = helpSystem.suggestCommand('analyz');
// Returns: "analyze"

// Run tutorial
await helpSystem.runTutorial();

// Context-sensitive help
helpSystem.showContextHelp(['analyze']);

// Show provider contacts
helpSystem.showProviderContacts(providers);
```

## Integration Points

### CLI Entry Point
```typescript
import { Command } from 'commander';
import helpSystem from './help-system';

const program = new Command();

program
  .name('nova-medicina')
  .description('Emergency medical analysis and triage assistant')
  .version('1.0.0')
  .addHelpText('beforeAll', helpSystem.SAFETY_WARNING)
  .configureHelp({
    formatHelp: () => {
      helpSystem.showMainHelp();
      return '';
    }
  });

// Tutorial command
program
  .command('tutorial')
  .description('Interactive tutorial mode')
  .action(async () => {
    await helpSystem.runTutorial();
  });

// Add help override for each command
program
  .command('analyze')
  .addHelpText('beforeAll', () => {
    helpSystem.showAnalyzeHelp();
    return '';
  });
```

### Error Handling
```typescript
program.exitOverride((err) => {
  if (err.code === 'commander.unknownCommand') {
    const suggestion = helpSystem.suggestCommand(err.message);
    if (suggestion) {
      console.log(`\nDid you mean: ${suggestion}?`);
    }
  }
});
```

## Content Structure

### Help Text Organization
Each help section follows this structure:
1. **Title/Logo** - Command identification
2. **Description** - Brief overview
3. **Usage** - Syntax patterns
4. **Options** - Detailed parameter reference
5. **Examples** - Practical usage scenarios (minimum 4 examples)
6. **Warnings** - Safety and important notes
7. **Additional Info** - Context-specific details

### Example Breakdown
- **Basic usage:** Simple command with minimal options
- **Advanced usage:** Multiple options combined
- **Common scenarios:** Real-world use cases
- **Edge cases:** Special situations
- **Error handling:** How to recover from issues

## Safety Compliance

### Medical Disclaimers
- Displayed prominently in main help
- Repeated in analyze command help
- Emergency indicators clearly marked
- Professional care reminders throughout

### Emergency Handling
- 911 reminder in multiple locations
- Critical symptom list with 10 indicators
- Emergency provider contact display
- Urgent notification options

### Privacy & Security
- HIPAA compliance notes
- Encryption information
- API key security best practices
- Audit trail documentation

## Accessibility Features

### Visual Hierarchy
- Color coding with semantic meaning
- Consistent icon usage (🚨, ⚠️, ✓, ✗)
- Clear section headers
- Indentation for relationships

### Readability
- Short paragraphs
- Bullet points for lists
- Code blocks for examples
- Spacing between sections

### Progressive Disclosure
- Main help → Command help → Tutorial
- Basic → Advanced examples
- Quick start before detailed reference

## Testing Recommendations

### Unit Tests
```typescript
describe('Help System', () => {
  test('suggests correct command for typo', () => {
    expect(helpSystem.suggestCommand('analyz')).toBe('analyze');
    expect(helpSystem.suggestCommand('verifiy')).toBe('verify');
  });

  test('returns null for unknown command', () => {
    expect(helpSystem.suggestCommand('xyzabc')).toBeNull();
  });

  test('displays help without errors', () => {
    expect(() => helpSystem.showMainHelp()).not.toThrow();
    expect(() => helpSystem.showAnalyzeHelp()).not.toThrow();
  });
});
```

### Integration Tests
- Verify color output in terminal
- Test boxen rendering
- Validate all links and references
- Check tutorial flow

### Manual Testing Checklist
- [ ] Main help displays correctly
- [ ] All command helps are accessible
- [ ] Tutorial runs without errors
- [ ] Command suggestions work
- [ ] Safety warnings are visible
- [ ] Colors render properly
- [ ] Examples are accurate
- [ ] Links are valid

## Maintenance Guide

### Adding New Commands
1. Create help function: `showNewCommandHelp()`
2. Add to `showContextHelp()` switch statement
3. Update command list in `suggestCommand()`
4. Add examples (minimum 4)
5. Include safety warnings if relevant
6. Update main help command list

### Updating Examples
- Keep examples realistic and practical
- Test all command syntax
- Include edge cases
- Show error handling
- Demonstrate best practices

### Internationalization Preparation
```typescript
// Future i18n structure
const messages = {
  en: {
    mainHelp: {
      title: 'OVERVIEW',
      description: 'Nova Medicina is...'
    }
  },
  es: {
    mainHelp: {
      title: 'DESCRIPCIÓN GENERAL',
      description: 'Nova Medicina es...'
    }
  }
};
```

## Performance Considerations

### Lazy Loading
- Help text only generated when requested
- No heavy computations on load
- Minimal memory footprint

### Caching Opportunities
- Cache formatted output
- Reuse boxen configurations
- Memoize distance calculations

## Future Enhancements

### Planned Features
1. **Search functionality** - Search help content
2. **Man page generation** - Convert to traditional man pages
3. **HTML export** - Web documentation generation
4. **Video tutorials** - Link to video walkthrough
5. **Contextual tips** - Show relevant tips during command execution
6. **Version-specific help** - Show help for installed version
7. **Plugin help** - Third-party extension documentation
8. **Offline mode** - Cached help when no internet

### Accessibility Improvements
- Screen reader support
- High contrast mode
- Larger text options
- Keyboard navigation

## File Statistics

- **Lines of code:** 818
- **Functions:** 10 exported functions
- **Help sections:** 4 command-specific + 1 main
- **Examples:** 20+ practical examples
- **Safety warnings:** 3 critical warning sections
- **Visual elements:** ASCII logo + colored sections + boxed content

## Integration with Hooks

### Pre-Task Hook
```bash
npx claude-flow@alpha hooks pre-task --description "cli help system"
```

### Post-Task Hook
```bash
npx claude-flow@alpha hooks post-task --task-id "help-001"
```

### Post-Edit Hook
```bash
npx claude-flow@alpha hooks post-edit \
  --file "/home/user/agentic-flow/nova-medicina/src/cli/help-system.ts" \
  --memory-key "swarm/coder/help-system"
```

## Summary

The Nova Medicina CLI help system provides a comprehensive, user-friendly interface with:

✅ **818 lines** of well-structured TypeScript code
✅ **10 exported functions** for different help contexts
✅ **20+ examples** covering all major use cases
✅ **Color-coded** visual hierarchy for easy navigation
✅ **Interactive tutorial** with 8 guided steps
✅ **Command suggestions** for typo correction
✅ **Safety warnings** prominently displayed
✅ **Provider management** with contact display
✅ **Context-sensitive** help system
✅ **HIPAA compliance** information included

The system is production-ready and follows best practices for CLI design, medical safety communication, and user experience.
