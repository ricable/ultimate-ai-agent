# Agent Booster NPM Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    User Application                             │
│  require('agent-booster') or import * as agentBooster from ... │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                 Auto-Detection Loader                           │
│                    (index.js)                                   │
│                                                                  │
│  1. Check if runtime already loaded (cached)                   │
│  2. Try loading native bindings for platform                   │
│  3. Fall back to WASM if native not available                  │
│  4. Throw helpful error if nothing available                   │
└────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌───────────────────┐  ┌───────────────────┐
        │ Try Load Native   │  │  Try Load WASM    │
        │  Bindings         │  │   Bindings        │
        └───────────────────┘  └───────────────────┘
                    │                   │
          ┌─────────┴─────────┐        │
          ▼                   ▼        ▼
    ┌──────────┐        ┌──────────┐  ┌──────────┐
    │  macOS   │        │  Linux   │  │  WASM    │
    │ darwin-  │        │ linux-   │  │ (nodejs) │
    │ arm64/x64│        │ arm64/x64│  │          │
    └──────────┘        └──────────┘  └──────────┘
          │                   │              │
          └───────────────────┴──────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Runtime Instance     │
                  │  (Native or WASM)     │
                  └───────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Public API Methods   │
                  │  • apply()            │
                  │  • batchApply()       │
                  │  • analyze()          │
                  │  • getVersion()       │
                  │  • getRuntime()       │
                  └───────────────────────┘
```

## Package Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      agent-booster                           │
│                  (Main SDK Package)                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Dependencies:                                               │
│  └─ @agent-booster/wasm ^0.1.0                             │
│                                                               │
│  Optional Dependencies:                                      │
│  ├─ @agent-booster/native-darwin-arm64 ^0.1.0             │
│  ├─ @agent-booster/native-darwin-x64 ^0.1.0               │
│  ├─ @agent-booster/native-linux-arm64 ^0.1.0              │
│  ├─ @agent-booster/native-linux-x64 ^0.1.0                │
│  └─ @agent-booster/native-win32-x64 ^0.1.0                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  agent-booster-cli                           │
│                  (CLI Package)                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Dependencies:                                               │
│  ├─ agent-booster ^0.1.0                                    │
│  ├─ commander ^11.0.0                                       │
│  ├─ chalk ^4.1.2                                            │
│  └─ ora ^5.4.1                                              │
│                                                               │
│  Commands:                                                   │
│  ├─ apply     - Single prompt optimization                  │
│  ├─ batch     - Batch file processing                       │
│  ├─ analyze   - Prompt analysis                             │
│  └─ version   - Version information                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Runtime Detection Flow

```
┌─────────────────────────────────────────────────────────────┐
│  initialize()                                                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Is runtime cached?    │
        └───────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
          [YES]           [NO]
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ Get platform and arch    │
            │   │ os.platform(), os.arch() │
            │   └──────────────────────────┘
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ Map to package name      │
            │   │ e.g., darwin-arm64 →     │
            │   │ @agent-booster/native-   │
            │   │ darwin-arm64             │
            │   └──────────────────────────┘
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ Try require(packageName) │
            │   └──────────────────────────┘
            │               │
            │       ┌───────┴───────┐
            │       ▼               ▼
            │   [SUCCESS]        [FAIL]
            │       │               │
            │       │               ▼
            │       │   ┌──────────────────────────┐
            │       │   │ Try require(             │
            │       │   │ '@agent-booster/wasm')   │
            │       │   └──────────────────────────┘
            │       │               │
            │       │       ┌───────┴───────┐
            │       │       ▼               ▼
            │       │   [SUCCESS]        [FAIL]
            │       │       │               │
            │       │       │               ▼
            │       │       │   ┌──────────────────────────┐
            │       │       │   │ Throw Error:             │
            │       │       │   │ "No runtime available"   │
            │       │       │   └──────────────────────────┘
            │       │       │
            └───────┴───────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Cache runtime         │
        │ Set loadedRuntime     │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Return runtime        │
        └───────────────────────┘
```

## API Call Flow

### Single Optimization: `apply(prompt, options)`

```
┌─────────────────────────────────────────────────────────────┐
│  agentBooster.apply(prompt, options)                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ initialize()          │
        │ (auto-detect runtime) │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Validate method       │
        │ exists in runtime     │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ runtime.apply(        │
        │   prompt, options)    │
        └───────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
      [Native Call]    [WASM Call]
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ WASM Module              │
            │   │ • Parse options          │
            │   │ • Optimize prompt        │
            │   │ • Return result          │
            │   └──────────────────────────┘
            │               │
            └───────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Return optimized      │
        │ prompt string         │
        └───────────────────────┘
```

### Batch Optimization: `batchApply(prompts, options)`

```
┌─────────────────────────────────────────────────────────────┐
│  agentBooster.batchApply(prompts, options)                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ initialize()          │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────┐
        │ Does runtime have batchApply()?   │
        └───────────────────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
          [YES]           [NO]
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ Fall back to sequential  │
            │   │ for (prompt of prompts)  │
            │   │   results.push(          │
            │   │     await apply(prompt)) │
            │   └──────────────────────────┘
            │               │
            ▼               │
  ┌──────────────────────┐ │
  │ runtime.batchApply(  │ │
  │   prompts, options)  │ │
  └──────────────────────┘ │
            │               │
            └───────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Return array of       │
        │ optimized prompts     │
        └───────────────────────┘
```

## CLI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  $ agent-booster <command> [options]                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ bin/agent-booster.js  │
        │ (Entry Point)         │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ commander.js          │
        │ Parse arguments       │
        └───────────────────────┘
                    │
        ┌───────────┴───────────────────────┐
        ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│ apply command   │               │ batch command   │
└─────────────────┘               └─────────────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│ commands/       │               │ commands/       │
│ apply.js        │               │ batch.js        │
└─────────────────┘               └─────────────────┘
        │                                   │
        │                                   │
        ├───── ora.spinner() ──────────────┤
        │      (Progress UI)                │
        │                                   │
        ├───── chalk.color() ──────────────┤
        │      (Colored output)             │
        │                                   │
        └───── agentBooster.apply() ───────┘
               (SDK calls)
                    │
                    ▼
        ┌───────────────────────┐
        │ Output to stdout      │
        │ or file               │
        └───────────────────────┘
```

## Memory Management

```
┌─────────────────────────────────────────────────────────────┐
│  Application Lifecycle                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ First API call        │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Load runtime (5-10MB) │
        │ Cache in memory       │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Subsequent calls      │
        │ Use cached runtime    │
        │ (0ms overhead)        │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Process exits         │
        │ Runtime unloaded      │
        └───────────────────────┘
```

## Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│  User calls API method                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Try initialize()      │
        └───────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
        [Success]        [Error]
            │               │
            │               ▼
            │   ┌──────────────────────────┐
            │   │ Error: No runtime        │
            │   │ Include helpful message: │
            │   │ "npm install             │
            │   │  @agent-booster/wasm"    │
            │   └──────────────────────────┘
            │               │
            ▼               │
  ┌──────────────────────┐ │
  │ Try API call         │ │
  └──────────────────────┘ │
            │               │
    ┌───────┴───────┐       │
    ▼               ▼       │
[Success]        [Error]    │
    │               │       │
    │               ▼       │
    │   ┌──────────────────────────┐
    │   │ Validation error,        │
    │   │ runtime error, etc.      │
    │   └──────────────────────────┘
    │               │
    └───────────────┘
            │
            ▼
  ┌──────────────────────┐
  │ Return result or     │
  │ throw descriptive    │
  │ error                │
  └──────────────────────┘
```

## TypeScript Integration

```
┌─────────────────────────────────────────────────────────────┐
│  TypeScript Application                                      │
│  import * as agentBooster from 'agent-booster';             │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ TypeScript Compiler   │
        │ Reads index.d.ts      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Type Checking         │
        │ • Function signatures │
        │ • Interface validation│
        │ • Option types        │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Compilation           │
        │ TypeScript → JS       │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Runtime (index.js)    │
        │ JavaScript execution  │
        └───────────────────────┘
```

## Performance Optimization

### Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│  Global Module Cache                                         │
├─────────────────────────────────────────────────────────────┤
│  let agentBooster = null;        // Cached runtime          │
│  let loadedRuntime = null;       // Runtime type            │
└─────────────────────────────────────────────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
    ┌──────────────┐  ┌──────────────┐
    │ First call   │  │ Later calls  │
    │ • Detect     │  │ • Use cache  │
    │ • Load       │  │ • 0ms delay  │
    │ • Cache      │  │              │
    │ ~1-5ms       │  │              │
    └──────────────┘  └──────────────┘
```

### Batch Processing Optimization

```
┌─────────────────────────────────────────────────────────────┐
│  Sequential Processing                                       │
│  for each prompt: await apply(prompt)                       │
│  Time: N × T (N prompts, T time per prompt)                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ (Optimize to)
┌─────────────────────────────────────────────────────────────┐
│  Batch Processing                                            │
│  await batchApply(prompts)                                  │
│  Time: T + ε (T time, ε small overhead)                    │
│  Speedup: ~2-5x for typical workloads                       │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NPM Registry                              │
├─────────────────────────────────────────────────────────────┤
│  agent-booster                                              │
│  agent-booster-cli                                          │
│  @agent-booster/wasm                                        │
│  @agent-booster/native-darwin-arm64                         │
│  @agent-booster/native-darwin-x64                           │
│  @agent-booster/native-linux-arm64                          │
│  @agent-booster/native-linux-x64                            │
│  @agent-booster/native-win32-x64                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ (npm install)
┌─────────────────────────────────────────────────────────────┐
│  User's node_modules/                                        │
├─────────────────────────────────────────────────────────────┤
│  agent-booster/                                             │
│  ├─ index.js                                                │
│  ├─ index.d.ts                                              │
│  └─ package.json                                            │
│                                                              │
│  @agent-booster/                                            │
│  ├─ wasm/                                                   │
│  └─ native-{platform}-{arch}/  (optional)                  │
└─────────────────────────────────────────────────────────────┘
```

## Cross-Platform Support

```
┌─────────────────────────────────────────────────────────────┐
│                Platform Detection                            │
└─────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ os.platform()   │     │ os.arch()       │
│ • darwin        │     │ • arm64         │
│ • linux         │     │ • x64           │
│ • win32         │     │                 │
└─────────────────┘     └─────────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │ Combine to package    │
        │ darwin-arm64          │
        │ linux-x64             │
        │ win32-x64             │
        └───────────────────────┘
                    │
        ┌───────────┴───────────────────────┐
        ▼                                   ▼
┌─────────────────┐               ┌─────────────────┐
│ Try load native │               │ Fall back to    │
│ for platform    │──────[Fail]──▶│ WASM (works on  │
│                 │               │ all platforms)  │
└─────────────────┘               └─────────────────┘
```

## Key Design Principles

1. **Zero Configuration**: Works out of the box with no setup
2. **Graceful Degradation**: Native → WASM → Error with help
3. **Performance First**: Cache runtime, batch processing
4. **Developer Experience**: TypeScript, examples, great errors
5. **Cross-Platform**: Universal WASM fallback
6. **Extensibility**: Easy to add new commands/features

## Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│  Package Installation                                        │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ npm install           │
        │ Verifies signatures   │
        │ Checks integrity      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Runtime Loading       │
        │ • No eval()           │
        │ • No dynamic imports  │
        │ • Safe require()      │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ API Calls             │
        │ • Input validation    │
        │ • Type checking       │
        │ • Error boundaries    │
        └───────────────────────┘
```

---

This architecture ensures optimal performance, excellent developer experience, and maximum compatibility across all platforms.
