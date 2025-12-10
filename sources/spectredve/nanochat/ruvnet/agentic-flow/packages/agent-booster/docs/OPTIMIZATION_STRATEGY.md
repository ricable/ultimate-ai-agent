# Agent Booster Optimization Strategy

## Current Performance Analysis

### Win Rate: 50% (6/12 tests)

**Wins (Simple Replacements):**
- ✅ test-001: Add type annotations (replacement)
- ✅ test-003: Convert var to const/let (replacement)
- ✅ test-004: Add JSDoc comments (insertion)
- ✅ test-005: Convert callback to Promise (replacement with insertion)
- ✅ test-007: Convert to arrow function (replacement)
- ✅ test-010: Add destructuring (replacement)

**Losses (Complex Transformations):**
- ❌ test-002: Add error handling + types (wrapping existing code in try-catch)
- ❌ test-006: Add null checks (adding conditional logic)
- ❌ test-008: Add input validation (adding validation logic at start)
- ❌ test-009: Convert class to TypeScript (adding types + new method)
- ❌ test-011: Add try-catch wrapper (wrapping existing code)
- ❌ test-012: Add async/await + error handling (transformation + wrapping)

---

## Root Cause Analysis

### Why Agent Booster Fails

**Pattern:** All failures involve **wrapping** or **inserting logic** into existing functions, not just replacing syntax.

1. **test-002: Add error handling**
   - Input: Simple async function
   - Expected: Wrap entire body in try-catch
   - Challenge: Parser doesn't understand "wrap function body"

2. **test-006: Add null checks**
   - Input: Direct property access
   - Expected: Add if statement before return
   - Challenge: Parser doesn't match the simple input with complex output

3. **test-008: Add input validation**
   - Input: Simple division function
   - Expected: Add type checks and error throwing
   - Challenge: No matching code to replace

4. **test-009: Convert class to TypeScript**
   - Input: Plain JavaScript class
   - Expected: Add types + new method
   - Challenge: Adding new method (getResult) not in original

5. **test-011: Add try-catch wrapper**
   - Input: Direct JSON.parse call
   - Expected: Wrap in try-catch
   - Challenge: Similar to test-002

6. **test-012: Add async/await + error handling**
   - Input: Promise chain
   - Expected: Convert to async/await + wrap in try-catch
   - Challenge: Complex transformation + wrapping

---

## Optimization Strategies

### Strategy 1: Template-Based Code Generation ⭐⭐⭐⭐⭐

**Concept:** Pre-define templates for common transformations

```rust
pub struct TransformationTemplate {
    pattern: String,
    template: String,
    placeholders: Vec<String>,
}

// Example: Try-catch wrapper
TransformationTemplate {
    pattern: r"function (\w+)\((.*?)\) \{([\s\S]+)\}",
    template: "function $1($2) {\n  try {\n    $3\n  } catch (error) {\n    console.error('Error:', error);\n    throw error;\n  }\n}",
    placeholders: vec!["$1", "$2", "$3"],
}
```

**Implementation:**
1. Add `templates.rs` with common patterns
2. Match input against template patterns
3. Extract placeholders
4. Fill template with extracted values
5. Return generated code

**Win Rate Impact:** +30% (would fix 3-4 failures)

**Estimated Effort:** 2-3 hours

---

### Strategy 2: Semantic Understanding via AST Transformation ⭐⭐⭐⭐

**Concept:** Use tree-sitter to understand code structure and apply transformations

```rust
pub struct SemanticTransform {
    rule: TransformRule,
    matcher: Box<dyn Fn(&Node) -> bool>,
    transformer: Box<dyn Fn(&Node, &str) -> String>,
}

// Example: Wrap function body in try-catch
SemanticTransform {
    rule: TransformRule::WrapFunctionBody,
    matcher: |node| node.kind() == "function_declaration",
    transformer: |node, code| {
        let body = extract_function_body(node, code);
        format!("try {{\n  {}\n}} catch (error) {{\n  throw error;\n}}", body)
    },
}
```

**Implementation:**
1. Parse both input and edit with tree-sitter
2. Detect transformation type (wrap, insert, replace)
3. Apply semantic transformation
4. Generate output code

**Win Rate Impact:** +40% (would fix 4-5 failures)

**Estimated Effort:** 4-6 hours

---

### Strategy 3: Multi-Pass Strategy Selection ⭐⭐⭐⭐⭐

**Concept:** Try multiple strategies in order of likelihood

```rust
pub struct MultiPassMerger {
    strategies: Vec<Box<dyn MergeStrategy>>,
}

impl MultiPassMerger {
    pub fn apply(&self, request: EditRequest) -> Result<EditResult> {
        // Pass 1: Template matching
        if let Ok(result) = self.try_template_match(request) {
            if result.confidence > 0.8 {
                return Ok(result);
            }
        }

        // Pass 2: Semantic transformation
        if let Ok(result) = self.try_semantic_transform(request) {
            if result.confidence > 0.6 {
                return Ok(result);
            }
        }

        // Pass 3: Current fuzzy matching
        self.try_fuzzy_match(request)
    }
}
```

**Win Rate Impact:** +50% (would fix 6 failures)

**Estimated Effort:** 3-4 hours

---

### Strategy 4: Context-Aware Insertion ⭐⭐⭐

**Concept:** Understand WHERE to insert code based on intent

```rust
pub enum InsertionPoint {
    FunctionStart,    // After function signature
    FunctionEnd,      // Before return
    BeforeStatement,  // Before specific statement
    AfterStatement,   // After specific statement
    WrapStatement,    // Wrap existing statement
}

pub fn detect_insertion_point(input: &str, edit: &str) -> InsertionPoint {
    if edit.contains("try {") && input.contains("function") {
        InsertionPoint::WrapStatement
    } else if edit.contains("if (") && edit.contains("return null") {
        InsertionPoint::FunctionStart
    } else {
        // Default to existing logic
        InsertionPoint::AfterStatement
    }
}
```

**Win Rate Impact:** +25% (would fix 3 failures)

**Estimated Effort:** 2 hours

---

### Strategy 5: Intent Detection via Edit Analysis ⭐⭐⭐⭐

**Concept:** Analyze the diff between input and edit to understand intent

```rust
pub enum TransformIntent {
    AddTypeAnnotations,
    AddErrorHandling,
    AddNullChecks,
    AddValidation,
    WrapInTryCatch,
    ConvertToAsync,
    AddDocumentation,
}

pub fn detect_intent(input: &str, edit: &str) -> TransformIntent {
    let diff = compute_diff(input, edit);

    if diff.contains("try {") && diff.contains("catch") {
        TransformIntent::WrapInTryCatch
    } else if diff.contains("if (!") && diff.contains("return null") {
        TransformIntent::AddNullChecks
    } else if diff.contains("typeof") && diff.contains("throw") {
        TransformIntent::AddValidation
    } else if diff.contains(": number") || diff.contains(": string") {
        TransformIntent::AddTypeAnnotations
    } else {
        // Default to replacement
        TransformIntent::AddTypeAnnotations
    }
}
```

**Win Rate Impact:** +35% (would fix 4 failures)

**Estimated Effort:** 3 hours

---

## Language Support Improvements

### Current Limitations

1. **Python:** 88% success (lower than brace-based languages)
2. **Complex Syntax:** Generics, templates, macros not well-supported
3. **Language-Specific Features:** Decorators, attributes, annotations

### Improvement Strategies

#### 1. Enhanced Python Parser ⭐⭐⭐⭐⭐

**Current Issues:**
- Indentation-based block detection is simplistic
- Doesn't handle nested functions well
- Decorator support is basic

**Solution:**
```rust
pub struct EnhancedPythonParser {
    // Track indentation levels
    indentation_stack: Vec<usize>,

    // Parse decorators
    decorator_regex: Regex,

    // Handle nested structures
    nesting_depth: usize,
}

impl EnhancedPythonParser {
    pub fn extract_function_with_decorators(&self, code: &str) -> Vec<CodeChunk> {
        // 1. Find decorators (@staticmethod, @property, etc.)
        // 2. Associate decorators with functions
        // 3. Extract complete function including decorators
        // 4. Handle nested functions
    }

    pub fn track_indentation(&mut self, line: &str) -> usize {
        // Smart indentation tracking
        // Handle mixed tabs/spaces
        // Detect indentation errors
    }
}
```

**Win Rate Impact:** +10% for Python (88% → 98%)

**Estimated Effort:** 2-3 hours

---

#### 2. Advanced Rust Pattern Matching ⭐⭐⭐⭐

**Add support for:**
- Generic functions: `fn foo<T>(x: T) -> T`
- Trait implementations: `impl<T> Trait for Type<T>`
- Macro invocations: `macro_rules!`
- Lifetime annotations: `fn foo<'a>(x: &'a str)`

```rust
// Enhanced Rust regex patterns
rust_generic_function: r"(?m)^\s*(?:pub\s+)?fn\s+(\w+)<([^>]+)>\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?\s*\{",
rust_trait_impl: r"(?m)^\s*impl<([^>]+)>\s+(\w+)\s+for\s+(\w+)<([^>]+)>\s*\{",
rust_lifetime: r"(?m)^\s*(?:pub\s+)?fn\s+(\w+)<'(\w+)>",
```

**Win Rate Impact:** Rust 100% → 100% (maintain excellence, support advanced features)

**Estimated Effort:** 2 hours

---

#### 3. Java Generics and Annotations ⭐⭐⭐

**Add support for:**
- Generic methods: `<T> T foo(T x)`
- Annotations: `@Override`, `@SuppressWarnings`
- Lambda expressions: `(x) -> x * 2`

```rust
java_generic_method: r"(?m)^\s*(?:public|private)?\s*<([^>]+)>\s*(\w+)\s+(\w+)\s*\(([^)]*)\)",
java_annotation: r"(?m)^\s*@(\w+)(?:\([^)]*\))?\s*$",
java_lambda: r"\([^)]*\)\s*->\s*",
```

**Win Rate Impact:** Java 100% → 100% (maintain + advanced features)

**Estimated Effort:** 1.5 hours

---

#### 4. C++ Template Support ⭐⭐⭐⭐

**Add support for:**
- Template functions: `template<typename T> T foo(T x)`
- Template classes: `template<class T> class Stack`
- Template specialization: `template<> class Stack<int>`

```rust
cpp_template_function: r"(?m)^\s*template\s*<([^>]+)>\s*(\w+)\s+(\w+)\s*\(",
cpp_template_class: r"(?m)^\s*template\s*<([^>]+)>\s*class\s+(\w+)",
cpp_template_specialization: r"(?m)^\s*template\s*<>\s*class\s+(\w+)<([^>]+)>",
```

**Win Rate Impact:** C++ 100% → 100% (maintain + templates)

**Estimated Effort:** 2 hours

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (4-6 hours) ⭐⭐⭐⭐⭐

**Target:** Improve win rate from 50% to 75%

1. **Template-Based Code Generation** (2-3 hours)
   - Add 10 common transformation templates
   - Try-catch wrapper
   - Null check insertion
   - Input validation
   - Type annotation addition

2. **Context-Aware Insertion** (2 hours)
   - Detect insertion points
   - Function start/end
   - Before/after statements

3. **Multi-Pass Strategy** (1 hour)
   - Implement strategy cascade
   - Template → Semantic → Fuzzy

**Expected Results:**
- Win rate: 50% → 75% (+25%)
- Would fix: test-002, test-011, test-006

---

### Phase 2: Advanced Transformations (6-8 hours) ⭐⭐⭐⭐

**Target:** Improve win rate from 75% to 90%

1. **Semantic Understanding** (4-6 hours)
   - AST-based transformation
   - Wrap function body
   - Insert at semantic locations

2. **Intent Detection** (2 hours)
   - Analyze edit patterns
   - Classify transformation types
   - Apply intent-specific logic

**Expected Results:**
- Win rate: 75% → 90% (+15%)
- Would fix: test-008, test-009, test-012

---

### Phase 3: Language Excellence (6-8 hours) ⭐⭐⭐

**Target:** Improve language support from 91% to 98%

1. **Enhanced Python Parser** (2-3 hours)
   - Better indentation tracking
   - Decorator support
   - Nested function handling

2. **Advanced Rust Patterns** (2 hours)
   - Generics
   - Lifetimes
   - Trait implementations

3. **Java/C++ Advanced Features** (2-3 hours)
   - Generics and templates
   - Annotations
   - Lambda expressions

**Expected Results:**
- Python: 88% → 98% (+10%)
- Overall language support: 91% → 98% (+7%)

---

## Performance Impact Analysis

### Current Performance

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| **Win Rate** | 50% | 75% | 90% | 90% |
| **Language Success** | 91% | 91% | 91% | 98% |
| **Avg Latency** | 1ms | 2ms | 3ms | 3ms |
| **Confidence** | 59.6% | 70% | 80% | 80% |

**Trade-offs:**
- +25-40% win rate
- +1-2ms latency (still 100x+ faster than Morph)
- +10-20% confidence scores

---

## Cost-Benefit Analysis

### Phase 1: Quick Wins

**Investment:** 4-6 hours
**Return:**
- +25% win rate (50% → 75%)
- Fix 3 major failures
- Still 175x faster than Morph (2ms vs 352ms)

**ROI:** ⭐⭐⭐⭐⭐ (Excellent)

### Phase 2: Advanced Transformations

**Investment:** 6-8 hours
**Return:**
- +15% win rate (75% → 90%)
- Fix 3 more failures
- Competitive with Morph LLM accuracy

**ROI:** ⭐⭐⭐⭐ (Very Good)

### Phase 3: Language Excellence

**Investment:** 6-8 hours
**Return:**
- +7% language support (91% → 98%)
- Best-in-class multi-language support
- Competitive advantage

**ROI:** ⭐⭐⭐ (Good)

---

## Recommended Next Steps

### Immediate (This Session)

1. ✅ Implement Template-Based Code Generation (2 hours)
   - Add try-catch wrapper template
   - Add null check template
   - Add validation template

2. ✅ Test with failed cases
   - Validate template matching works
   - Measure win rate improvement

### Short Term (Next Session)

1. Complete Phase 1 (Quick Wins)
   - Context-aware insertion
   - Multi-pass strategy
   - Target: 75% win rate

2. Enhanced Python Parser
   - Better indentation
   - Decorator support
   - Target: 98% Python success

### Medium Term (Future)

1. Complete Phase 2 (Advanced)
   - Semantic understanding
   - Intent detection
   - Target: 90% win rate

2. Complete Phase 3 (Excellence)
   - Advanced language features
   - Target: 98% language support

---

## Success Metrics

### Target Performance (After All Phases)

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Win Rate** | 50% | **90%** | +40% |
| **Language Success** | 91% | **98%** | +7% |
| **Avg Latency** | 1ms | **3ms** | Still 117x faster |
| **Avg Confidence** | 59.6% | **80%** | +20.4% |
| **Languages** | 8 | **10+** | +2+ |

**Result:** Agent Booster would be competitive with Morph LLM in accuracy while maintaining 100x+ speed advantage.

---

## Conclusion

**Highest Priority:** Phase 1 (Quick Wins)
- 4-6 hour investment
- +25% win rate
- Immediate impact

**Best ROI:** Template-Based Code Generation
- 2-3 hour investment
- Fixes 3-4 failures
- Simple to implement

**Recommended Approach:**
1. Start with template-based generation
2. Test against failed cases
3. Iterate based on results
4. Continue with Phase 2 if needed

**Expected Outcome:**
Agent Booster with 75-90% win rate, maintaining 100x+ speed advantage over Morph LLM.
