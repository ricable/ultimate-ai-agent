use agent_booster::merge::{Merger, MergeResult};
use agent_booster::similarity::SearchResult;
use agent_booster::models::{CodeChunk, Language, MergeStrategy};

fn create_search_result(code: &str, similarity: f32, start: usize, end: usize) -> SearchResult {
    SearchResult {
        chunk: CodeChunk {
            code: code.to_string(),
            start_byte: start,
            end_byte: end,
            start_line: 0,
            end_line: 0,
            node_type: "function_declaration".to_string(),
            parent_type: None,
        },
        similarity,
        chunk_index: 0,
    }
}

#[test]
fn test_exact_replace_strategy() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }\nfunction bar() { return 2; }";
    let edit = "function foo() { return 99; }";

    let match_result = create_search_result("function foo() { return 1; }", 0.96, 0, 28);

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("return 99"));
    assert!(result.code.contains("bar"));
    assert_eq!(result.strategy, MergeStrategy::ExactReplace);
    assert!(result.confidence > 0.9);
    assert!(result.syntax_valid);
}

#[test]
fn test_fuzzy_replace_strategy() {
    let mut merger = Merger::new().unwrap();
    let original = "function calculate(x, y) {\n    return x + y;\n}";
    let edit = "function calculate(x, y) {\n    return x * y;\n}";

    let match_result = create_search_result(original, 0.85, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("x * y"));
    assert_eq!(result.strategy, MergeStrategy::FuzzyReplace);
    assert!(result.confidence > 0.7);
}

#[test]
fn test_insert_after_strategy() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }";
    let edit = "function bar() { return 2; }";

    let match_result = create_search_result(original, 0.65, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("foo"));
    assert!(result.code.contains("bar"));
    assert_eq!(result.strategy, MergeStrategy::InsertAfter);
    assert!(result.code.lines().count() > 1); // Should have newlines
}

#[test]
fn test_insert_before_strategy() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }";
    let edit = "function bar() { return 2; }";

    let match_result = create_search_result(original, 0.55, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("foo"));
    assert!(result.code.contains("bar"));
    assert_eq!(result.strategy, MergeStrategy::InsertBefore);
    // Bar should come before foo in the result
    let bar_pos = result.code.find("bar").unwrap();
    let foo_pos = result.code.find("foo").unwrap();
    assert!(bar_pos < foo_pos);
}

#[test]
fn test_append_strategy() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }";
    let edit = "function bar() { return 2; }";

    let match_result = create_search_result(original, 0.30, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.1).unwrap();

    assert!(result.code.contains("foo"));
    assert!(result.code.contains("bar"));
    assert_eq!(result.strategy, MergeStrategy::Append);
    // Bar should be at the end
    assert!(result.code.ends_with("function bar() { return 2; }"));
}

#[test]
fn test_low_confidence_error() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }";
    let edit = "completely different code that won't match";

    let match_result = create_search_result(original, 0.20, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.8);

    assert!(result.is_err());
}

#[test]
fn test_invalid_syntax_detection() {
    let mut merger = Merger::new().unwrap();
    let original = "function foo() { return 1; }";
    let edit = "function bar() { return 2;"; // Missing closing brace

    let match_result = create_search_result(original, 0.96, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5);

    assert!(result.is_err());
}

#[test]
fn test_merge_preserves_surrounding_code() {
    let mut merger = Merger::new().unwrap();
    let original = r#"
// Header comment
function foo() { return 1; }

// Middle comment
function bar() { return 2; }

// Footer comment
"#;
    let edit = "function foo() { return 99; }";

    let match_result = create_search_result("function foo() { return 1; }", 0.96, 18, 46);

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("Header comment"));
    assert!(result.code.contains("Middle comment"));
    assert!(result.code.contains("Footer comment"));
    assert!(result.code.contains("return 99"));
    assert!(result.code.contains("bar"));
}

#[test]
fn test_merge_class_method() {
    let mut merger = Merger::new().unwrap();
    let original = r#"
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
"#;
    let edit = r#"
    multiply(a, b) {
        return a * b;
    }
"#;

    // Insert after the subtract method
    let subtract_start = original.find("subtract").unwrap() - 4;
    let subtract_end = original.find("    }\n}").unwrap() + 5;
    let match_result = create_search_result(
        &original[subtract_start..subtract_end],
        0.70,
        subtract_start,
        subtract_end,
    );

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("add"));
    assert!(result.code.contains("subtract"));
    assert!(result.code.contains("multiply"));
    assert_eq!(result.strategy, MergeStrategy::InsertAfter);
}

#[test]
fn test_merge_typescript_interface() {
    let mut merger = Merger::new().unwrap();
    let original = r#"
interface User {
    id: number;
    name: string;
}
"#;
    let edit = r#"
interface Product {
    id: number;
    price: number;
}
"#;

    let match_result = create_search_result(original.trim(), 0.65, 1, original.trim().len() + 1);

    let result = merger.merge(original, edit, &match_result, Language::TypeScript, 0.5).unwrap();

    assert!(result.code.contains("User"));
    assert!(result.code.contains("Product"));
    assert!(result.syntax_valid);
}

#[test]
fn test_confidence_calculation() {
    let mut merger = Merger::new().unwrap();

    // High confidence: exact replace with valid syntax
    let high_match = create_search_result("code", 0.96, 0, 4);
    let high_result = merger.merge("code", "code", &high_match, Language::JavaScript, 0.3).unwrap();
    assert!(high_result.confidence > 0.9);

    // Medium confidence: insert with valid syntax
    let medium_match = create_search_result("code", 0.65, 0, 4);
    let medium_result = merger.merge("code", "more", &medium_match, Language::JavaScript, 0.3).unwrap();
    assert!(medium_result.confidence > 0.5 && medium_result.confidence < 0.9);

    // Lower confidence: append
    let low_match = create_search_result("code", 0.30, 0, 4);
    let low_result = merger.merge("code", "more", &low_match, Language::JavaScript, 0.1).unwrap();
    assert!(low_result.confidence < 0.5);
}

#[test]
fn test_merge_empty_file() {
    let mut merger = Merger::new().unwrap();
    let original = "";
    let edit = "function test() { return 1; }";

    let match_result = create_search_result("", 0.30, 0, 0);

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.1).unwrap();

    assert_eq!(result.strategy, MergeStrategy::Append);
    assert!(result.code.contains("function test"));
}

#[test]
fn test_merge_with_unicode() {
    let mut merger = Merger::new().unwrap();
    let original = "function greet() { return '你好'; }";
    let edit = "function greet() { return 'Hello'; }";

    let match_result = create_search_result(original, 0.96, 0, original.len());

    let result = merger.merge(original, edit, &match_result, Language::JavaScript, 0.5).unwrap();

    assert!(result.code.contains("Hello"));
    assert!(result.syntax_valid);
}

#[test]
fn test_multiple_sequential_merges() {
    let mut merger = Merger::new().unwrap();
    let mut code = "function foo() { return 1; }";

    // First merge: add bar
    let edit1 = "function bar() { return 2; }";
    let match1 = create_search_result(code, 0.30, 0, code.len());
    let result1 = merger.merge(code, edit1, &match1, Language::JavaScript, 0.1).unwrap();
    code = &result1.code;

    // Second merge: add baz
    let edit2 = "function baz() { return 3; }";
    let match2 = create_search_result(code, 0.30, 0, code.len());
    let result2 = merger.merge(code, edit2, &match2, Language::JavaScript, 0.1).unwrap();

    assert!(result2.code.contains("foo"));
    assert!(result2.code.contains("bar"));
    assert!(result2.code.contains("baz"));
}
