use agent_booster::similarity::SimilarityMatcher;
use agent_booster::models::CodeChunk;

fn create_chunk(code: &str, node_type: &str) -> CodeChunk {
    CodeChunk {
        code: code.to_string(),
        start_byte: 0,
        end_byte: code.len(),
        start_line: 0,
        end_line: code.lines().count().saturating_sub(1),
        node_type: node_type.to_string(),
        parent_type: None,
    }
}

#[test]
fn test_exact_match_identical_code() {
    let code1 = "function test() { return 42; }";
    let code2 = "function test() { return 42; }";

    assert!(SimilarityMatcher::is_exact_match(code1, code2));
}

#[test]
fn test_exact_match_with_whitespace_differences() {
    let code1 = "function test() { return 42; }";
    let code2 = "function  test()  {  return  42;  }";

    assert!(SimilarityMatcher::is_exact_match(code1, code2));
}

#[test]
fn test_exact_match_with_newlines() {
    let code1 = "function test() {\n    return 42;\n}";
    let code2 = "function test() { return 42; }";

    assert!(SimilarityMatcher::is_exact_match(code1, code2));
}

#[test]
fn test_no_match_different_code() {
    let code1 = "function foo() { return 1; }";
    let code2 = "function bar() { return 2; }";

    assert!(!SimilarityMatcher::is_exact_match(code1, code2));
}

#[test]
fn test_find_best_match_obvious_match() {
    let chunks = vec![
        create_chunk("function foo() { return 1; }", "function_declaration"),
        create_chunk("function bar() { return 2; }", "function_declaration"),
        create_chunk("function baz() { return 3; }", "function_declaration"),
    ];

    let edit = "function bar() { return 99; }";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    assert!(result.chunk.code.contains("bar"));
    assert!(result.similarity > 0.7);
    assert_eq!(result.chunk_index, 1);
}

#[test]
fn test_find_best_match_similar_structure() {
    let chunks = vec![
        create_chunk("const x = 1;", "variable_declaration"),
        create_chunk("function compute(a, b) { return a + b; }", "function_declaration"),
        create_chunk("class Helper {}", "class_declaration"),
    ];

    let edit = "function calculate(x, y) { return x * y; }";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Should match the function, not the variable or class
    assert!(result.chunk.code.contains("compute"));
    assert!(result.similarity > 0.5);
}

#[test]
fn test_find_best_match_empty_chunks() {
    let chunks = vec![];
    let edit = "function test() {}";
    let result = SimilarityMatcher::find_best_match(edit, &chunks);

    assert!(result.is_none());
}

#[test]
fn test_find_top_k_matches() {
    let chunks = vec![
        create_chunk("function foo() { return 1; }", "function_declaration"),
        create_chunk("function bar() { return 2; }", "function_declaration"),
        create_chunk("function baz() { return 3; }", "function_declaration"),
        create_chunk("const x = 42;", "variable_declaration"),
        create_chunk("class Test {}", "class_declaration"),
    ];

    let edit = "function test() { return 99; }";
    let results = SimilarityMatcher::find_top_k_matches(edit, &chunks, 3);

    assert_eq!(results.len(), 3);
    // All top 3 should be functions
    assert!(results[0].chunk.code.contains("function"));
    // Results should be sorted by similarity (descending)
    assert!(results[0].similarity >= results[1].similarity);
    assert!(results[1].similarity >= results[2].similarity);
}

#[test]
fn test_find_top_k_more_than_available() {
    let chunks = vec![
        create_chunk("function foo() {}", "function_declaration"),
        create_chunk("function bar() {}", "function_declaration"),
    ];

    let edit = "function test() {}";
    let results = SimilarityMatcher::find_top_k_matches(edit, &chunks, 10);

    // Should return all available chunks (2), not 10
    assert_eq!(results.len(), 2);
}

#[test]
fn test_similarity_with_comments() {
    let chunks = vec![
        create_chunk(
            r#"
            // This is a comment
            function test() {
                // Another comment
                return 42;
            }
            "#,
            "function_declaration"
        ),
        create_chunk("function test() { return 42; }", "function_declaration"),
    ];

    let edit = "function test() { return 42; }";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Both should match well, comments should be normalized away
    assert!(result.similarity > 0.8);
}

#[test]
fn test_structural_similarity_matching_braces() {
    let chunks = vec![
        create_chunk("{ a: 1, b: 2 }", "object"),
        create_chunk("function test() { return 1; }", "function_declaration"),
        create_chunk("[1, 2, 3]", "array"),
    ];

    let edit = "function other() { return 2; }";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Should prefer the function (matching structure) over object/array
    assert!(result.chunk.code.contains("function"));
}

#[test]
fn test_token_similarity_keyword_matching() {
    let chunks = vec![
        create_chunk("const x = 1;", "variable_declaration"),
        create_chunk("function process() { return compute(); }", "function_declaration"),
        create_chunk("class Handler {}", "class_declaration"),
    ];

    let edit = "class Processor {}";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Should match the class due to 'class' keyword
    assert!(result.chunk.code.contains("class"));
}

#[test]
fn test_similarity_with_different_indentation() {
    let chunk1 = create_chunk(
        r#"function test() {
return 42;
}"#,
        "function_declaration"
    );

    let chunk2 = create_chunk(
        r#"function test() {
    return 42;
}"#,
        "function_declaration"
    );

    let chunks = vec![chunk1, chunk2];
    let edit = "function test() { return 42; }";

    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Both should match well despite indentation differences
    assert!(result.similarity > 0.9);
}

#[test]
fn test_find_match_with_partial_content() {
    let chunks = vec![
        create_chunk(
            r#"function calculate(a, b) {
    const sum = a + b;
    const product = a * b;
    return { sum, product };
}"#,
            "function_declaration"
        ),
    ];

    let edit = r#"function calculate(a, b) {
    return a + b;
}"#;

    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Should still match despite different implementation
    assert!(result.similarity > 0.6);
}

#[test]
fn test_similarity_case_sensitive() {
    let chunks = vec![
        create_chunk("function TEST() {}", "function_declaration"),
        create_chunk("function test() {}", "function_declaration"),
    ];

    let edit = "function test() {}";
    let result = SimilarityMatcher::find_best_match(edit, &chunks).unwrap();

    // Should prefer exact case match
    assert!(result.chunk.code.contains("function test()"));
}
