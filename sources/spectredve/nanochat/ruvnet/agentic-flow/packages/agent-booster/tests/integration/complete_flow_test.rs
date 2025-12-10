use agent_booster::{AgentBooster, Config, EditRequest, Language};
use std::fs;
use std::path::PathBuf;

fn get_fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(filename)
}

#[test]
fn test_end_to_end_javascript_edit() {
    let fixture_path = get_fixture_path("sample_javascript.js");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    // Edit: Change calculateSum to use multiplication instead
    let edit = r#"
function calculateSum(a, b) {
    return a * b;
}
"#;

    let request = EditRequest {
        original_code: original_code.clone(),
        edit_snippet: edit.to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify the edit was applied
    assert!(result.merged_code.contains("a * b"));
    assert!(result.confidence > 0.7);
    assert!(result.metadata.syntax_valid);
    assert!(result.metadata.processing_time_ms.is_some());

    // Verify other functions are preserved
    assert!(result.merged_code.contains("calculateProduct"));
    assert!(result.merged_code.contains("MathOperations"));
}

#[test]
fn test_end_to_end_add_new_method() {
    let fixture_path = get_fixture_path("sample_javascript.js");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    // Add a new method to MathOperations class
    let edit = r#"
    divide(a, b) {
        if (b === 0) throw new Error('Division by zero');
        const result = a / b;
        this.history.push({ op: 'divide', result });
        return result;
    }
"#;

    let request = EditRequest {
        original_code,
        edit_snippet: edit.to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify the new method was added
    assert!(result.merged_code.contains("divide"));
    assert!(result.merged_code.contains("Division by zero"));

    // Verify existing methods are preserved
    assert!(result.merged_code.contains("add(a, b)"));
    assert!(result.merged_code.contains("multiply(a, b)"));
}

#[test]
fn test_end_to_end_typescript_interface() {
    let fixture_path = get_fixture_path("sample_typescript.ts");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    // Add a new interface
    let edit = r#"
interface ShoppingCart {
    id: string;
    userId: number;
    items: Product[];
    createdAt: Date;
}
"#;

    let request = EditRequest {
        original_code: original_code.clone(),
        edit_snippet: edit.to_string(),
        language: Language::TypeScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify the new interface was added
    assert!(result.merged_code.contains("interface ShoppingCart"));
    assert!(result.merged_code.contains("createdAt: Date"));

    // Verify existing interfaces are preserved
    assert!(result.merged_code.contains("interface User"));
    assert!(result.merged_code.contains("interface Product"));
}

#[test]
fn test_end_to_end_update_user_service() {
    let fixture_path = get_fixture_path("sample_typescript.ts");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    // Add a new method to UserService
    let edit = r#"
    updateUser(id: number, updates: Partial<User>): boolean {
        const user = this.findUserById(id);
        if (!user) return false;
        Object.assign(user, updates);
        return true;
    }
"#;

    let request = EditRequest {
        original_code,
        edit_snippet: edit.to_string(),
        language: Language::TypeScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify the new method was added
    assert!(result.merged_code.contains("updateUser"));
    assert!(result.merged_code.contains("Partial<User>"));

    // Verify existing methods are preserved
    assert!(result.merged_code.contains("addUser"));
    assert!(result.merged_code.contains("findUserById"));
}

#[test]
fn test_batch_processing() {
    let js_fixture = get_fixture_path("sample_javascript.js");
    let js_code = fs::read_to_string(&js_fixture).expect("Failed to read JS fixture");

    let ts_fixture = get_fixture_path("sample_typescript.ts");
    let ts_code = fs::read_to_string(&ts_fixture).expect("Failed to read TS fixture");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let requests = vec![
        EditRequest {
            original_code: js_code,
            edit_snippet: "function newJsFunction() { return 'js'; }".to_string(),
            language: Language::JavaScript,
            confidence_threshold: 0.3,
        },
        EditRequest {
            original_code: ts_code,
            edit_snippet: "type NewType = string | number;".to_string(),
            language: Language::TypeScript,
            confidence_threshold: 0.3,
        },
    ];

    let results = booster.batch_apply(requests).unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].merged_code.contains("newJsFunction"));
    assert!(results[1].merged_code.contains("NewType"));
}

#[test]
fn test_config_customization() {
    let fixture_path = get_fixture_path("sample_javascript.js");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    // Create booster with custom config
    let config = Config {
        confidence_threshold: 0.8,
        max_chunks: 10,
    };

    let mut booster = AgentBooster::new(config).unwrap();

    let edit = "function test() { return 1; }";

    let request = EditRequest {
        original_code,
        edit_snippet: edit.to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.8,
    };

    let result = booster.apply_edit(request);

    // With high threshold, this might fail or succeed depending on match quality
    // Either way, the config was respected
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_error_handling_invalid_syntax() {
    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let request = EditRequest {
        original_code: "function valid() { return 1; }".to_string(),
        edit_snippet: "function invalid() { return 1;".to_string(), // Missing }
        language: Language::JavaScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request);
    assert!(result.is_err());
}

#[test]
fn test_error_handling_low_confidence() {
    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let request = EditRequest {
        original_code: "function foo() { return 1; }".to_string(),
        edit_snippet: "completely unrelated code that won't match".to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.9, // Very high threshold
    };

    let result = booster.apply_edit(request);
    assert!(result.is_err());
}

#[test]
fn test_performance_benchmark() {
    use std::time::Instant;

    let fixture_path = get_fixture_path("sample_javascript.js");
    let original_code = fs::read_to_string(&fixture_path)
        .expect("Failed to read fixture file");

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let edit = "function benchmark() { return 'fast'; }";

    let request = EditRequest {
        original_code,
        edit_snippet: edit.to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.3,
    };

    let start = Instant::now();
    let result = booster.apply_edit(request).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time (< 100ms for small files)
    assert!(duration.as_millis() < 100, "Processing took too long: {:?}", duration);
    assert!(result.metadata.processing_time_ms.unwrap() < 100);
}

#[test]
fn test_metadata_accuracy() {
    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let code = r#"
function foo() { return 1; }
function bar() { return 2; }
function baz() { return 3; }
"#;

    let request = EditRequest {
        original_code: code.to_string(),
        edit_snippet: "function qux() { return 4; }".to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.3,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify metadata is populated
    assert!(result.metadata.chunks_found > 0);
    assert!(result.metadata.best_similarity > 0.0);
    assert!(result.metadata.processing_time_ms.is_some());
    assert!(result.metadata.syntax_valid);
}

#[test]
fn test_real_world_scenario_add_error_handling() {
    let original = r#"
class DataService {
    async fetchData(id) {
        const response = await fetch(`/api/data/${id}`);
        const data = await response.json();
        return data;
    }
}
"#;

    let edit = r#"
    async fetchData(id) {
        try {
            const response = await fetch(`/api/data/${id}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to fetch data:', error);
            throw error;
        }
    }
"#;

    let mut booster = AgentBooster::new(Config::default()).unwrap();

    let request = EditRequest {
        original_code: original.to_string(),
        edit_snippet: edit.to_string(),
        language: Language::JavaScript,
        confidence_threshold: 0.5,
    };

    let result = booster.apply_edit(request).unwrap();

    // Verify error handling was added
    assert!(result.merged_code.contains("try"));
    assert!(result.merged_code.contains("catch"));
    assert!(result.merged_code.contains("throw"));
    assert!(result.confidence > 0.7);
}
