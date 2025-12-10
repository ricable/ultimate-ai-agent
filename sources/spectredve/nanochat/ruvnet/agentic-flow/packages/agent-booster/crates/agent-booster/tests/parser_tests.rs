use agent_booster::parser::Parser;
use agent_booster::models::Language;

#[test]
fn test_parse_complex_javascript() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    function complexFunction(x, y) {
        const result = x + y;
        if (result > 10) {
            return result * 2;
        }
        return result;
    }

    class ComplexClass {
        constructor(value) {
            this.value = value;
        }

        method1() {
            return this.value * 2;
        }

        method2(x) {
            return this.value + x;
        }
    }
    "#;

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    assert!(!tree.root_node().has_error());

    let chunks = parser.extract_chunks(&tree, code);
    assert!(chunks.len() >= 2, "Should extract function and class");
}

#[test]
fn test_parse_typescript_with_types() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    interface Config {
        apiKey: string;
        timeout: number;
    }

    type Status = 'active' | 'inactive';

    class Service<T> {
        private data: T[] = [];

        add(item: T): void {
            this.data.push(item);
        }

        get(index: number): T | undefined {
            return this.data[index];
        }
    }

    function processData<T>(items: T[]): T[] {
        return items.filter(item => item !== null);
    }
    "#;

    let tree = parser.parse(code, Language::TypeScript).unwrap();
    assert!(!tree.root_node().has_error());

    let chunks = parser.extract_chunks(&tree, code);
    assert!(chunks.len() >= 4, "Should extract interface, type, class, and function");
}

#[test]
fn test_extract_chunks_with_nested_functions() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    function outer() {
        function inner() {
            return 42;
        }
        return inner();
    }
    "#;

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    let chunks = parser.extract_chunks(&tree, code);

    // Should extract both outer and inner functions
    assert!(chunks.len() >= 2, "Should extract nested functions");
}

#[test]
fn test_extract_chunks_from_module() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    import { helper } from './utils';

    export function publicFunction() {
        return 'public';
    }

    const privateFunction = () => {
        return 'private';
    };

    export default class MyClass {
        method() {
            return 'method';
        }
    }
    "#;

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    let chunks = parser.extract_chunks(&tree, code);

    assert!(chunks.len() >= 3, "Should extract imports, exports, and functions");
}

#[test]
fn test_validate_syntax_valid_code() {
    let mut parser = Parser::new().unwrap();
    let valid_code = "function test() { return 42; }";
    assert!(parser.validate_syntax(valid_code, Language::JavaScript));
}

#[test]
fn test_validate_syntax_invalid_code() {
    let mut parser = Parser::new().unwrap();
    let invalid_code = "function test() { return 42;"; // Missing closing brace
    assert!(!parser.validate_syntax(invalid_code, Language::JavaScript));
}

#[test]
fn test_validate_syntax_typescript() {
    let mut parser = Parser::new().unwrap();
    let valid_ts = "function test(): number { return 42; }";
    assert!(parser.validate_syntax(valid_ts, Language::TypeScript));

    let invalid_ts = "function test(): number { return 'string'; "; // Missing brace
    assert!(!parser.validate_syntax(invalid_ts, Language::TypeScript));
}

#[test]
fn test_extract_full_file_small_file() {
    let parser = Parser::new().unwrap();
    let small_code = "const x = 42;\nconst y = 24;";

    let chunk = parser.extract_full_file(small_code);

    assert_eq!(chunk.start_byte, 0);
    assert_eq!(chunk.end_byte, small_code.len());
    assert_eq!(chunk.node_type, "file");
    assert_eq!(chunk.code, small_code);
}

#[test]
fn test_extract_full_file_empty() {
    let parser = Parser::new().unwrap();
    let empty_code = "";

    let chunk = parser.extract_full_file(empty_code);

    assert_eq!(chunk.start_byte, 0);
    assert_eq!(chunk.end_byte, 0);
    assert_eq!(chunk.code, "");
}

#[test]
fn test_chunk_boundaries_are_correct() {
    let mut parser = Parser::new().unwrap();
    let code = "function foo() { return 1; }\nfunction bar() { return 2; }";

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    let chunks = parser.extract_chunks(&tree, code);

    for chunk in &chunks {
        // Verify chunk boundaries are within code
        assert!(chunk.start_byte <= chunk.end_byte);
        assert!(chunk.end_byte <= code.len());

        // Verify we can extract the chunk code
        let extracted = &code[chunk.start_byte..chunk.end_byte];
        assert!(!extracted.is_empty());
    }
}

#[test]
fn test_parse_arrow_functions() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    const add = (a, b) => a + b;
    const multiply = (a, b) => {
        return a * b;
    };
    "#;

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    let chunks = parser.extract_chunks(&tree, code);

    // Should extract arrow functions
    assert!(chunks.len() >= 2, "Should extract arrow functions");
}

#[test]
fn test_parse_class_with_methods() {
    let mut parser = Parser::new().unwrap();
    let code = r#"
    class Calculator {
        constructor(initial) {
            this.value = initial;
        }

        add(x) {
            this.value += x;
            return this;
        }

        multiply(x) {
            this.value *= x;
            return this;
        }

        get result() {
            return this.value;
        }
    }
    "#;

    let tree = parser.parse(code, Language::JavaScript).unwrap();
    let chunks = parser.extract_chunks(&tree, code);

    // Should extract class and methods
    let has_class = chunks.iter().any(|c| c.node_type == "class_declaration");
    let has_methods = chunks.iter().any(|c| c.node_type == "method_definition");

    assert!(has_class, "Should extract class declaration");
    assert!(has_methods, "Should extract methods");
}
