//! Polyglot Rust Environment
//!
//! This is a library crate for the Rust component of the polyglot development environment.

/// A simple greeting function
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

/// A simple calculation function
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet("World"), "Hello, World!");
    }

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
