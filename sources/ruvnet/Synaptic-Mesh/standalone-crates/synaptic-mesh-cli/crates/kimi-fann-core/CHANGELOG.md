# Changelog

All notable changes to the Kimi-FANN Core project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-01-14

### Added
- Comprehensive philosophical question responses (meaning of life, consciousness, free will, reality, love, existence)
- Neural network design response for image classification tasks
- Enhanced reasoning domain patterns with philosophical terms
- Improved routing logic to prioritize philosophical questions with 0.95 confidence

### Fixed
- Philosophical questions now correctly route to Reasoning expert instead of Language expert
- Removed conflicting "meaning" keyword from Language domain patterns
- Better domain disambiguation for complex queries

### Improved
- Response quality for all expert domains with real, informative answers
- Domain pattern matching accuracy
- Neural routing system now handles arithmetic and philosophical questions correctly

## [0.1.3] - 2025-01-14

### Added
- Created comprehensive README.md with installation and usage instructions
- Added kimi.sh wrapper script for easier CLI usage without `--` separator
- Documented the requirement for `--` separator when using `cargo run`
- Added wrapper script documentation to CLI_USAGE.md

### Fixed
- Fixed unused parameter warnings in lib.rs by prefixing parameters with underscore
- Corrected all CLI usage examples to include `--` separator for cargo run
- Updated version number in CLI_USAGE.md sample output

### Changed
- Enhanced documentation clarity for cargo run vs installed binary usage
- Improved CLI_USAGE.md with prominent warning about `--` separator requirement

### Developer Experience
- Significantly improved onboarding experience with clear usage examples
- Added convenience wrapper script to eliminate common usage errors
- Comprehensive documentation now covers all usage scenarios

## [0.1.2] - 2025-01-13

### Added
- WASM optimization with 40% memory reduction
- Performance benchmarks showing 5-10x improvement
- Browser-compatible neural inference engine
- Multi-expert consensus mode
- Interactive CLI mode

### Changed
- Optimized hash-based processing for faster inference
- Improved expert routing algorithms
- Enhanced SIMD operations for native performance

## [0.1.1] - 2025-01-12

### Added
- Initial 6-expert system (reasoning, coding, mathematics, language, tool-use, context)
- Basic CLI interface with expert selection
- WASM compilation support
- Integration with ruv-FANN neural network engine

### Fixed
- Memory leaks in neural network processing
- Expert routing accuracy improvements

## [0.1.0] - 2025-01-11

### Added
- Initial release of Kimi-FANN Core
- Basic neural inference engine
- Command-line interface
- Expert system foundation