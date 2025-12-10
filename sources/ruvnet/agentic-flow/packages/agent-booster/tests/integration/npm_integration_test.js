/**
 * Integration tests for NPM SDK
 * Tests auto-detection of native vs WASM bindings
 */

const assert = require('assert');
const path = require('path');
const fs = require('fs');

// Test configuration
const TEST_TIMEOUT = 30000;

describe('NPM SDK Integration Tests', function() {
    this.timeout(TEST_TIMEOUT);

    let AgentBooster;

    before(function() {
        // Try to load the agent-booster package
        try {
            const pkgPath = path.join(__dirname, '../../npm/agent-booster');
            AgentBooster = require(pkgPath);
            console.log('✓ Successfully loaded agent-booster package');
        } catch (error) {
            console.error('Failed to load agent-booster:', error.message);
            this.skip();
        }
    });

    describe('Module Loading', function() {
        it('should load the module successfully', function() {
            assert.ok(AgentBooster);
            assert.equal(typeof AgentBooster.applyEdit, 'function');
            assert.equal(typeof AgentBooster.batchApply, 'function');
        });

        it('should report which binding is loaded (native or wasm)', function() {
            if (AgentBooster.getImplementation) {
                const impl = AgentBooster.getImplementation();
                console.log(`  Using implementation: ${impl}`);
                assert.ok(['native', 'wasm', 'fallback'].includes(impl));
            }
        });
    });

    describe('Basic Edit Operations', function() {
        it('should apply a simple function edit', async function() {
            const original = 'function foo() { return 1; }';
            const edit = 'function foo() { return 2; }';

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('return 2'));
            assert.ok(result.confidence > 0.7);
            assert.equal(result.metadata.syntaxValid, true);
        });

        it('should handle TypeScript edits', async function() {
            const original = 'interface User { id: number; name: string; }';
            const edit = 'interface Product { id: number; price: number; }';

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'typescript',
                confidenceThreshold: 0.3
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('User'));
            assert.ok(result.mergedCode.includes('Product'));
        });

        it('should preserve surrounding code', async function() {
            const original = `
// Header
function foo() { return 1; }
// Middle
function bar() { return 2; }
// Footer
            `.trim();

            const edit = 'function foo() { return 99; }';

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            assert.ok(result.mergedCode.includes('Header'));
            assert.ok(result.mergedCode.includes('Middle'));
            assert.ok(result.mergedCode.includes('Footer'));
            assert.ok(result.mergedCode.includes('return 99'));
            assert.ok(result.mergedCode.includes('bar'));
        });
    });

    describe('Batch Processing', function() {
        it('should process multiple edits', async function() {
            const edits = [
                {
                    originalCode: 'function foo() { return 1; }',
                    editSnippet: 'function foo() { return 2; }',
                    language: 'javascript',
                    confidenceThreshold: 0.5
                },
                {
                    originalCode: 'function bar() { return 3; }',
                    editSnippet: 'function bar() { return 4; }',
                    language: 'javascript',
                    confidenceThreshold: 0.5
                }
            ];

            const results = await AgentBooster.batchApply(edits);

            assert.equal(results.length, 2);
            assert.ok(results[0].mergedCode.includes('return 2'));
            assert.ok(results[1].mergedCode.includes('return 4'));
        });
    });

    describe('Error Handling', function() {
        it('should handle invalid syntax gracefully', async function() {
            const original = 'function valid() { return 1; }';
            const edit = 'function invalid() { return 1;'; // Missing }

            try {
                await AgentBooster.applyEdit({
                    originalCode: original,
                    editSnippet: edit,
                    language: 'javascript',
                    confidenceThreshold: 0.5
                });
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error.message);
            }
        });

        it('should handle low confidence matches', async function() {
            const original = 'function foo() { return 1; }';
            const edit = 'completely unrelated code';

            try {
                await AgentBooster.applyEdit({
                    originalCode: original,
                    editSnippet: edit,
                    language: 'javascript',
                    confidenceThreshold: 0.9
                });
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error.message.toLowerCase().includes('confidence') ||
                         error.message.toLowerCase().includes('match'));
            }
        });

        it('should validate input parameters', async function() {
            try {
                await AgentBooster.applyEdit({
                    // Missing originalCode
                    editSnippet: 'test',
                    language: 'javascript'
                });
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error.message);
            }
        });
    });

    describe('Real-world Fixtures', function() {
        it('should process sample JavaScript file', async function() {
            const fixturePath = path.join(__dirname, '../fixtures/sample_javascript.js');

            if (!fs.existsSync(fixturePath)) {
                this.skip();
                return;
            }

            const original = fs.readFileSync(fixturePath, 'utf8');
            const edit = `
function calculateDifference(a, b) {
    return a - b;
}
            `;

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.3
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('calculateDifference'));
            assert.ok(result.mergedCode.includes('calculateSum'));
            assert.ok(result.mergedCode.includes('MathOperations'));
        });

        it('should process sample TypeScript file', async function() {
            const fixturePath = path.join(__dirname, '../fixtures/sample_typescript.ts');

            if (!fs.existsSync(fixturePath)) {
                this.skip();
                return;
            }

            const original = fs.readFileSync(fixturePath, 'utf8');
            const edit = `
interface Admin extends User {
    role: string;
    permissions: string[];
}
            `;

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'typescript',
                confidenceThreshold: 0.3
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('Admin'));
            assert.ok(result.mergedCode.includes('User'));
            assert.ok(result.mergedCode.includes('permissions'));
        });
    });

    describe('Performance', function() {
        it('should process edits in reasonable time', async function() {
            const original = 'function test() { return 1; }';
            const edit = 'function test() { return 2; }';

            const start = Date.now();

            await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            const duration = Date.now() - start;

            // Should complete in less than 1 second
            assert.ok(duration < 1000, `Processing took ${duration}ms`);
        });

        it('should handle large files efficiently', async function() {
            // Generate a large file
            const lines = [];
            for (let i = 0; i < 100; i++) {
                lines.push(`function func${i}() { return ${i}; }`);
            }
            const original = lines.join('\n\n');
            const edit = 'function func50() { return 999; }';

            const start = Date.now();

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            const duration = Date.now() - start;

            assert.ok(result);
            assert.ok(result.mergedCode.includes('return 999'));
            // Should complete in reasonable time even for large files
            assert.ok(duration < 5000, `Processing took ${duration}ms`);
        });
    });

    describe('Edge Cases', function() {
        it('should handle empty files', async function() {
            const original = '';
            const edit = 'function test() { return 1; }';

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.1
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('function test'));
        });

        it('should handle files with only whitespace', async function() {
            const original = '   \n\n   \n   ';
            const edit = 'function test() { return 1; }';

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.1
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('function test'));
        });

        it('should handle Unicode characters', async function() {
            const original = "function greet() { return '你好'; }";
            const edit = "function greet() { return 'Hello'; }";

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('Hello'));
        });

        it('should handle very long lines', async function() {
            const longString = 'x'.repeat(10000);
            const original = `function test() { return '${longString}'; }`;
            const edit = `function test() { return 'short'; }`;

            const result = await AgentBooster.applyEdit({
                originalCode: original,
                editSnippet: edit,
                language: 'javascript',
                confidenceThreshold: 0.5
            });

            assert.ok(result);
            assert.ok(result.mergedCode.includes('short'));
        });
    });
});

// Run tests if this file is executed directly
if (require.main === module) {
    console.log('Running NPM Integration Tests...\n');

    // Simple test runner
    const Mocha = require('mocha');
    const mocha = new Mocha({
        timeout: TEST_TIMEOUT,
        reporter: 'spec'
    });

    mocha.addFile(__filename);

    mocha.run(failures => {
        process.exitCode = failures ? 1 : 0;
    });
}

module.exports = {
    // Export for external test runners
};
