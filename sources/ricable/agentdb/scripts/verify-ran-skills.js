#!/usr/bin/env node

/**
 * RAN Skills Verification Script
 * Verifies that all 5 RAN skills are properly configured with progressive disclosure
 * and meet the required specifications
 */

const fs = require('fs').promises;
const path = require('path');

// Configuration
const RAN_SKILLS = [
    'ran-agentdb-integration-specialist',
    'ml-researcher',
    'ran-causal-inference-specialist',
    'ran-reinforcement-learning-engineer',
    'ran-dspy-mobility-optimizer'
];

const REQUIRED_YAML_FIELDS = [
    'name',
    'description',
    'category',
    'tags',
    'dependencies',
    'progressive_disclosure',
    'prerequisites',
    'estimated_time',
    'difficulty',
    'performance_targets'
];

const REQUIRED_PROGRESSIVE_LEVELS = [
    'Level 1',
    'Level 2',
    'Level 3'
];

class SkillVerifier {
    constructor() {
        this.issues = [];
        this.warnings = [];
        this.successes = [];
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : type === 'success' ? '✅' : 'ℹ️';
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    async verifySkillDirectory(skill) {
        const skillPath = path.join('.claude/skills', skill);

        try {
            await fs.access(skillPath);
            this.successes.push(`Directory exists for ${skill}`);
            return true;
        } catch {
            this.issues.push(`Directory missing for ${skill}`);
            return false;
        }
    }

    async verifySkillYAML(skill) {
        const yamlPath = path.join('.claude/skills', skill, 'skill.yml');

        try {
            const yamlContent = await fs.readFile(yamlPath, 'utf8');
            const yamlData = this.parseYAML(yamlContent);

            // Check required fields
            let missingFields = [];
            for (const field of REQUIRED_YAML_FIELDS) {
                if (!yamlData[field]) {
                    missingFields.push(field);
                }
            }

            if (missingFields.length > 0) {
                this.issues.push(`${skill}: Missing YAML fields: ${missingFields.join(', ')}`);
            } else {
                this.successes.push(`${skill}: All required YAML fields present`);
            }

            // Verify specific values
            if (yamlData.progressive_disclosure !== true) {
                this.issues.push(`${skill}: progressive_disclosure must be true`);
            }

            if (!yamlData.performance_targets || Object.keys(yamlData.performance_targets).length === 0) {
                this.warnings.push(`${skill}: No performance targets defined`);
            }

            return yamlData;

        } catch (error) {
            this.issues.push(`${skill}: Cannot read skill.yml - ${error.message}`);
            return null;
        }
    }

    async verifySkillMarkdown(skill) {
        const mdPath = path.join('.claude/skills', skill, 'skill.md');

        try {
            const mdContent = await fs.readFile(mdPath, 'utf8');

            // Check for progressive disclosure levels
            let foundLevels = [];
            for (const level of REQUIRED_PROGRESSIVE_LEVELS) {
                if (mdContent.includes(level)) {
                    foundLevels.push(level);
                }
            }

            if (foundLevels.length < 3) {
                const missing = REQUIRED_PROGRESSIVE_LEVELS.filter(l => !foundLevels.includes(l));
                this.issues.push(`${skill}: Missing progressive disclosure levels: ${missing.join(', ')}`);
            } else {
                this.successes.push(`${skill}: All progressive disclosure levels present`);
            }

            // Check for code examples
            if (mdContent.includes('```typescript') || mdContent.includes('```javascript')) {
                this.successes.push(`${skill}: Contains code examples`);
            } else {
                this.warnings.push(`${skill}: No code examples found`);
            }

            // Check for performance targets
            if (mdContent.includes('Performance') || mdContent.includes('Targets')) {
                this.successes.push(`${skill}: Performance targets documented`);
            } else {
                this.warnings.push(`${skill}: Performance targets not documented in markdown`);
            }

            // Estimate documentation length
            const wordCount = mdContent.split(/\s+/).length;
            if (wordCount > 1000) {
                this.successes.push(`${skill}: Comprehensive documentation (${wordCount} words)`);
            } else {
                this.warnings.push(`${skill}: Brief documentation (${wordCount} words)`);
            }

            return mdContent;

        } catch (error) {
            this.issues.push(`${skill}: Cannot read skill.md - ${error.message}`);
            return null;
        }
    }

    parseYAML(yamlContent) {
        // Simple YAML parser for frontmatter
        const data = {};
        const lines = yamlContent.split('\n');

        let inFrontmatter = false;
        let currentKey = null;
        let arrayItems = [];

        for (const line of lines) {
            if (line.trim() === '---') {
                if (!inFrontmatter) {
                    inFrontmatter = true;
                    continue;
                } else {
                    break;
                }
            }

            if (!inFrontmatter) continue;

            const trimmed = line.trim();
            if (trimmed.startsWith('#') || trimmed === '') continue;

            // Array items (with proper indentation handling)
            if (trimmed.startsWith('- ')) {
                const item = trimmed.substring(2).trim().replace(/['"]/g, '');
                if (item) {
                    arrayItems.push(item);
                }
                continue;
            }

            // Handle indented array items (like prerequisites)
            if (trimmed.startsWith('  - ') && currentKey) {
                const item = trimmed.substring(4).trim().replace(/['"]/g, '');
                if (item) {
                    arrayItems.push(item);
                }
                continue;
            }

            // Save previous array if exists
            if (currentKey && arrayItems.length > 0) {
                data[currentKey] = arrayItems;
                arrayItems = [];
            }

            // Key-value pairs
            const match = trimmed.match(/^(\w+):\s*(.*)$/);
            if (match) {
                currentKey = match[1];
                let value = match[2].trim();

                // Handle boolean values
                if (value === 'true') {
                    data[currentKey] = true;
                    arrayItems = [];
                    continue;
                } else if (value === 'false') {
                    data[currentKey] = false;
                    arrayItems = [];
                    continue;
                }

                // Handle inline arrays
                if (value.startsWith('[') && value.endsWith(']')) {
                    data[currentKey] = value.slice(1, -1).split(',').map(item =>
                        item.trim().replace(/['"]/g, '')
                    );
                    arrayItems = [];
                } else if (value) {
                    // Simple value
                    data[currentKey] = value.replace(/['"]/g, '');
                    arrayItems = [];
                }
            }
        }

        // Save any pending array
        if (currentKey && arrayItems.length > 0) {
            data[currentKey] = arrayItems;
        }

        return data;
    }

    async verifyDependencies(skill, yamlData) {
        if (!yamlData || !yamlData.dependencies) {
            return;
        }

        for (const dep of yamlData.dependencies) {
            const depPath = path.join('.claude/skills', dep);
            try {
                await fs.access(depPath);
                this.successes.push(`${skill}: Dependency ${dep} is available`);
            } catch {
                this.warnings.push(`${skill}: Dependency ${dep} is not available`);
            }
        }
    }

    async verifyPerformanceTargets(skill, yamlData) {
        if (!yamlData || !yamlData.performance_targets) {
            return;
        }

        const targets = yamlData.performance_targets;
        if (typeof targets === 'object' && Object.keys(targets).length > 0) {
            this.successes.push(`${skill}: Has ${Object.keys(targets).length} performance targets`);

            // Check for specific target patterns
            for (const [target, value] of Object.entries(targets)) {
                if (typeof value === 'string' && value.includes('%')) {
                    this.successes.push(`${skill}: Percentage target defined for ${target}`);
                }
                if (typeof value === 'string' && value.includes('x')) {
                    this.successes.push(`${skill}: Multiplier target defined for ${target}`);
                }
                if (typeof value === 'string' && value.includes('ms') || value.includes('s')) {
                    this.successes.push(`${skill}: Time target defined for ${target}`);
                }
            }
        }
    }

    async verifyAllSkills() {
        this.log('Starting RAN Skills Verification...');
        this.log(`Checking ${RAN_SKILLS.length} skills`);

        const results = {};

        for (const skill of RAN_SKILLS) {
            this.log(`\nVerifying ${skill}...`);
            results[skill] = {
                directory: false,
                yaml: false,
                markdown: false,
                dependencies: false,
                performance: false
            };

            // Verify directory exists
            if (await this.verifySkillDirectory(skill)) {
                results[skill].directory = true;
            }

            // Verify YAML configuration
            const yamlData = await this.verifySkillYAML(skill);
            if (yamlData) {
                results[skill].yaml = true;

                // Verify dependencies
                await this.verifyDependencies(skill, yamlData);
                if (yamlData.dependencies) {
                    results[skill].dependencies = true;
                }

                // Verify performance targets
                await this.verifyPerformanceTargets(skill, yamlData);
                if (yamlData.performance_targets) {
                    results[skill].performance = true;
                }
            }

            // Verify markdown documentation
            const mdContent = await this.verifySkillMarkdown(skill);
            if (mdContent) {
                results[skill].markdown = true;
            }
        }

        return results;
    }

    generateReport(results) {
        this.log('\n' + '='.repeat(60));
        this.log('RAN SKILLS VERIFICATION REPORT');
        this.log('='.repeat(60));

        // Summary table
        this.log('\n📊 SUMMARY TABLE:');
        this.log('┌─────────────────────────────────────┬──────┬──────┬──────┬──────┬──────┐');
        this.log('│ Skill                                │ Dir  │ YAML │ MD   │ Deps │ Perf │');
        this.log('├─────────────────────────────────────┼──────┼──────┼──────┼──────┼──────┤');

        for (const [skill, result] of Object.entries(results)) {
            const shortName = skill.substring(4).replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const name = shortName.padEnd(37);
            const dir = result.directory ? '✅' : '❌';
            const yaml = result.yaml ? '✅' : '❌';
            const md = result.markdown ? '✅' : '❌';
            const deps = result.dependencies ? '✅' : '❌';
            const perf = result.performance ? '✅' : '❌';
            this.log(`│ ${name} │ ${dir}    │ ${yaml}    │ ${md}    │ ${deps}    │ ${perf}    │`);
        }

        this.log('└─────────────────────────────────────┴──────┴──────┴──────┴──────┴──────┘');

        // Issues and warnings
        if (this.issues.length > 0) {
            this.log('\n❌ ISSUES FOUND:');
            this.issues.forEach(issue => this.log(`  • ${issue}`, 'error'));
        }

        if (this.warnings.length > 0) {
            this.log('\n⚠️ WARNINGS:');
            this.warnings.forEach(warning => this.log(`  • ${warning}`, 'warning'));
        }

        if (this.successes.length > 0) {
            this.log('\n✅ SUCCESSFUL VERIFICATIONS:');
            this.successes.slice(0, 10).forEach(success => this.log(`  • ${success}`, 'success'));
            if (this.successes.length > 10) {
                this.log(`  • ... and ${this.successes.length - 10} more successes`, 'success');
            }
        }

        // Overall status
        const totalChecks = Object.values(results).reduce((sum, r) =>
            sum + Object.values(r).filter(Boolean).length, 0);
        const maxChecks = RAN_SKILLS.length * 5; // 5 checks per skill
        const successRate = Math.round((totalChecks / maxChecks) * 100);

        this.log(`\n📈 OVERALL SUCCESS RATE: ${successRate}% (${totalChecks}/${maxChecks} checks passed)`);

        if (successRate >= 90) {
            this.log('🎉 EXCELLENT: All skills are properly configured!', 'success');
        } else if (successRate >= 70) {
            this.log('👍 GOOD: Most skills are properly configured', 'success');
        } else {
            this.log('⚠️ NEEDS ATTENTION: Some skills need configuration', 'warning');
        }

        return {
            totalChecks,
            maxChecks,
            successRate,
            issues: this.issues.length,
            warnings: this.warnings.length,
            successes: this.successes.length
        };
    }
}

// Main execution
async function main() {
    const verifier = new SkillVerifier();

    try {
        const results = await verifier.verifyAllSkills();
        const report = verifier.generateReport(results);

        // Exit with appropriate code
        if (report.issues > 0) {
            process.exit(1);
        } else if (report.warnings > 0) {
            process.exit(2);
        } else {
            process.exit(0);
        }

    } catch (error) {
        verifier.log(`Fatal error: ${error.message}`, 'error');
        process.exit(3);
    }
}

// Handle command line arguments
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.includes('--help') || args.includes('-h')) {
        console.log('RAN Skills Verification Script');
        console.log('');
        console.log('Usage: node verify-ran-skills.js [options]');
        console.log('');
        console.log('Options:');
        console.log('  --help, -h    Show this help message');
        console.log('');
        console.log('Exit codes:');
        console.log('  0  All checks passed');
        console.log('  1  Issues found');
        console.log('  2  Warnings only');
        console.log('  3  Fatal error');
        process.exit(0);
    }

    main();
}

module.exports = { SkillVerifier };