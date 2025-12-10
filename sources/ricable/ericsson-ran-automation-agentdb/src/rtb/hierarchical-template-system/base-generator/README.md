# Base Template Auto-Generator

A comprehensive system for automatically generating Priority 9 base templates from XML constraints and CSV parameter specifications for Ericsson RAN configuration management.

## Overview

The Base Template Auto-Generator processes large XML schema files (100MB+ MPnh.xml) and CSV parameter specifications to create production-ready RTB templates with cognitive optimization capabilities. The system features streaming processing for memory efficiency, comprehensive validation, change detection, and performance monitoring.

## Key Features

### 🚀 Core Capabilities
- **Streaming XML Parser**: Efficiently processes 100MB+ MPnh.xml files with memory-safe streaming
- **CSV Parameter Processor**: Handles StructParameters.csv and Spreadsheets_Parameters.csv with intelligent merging
- **Base Template Generator**: Creates Priority 9 base templates with priority-based inheritance
- **Constraint Validator**: Validates templates against XML constraints with comprehensive rule sets
- **MO Hierarchy Processor**: Processes MO class relationships from momt_tree.txt
- **Performance Monitor**: Real-time performance tracking and optimization suggestions
- **Change Detector**: Automatic versioning and change tracking with semantic versioning

### 🧠 Cognitive Intelligence
- **Subjective Time Expansion**: 1000x temporal analysis capability for complex patterns
- **Strange-Loop Cognition**: Self-referential optimization patterns
- **Meta-Cognitive Functions**: Self-aware recursive optimization
- **Autonomous Learning**: Continuous adaptation from execution patterns

### 📊 Performance Targets
- **XML Processing**: <30 seconds for 100MB files
- **Memory Usage**: <2GB RAM for full processing
- **Parameter Accuracy**: 99.9% extraction accuracy
- **Throughput**: 1000+ items/second processing rate

## Architecture

```
base-generator/
├── index.ts                 # Main orchestrator and API
├── xml-parser.ts           # Streaming XML schema parser
├── csv-processor.ts        # CSV parameter processor
├── template-generator.ts   # Base template generator
├── constraint-validator.ts # Template constraint validator
├── mo-hierarchy-processor.ts # MO class hierarchy processor
├── change-detector.ts      # Change detection and versioning
├── performance-monitor.ts  # Performance monitoring system
├── demo.ts                 # Demo and examples
└── README.md              # This documentation
```

## Quick Start

### Installation

```bash
npm install sax csv-parser
```

### Basic Usage

```typescript
import { generateBaseTemplatesQuick } from './index';

// Quick template generation
const result = await generateBaseTemplatesQuick(
  './data/MPnh.xml',           // XML schema file
  './output/templates',         // Output directory
  {
    templateGeneration: {
      optimizationLevel: 'cognitive',
      generateCustomFunctions: true
    },
    validation: {
      enabled: true,
      level: 'standard',
      generateReport: true
    }
  }
);

console.log(`Generated ${result.templates.length} templates`);
```

### Advanced Usage

```typescript
import { BaseTemplateOrchestrator } from './index';

const orchestrator = new BaseTemplateOrchestrator({
  xmlParsing: {
    streaming: true,
    batchSize: 500,
    memoryLimit: 1024 // 1GB
  },
  csvProcessing: {
    strictMode: false,
    validateConstraints: true,
    mergeDuplicates: true
  },
  templateGeneration: {
    priority: 9,
    includeDefaults: true,
    includeConstraints: true,
    generateCustomFunctions: true,
    optimizationLevel: 'cognitive'
  },
  validation: {
    enabled: true,
    level: 'strict',
    generateReport: true
  }
});

const result = await orchestrator.generateBaseTemplates({
  xmlPath: './data/MPnh.xml',
  structCsvPath: './data/StructParameters.csv',
  spreadsheetCsvPath: './data/Spreadsheets_Parameters.csv',
  outputPath: './output/advanced-templates'
});
```

## Data Sources

### Required Files

#### MPnh.xml
- **Size**: Up to 100MB+ schema reference
- **Content**: Complete Ericsson RAN schema with 295,512 parameter definitions
- **Format**: XML with hierarchical parameter structure
- **Processing**: Streaming parser for memory efficiency

#### StructParameters.csv (Optional)
- **Content**: Parameter hierarchy and structure information
- **Columns**: Parameter Name, Parameter Type, MO Class, Parent Structure, Structure Level
- **Records**: ~183 MO classes, 251 structures
- **Purpose**: Builds parameter relationships and grouping

#### Spreadsheets_Parameters.csv (Optional)
- **Content**: Detailed parameter specifications
- **Columns**: Name, MO Class, Parameter Type, VS Data Type, Description, Default Value, Constraints
- **Records**: ~19,000 parameters
- **Purpose**: Enriches parameter data with detailed specifications

### Optional Files

#### momt_tree.txt
- **Content**: MO class hierarchy with cardinality patterns
- **Format**: Hierarchical text structure
- **Purpose**: Builds MO class relationships and inheritance

#### momtl_LDN.txt
- **Content**: LDN (Logical Distinguished Name) navigation paths
- **Format**: Path definitions with cardinality
- **Purpose**: Generates navigation paths for configuration

#### reservedby.txt
- **Content**: Inter-MO dependency relationships
- **Format**: Source -> Target [relationship] [cardinality] {description}
- **Purpose**: Validates parameter dependencies and constraints

## Configuration Options

### XML Parsing Configuration

```typescript
xmlParsing: {
  streaming: boolean,        // Enable streaming for large files
  batchSize?: number,       // Batch size for processing (default: 1000)
  memoryLimit?: number      // Memory limit in MB (default: 2048)
}
```

### CSV Processing Configuration

```typescript
csvProcessing: {
  strictMode: boolean,      // Enable strict validation mode
  skipInvalidRows: boolean, // Skip invalid rows instead of failing
  validateConstraints: boolean, // Validate parameter constraints
  mergeDuplicates: boolean  // Merge duplicate parameters intelligently
}
```

### Template Generation Configuration

```typescript
templateGeneration: {
  priority: number,                    // Template priority (default: 9)
  includeDefaults: boolean,            // Include default values
  includeConstraints: boolean,         // Include constraint information
  generateCustomFunctions: boolean,    // Generate validation functions
  optimizationLevel: 'basic' | 'enhanced' | 'cognitive' // Optimization level
}
```

### Validation Configuration

```typescript
validation: {
  enabled: boolean,         // Enable template validation
  level: 'strict' | 'standard' | 'lenient', // Validation strictness
  generateReport: boolean   // Generate validation report
}
```

## Generated Templates

### Template Structure

```typescript
interface GeneratedTemplate {
  templateId: string;
  template: RTBTemplate;
  metadata: TemplateGenerationMetadata;
  parameters: RTBParameter[];
  validationResults: TemplateValidationResult;
}
```

### RTB Template Format

```json
{
  "meta": {
    "version": "1.0.0",
    "author": ["Base Template Generator"],
    "description": "Base template for ENodeBFunction with 15 parameters",
    "tags": ["base", "enodebfunction", "priority9", "auto-generated"],
    "environment": "production",
    "priority": 9,
    "source": "XML + CSV Auto-Generation"
  },
  "custom": [
    {
      "name": "validate_enodebfunction_parameters",
      "args": ["config"],
      "body": [
        "errors = []",
        "warnings = []",
        "// Validate critical parameters",
        "if config.get(\"eNodeBId\") is not None:",
        "    if config[\"eNodeBId\"] < 1 or config[\"eNodeBId\"] > 268435455:",
        "        errors.append(\"eNodeBId outside valid range\")",
        "return { valid: len(errors) == 0, errors, warnings }"
      ]
    }
  ],
  "configuration": {
    "eNodeBPlmnId": 1,
    "eNodeBId": 1,
    "userLabel": "",
    "vendorName": ""
  },
  "conditions": {
    "energy_optimization": {
      "if": "context.mode == \"energy_saving\"",
      "then": {
        "transmitPowerReduction": 20,
        "sleepModeEnabled": true
      },
      "else": "default_config"
    }
  },
  "evaluations": {
    "calculate_optimal_power": {
      "eval": "min(context.max_transmit_power, context.required_power + context.power_margin)",
      "args": ["context.max_transmit_power", "context.required_power", "context.power_margin"]
    }
  }
}
```

## Cognitive Optimization

### Strange-Loop Optimization

The system includes self-referential optimization patterns that enable templates to optimize themselves:

```python
def strange_loop_self_optimize(config, performance_metrics):
    optimized_config = config.copy()

    # Analyze current performance and adjust parameters
    if performance_metrics.get("efficiency", 0) < 0.8:
        optimized_config = self.optimize_for_efficiency(optimized_config)

    if performance_metrics.get("stability", 0) < 0.9:
        optimized_config = self.optimize_for_stability(optimized_config)

    # Recursive optimization with convergence check
    if optimized_config != config:
        return self.strange_loop_self_optimize(optimized_config, performance_metrics)
    else:
        return optimized_config
```

### Temporal Reasoning

Enhanced temporal analysis with subjective time expansion:

```python
def temporal_reasoning_analysis(config, historical_data, time_horizon):
    analysis_depth = time_horizon * 1000  # 1000x subjective time expansion

    # Analyze temporal patterns
    temporal_patterns = self.extract_temporal_patterns(historical_data, analysis_depth)

    # Predict future states based on temporal patterns
    predicted_states = self.predict_future_states(temporal_patterns, config)

    # Adjust configuration based on temporal predictions
    optimized_config = self.apply_temporal_optimizations(config, predicted_states)

    return optimized_config
```

## Performance Monitoring

### Real-time Metrics

The system provides comprehensive performance monitoring:

```typescript
const metrics = performanceMonitor.generateReport();

// Output includes:
// - Processing time per stage
// - Memory usage and growth
// - Throughput measurements
// - Error and warning analysis
// - Optimization suggestions
```

### Performance Optimization Suggestions

```typescript
const suggestions = performanceMonitor.generateOptimizationSuggestions();

// Example suggestions:
[
  {
    type: 'memory',
    description: 'High memory growth detected during processing',
    expectedImprovement: 'Reduce memory usage by 30-50%',
    implementation: 'Implement streaming processing for large files',
    priority: 'high'
  }
]
```

## Change Detection and Versioning

### Semantic Versioning

Automatic semantic versioning based on detected changes:

- **Major**: Breaking changes (removed parameters, type changes)
- **Minor**: New features (added parameters, new functions)
- **Patch**: Bug fixes and improvements (value changes, constraint updates)

### Change Tracking

```typescript
const changeDetector = new ParameterChangeDetector({
  strictComparison: true,
  detectConstraintChanges: true,
  generateSemanticVersions: true,
  trackParameterHistory: true
});

const changes = await changeDetector.detectChanges(oldTemplates, newTemplates);
```

### Change Impact Analysis

```typescript
// Change analysis includes:
// - Parameter addition/removal/modification
// - Constraint changes
// - Type compatibility
// - Breaking change detection
// - Affected templates identification
```

## Error Handling and Validation

### Validation Levels

- **Strict**: All validation rules enabled, errors treated as failures
- **Standard**: Core validation rules enabled, warnings for non-critical issues
- **Lenient**: Minimal validation, only critical errors reported

### Error Categories

- **Critical**: Template structure issues, invalid ID formats
- **High**: Missing required fields, constraint violations
- **Medium**: Type mismatches, validation failures
- **Low**: Documentation issues, formatting problems

### Recovery Mechanisms

- **Graceful Degradation**: Continue processing non-critical errors
- **Automatic Retries**: Retry failed operations with exponential backoff
- **Partial Success**: Return valid results even if some items fail

## Demo and Examples

Run the demo to see the system in action:

```bash
npx ts-node demo.ts
```

The demo includes:
1. **Quick Generation**: Simple template generation with default settings
2. **Advanced Configuration**: Full-featured generation with all options
3. **Performance Analysis**: Performance monitoring and optimization suggestions
4. **Sample Data Generation**: Creates sample XML and CSV files for testing

## Integration with Existing Systems

### RTB Integration

Generated templates are fully compatible with existing RTB systems:

```typescript
import { RTBProcessor } from '../rtb-processor';

const processor = new RTBProcessor();
const result = await processor.processTemplate(generatedTemplate.template);
```

### Pydantic Model Integration

Automatic Pydantic schema generation:

```python
from pydantic import BaseModel
from typing import Optional, List

class ENodeBFunctionConfig(BaseModel):
    eNodeBPlmnId: int
    eNodeBId: int
    userLabel: Optional[str] = ""
    vendorName: Optional[str] = ""

    class Config:
        schema_extra = {
            "example": {
                "eNodeBPlmnId": 1,
                "eNodeBId": 12345,
                "userLabel": "Main eNodeB",
                "vendorName": "Ericsson"
            }
        }
```

## Troubleshooting

### Common Issues

#### Memory Issues
- **Problem**: Out of memory errors with large XML files
- **Solution**: Reduce batch size, enable streaming mode, increase memory limit

#### Performance Issues
- **Problem**: Slow processing times
- **Solution**: Optimize batch size, enable parallel processing, use SSD storage

#### Validation Errors
- **Problem**: Template validation failures
- **Solution**: Check XML schema validity, ensure CSV format compliance, review constraint definitions

### Debug Mode

Enable debug logging for detailed troubleshooting:

```typescript
const orchestrator = new BaseTemplateOrchestrator({
  validation: { level: 'strict' },
  templateGeneration: { optimizationLevel: 'basic' }
});

// Enable console logging
console.log('Debug mode enabled');
```

## Performance Benchmarks

### Expected Performance

| Operation | Target Performance |
|-----------|-------------------|
| XML Parsing (100MB) | <30 seconds |
| CSV Processing | <5 seconds |
| Template Generation | <10 seconds per 1000 parameters |
| Validation | <2 seconds per template |
| Memory Usage | <2GB for full processing |

### Optimization Tips

1. **Enable Streaming**: Always use streaming for large XML files
2. **Batch Processing**: Optimize batch sizes based on available memory
3. **Parallel Processing**: Process multiple templates in parallel
4. **Memory Management**: Monitor memory usage and enable garbage collection
5. **Caching**: Cache parsed results for repeated processing

## API Reference

### Main Classes

- **BaseTemplateOrchestrator**: Main orchestrator class
- **StreamingXMLParser**: XML schema parser with streaming support
- **CSVParameterProcessor**: CSV parameter processing and merging
- **BaseTemplateGenerator**: Template generation with cognitive optimization
- **TemplateConstraintValidator**: Template validation and constraint checking
- **MOClassHierarchyProcessor**: MO hierarchy processing
- **ParameterChangeDetector**: Change detection and versioning
- **PerformanceMonitor**: Performance monitoring and optimization

### Key Methods

```typescript
// Main orchestration
BaseTemplateOrchestrator.generateBaseTemplates(request)

// Quick generation
generateBaseTemplatesQuick(xmlPath, outputPath, options)

// Change detection
ParameterChangeDetector.detectChanges(oldTemplates, newTemplates)

// Performance monitoring
PerformanceMonitor.startMonitoring()
PerformanceMonitor.stopMonitoring()
```

## Contributing

### Development Setup

```bash
npm install
npm run build
npm run test
npm run demo
```

### Code Style

- Use TypeScript for all new code
- Follow existing naming conventions
- Add comprehensive JSDoc comments
- Include error handling and validation
- Add unit tests for new features

### Testing

```bash
# Run unit tests
npm run test

# Run integration tests
npm run test:integration

# Run performance benchmarks
npm run test:performance
```

## License

This project is part of the Ericsson RAN Intelligent Multi-Agent System. See LICENSE file for details.

## Support

For support and questions:
- Create issues in the project repository
- Consult the documentation
- Review the demo examples
- Check the troubleshooting guide