# Activation Commands Cheat Sheet

Quick reference for feature activation commands.

## Command Templates

### Basic Activation
```bash
# Activate a feature
set FeatureState.featureState=ACTIVATED FeatureState=<CXC_CODE>

# Deactivate a feature
set FeatureState.featureState=DEACTIVATED FeatureState=<CXC_CODE>
```

### Parameter Configuration
```bash
# Set a parameter value
set <MO_Class>.<ParameterName>=<Value> <MO_Instance>

# Example: Set MIMO sleep mode
set MimoSleepFunction.sleepMode=1 MimoSleepFunction=1
```

### Verification Commands
```bash
# Check feature status
get FeatureState.featureState FeatureState=<CXC_CODE>

# Check parameter value
get <MO_Class>.<ParameterName> <MO_Instance>

# List all features
lst FeatureState
```

## Common CXC Codes

- **CXC4010841**: Cell ID-Based Location Support
- **CXC4011715**: Prescheduling
- **CXC4011808**: MIMO Sleep Mode
- **CXC4011955**: Dynamic PUCCH
- **CXC4011996**: TM8 Mode Switching
