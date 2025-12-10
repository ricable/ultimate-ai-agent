# Quick Reference Guide

Essential information for Ericsson RAN Features at a glance.

## Feature Activation Quick Reference

### Common Activation Pattern

```bash
# Standard activation command
set FeatureState.featureState=ACTIVATED FeatureState=<CXC_CODE>

# Standard deactivation command
set FeatureState.featureState=DEACTIVATED FeatureState=<CXC_CODE>
```

### Top Features by Category

#### Energy Efficiency

- **MIMO Sleep Mode** (FAJ 121 3094) - CXC CXC4011808

#### Other

- **Cell ID-Based Location Support** (FAJ 121 0735) - CXC CXC4010841
- **Dynamic PUCCH** (FAJ 121 4377) - CXC CXC4011955
- **Prescheduling** (FAJ 121 3085) - CXC CXC4011715
- **TM8 Mode Switching** (FAJ 121 4508) - CXC CXC4011996

