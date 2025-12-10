# Feature Activation Guide

Comprehensive guide for activating Ericsson RAN features.

## Activation Process

### 1. Preparation Phase

- Review feature documentation
- Verify license requirements
- Check software/hardware compatibility
- Plan activation window
- Prepare rollback procedures

### 2. Configuration Phase

- Set required parameters
- Validate parameter consistency
- Verify MO instances exist
- Check prerequisite features

### 3. Activation Phase

```bash
# Standard activation command
Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=<CXC_CODE> MO instance.
```

### 4. Verification Phase

- Monitor activation events
- Check feature status
- Validate performance counters
- Confirm expected behavior

## Sample Activation Procedures

### TM8 Mode Switching

**CXC Code**: CXC4011996
**FAJ ID**: FAJ 121 4508

**Activation Command**:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011996 MO instance.
```

### Prescheduling

**CXC Code**: CXC4011715
**FAJ ID**: FAJ 121 3085

**Activation Command**:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011715 MO instance.
```

### Cell ID-Based Location Support

**CXC Code**: CXC4010841
**FAJ ID**: FAJ 121 0735

**Activation Command**:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4010841 MO instance.
```

### MIMO Sleep Mode

**CXC Code**: CXC4011808
**FAJ ID**: FAJ 121 3094

**Activation Command**:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011808 MO instance.
```

### Dynamic PUCCH

**CXC Code**: CXC4011955
**FAJ ID**: FAJ 121 4377

**Activation Command**:
```bash
1. Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011955 MO instance.
```
