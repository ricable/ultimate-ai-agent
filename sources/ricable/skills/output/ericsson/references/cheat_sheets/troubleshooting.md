# Troubleshooting Cheat Sheet

Quick troubleshooting steps for common issues.

## Quick Diagnostic Commands

```bash
# Check feature status
get FeatureState.featureState FeatureState=<CXC_CODE>

# Check alarm status
lst Alarm

# Check performance counters
get <CounterName> <PM_Object>

# Check cell status
get UtranCellFDD.operationalState UtranCellFDD=<CellId>
```

## Common Issues and Solutions

### Feature Not Working

**Checks**:
1. Verify feature activation: `get FeatureState.featureState FeatureState=<CXC_CODE>`
2. Check prerequisites are activated
3. Verify software version compatibility
4. Check for related alarms

### Performance Degradation

**Checks**:
1. Review performance counters
2. Check recent configuration changes
3. Analyze interference levels
4. Verify capacity utilization

### MIMO Issues

**Checks**:
1. Check UE MIMO capabilities
2. Verify antenna configuration
3. Review MIMO parameter settings
4. Monitor MIMO performance counters

### Energy Saving Issues

**Checks**:
1. Verify sleep mode thresholds
2. Check traffic patterns
3. Review timer configurations
4. Monitor QoS impact

