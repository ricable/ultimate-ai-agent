# Common Tasks Quick Reference

Quick reference for frequently performed tasks with Ericsson RAN features.

## Feature Activation Checklist

### Before Activation

- [ ] Verify feature prerequisites
- [ ] Check feature compatibility
- [ ] Backup current configuration
- [ ] Review parameter recommendations
- [ ] Plan monitoring strategy

### Activation Steps

- [ ] Configure required parameters
- [ ] Set FeatureState.featureState to ACTIVATED
- [ ] Monitor activation events
- [ ] Verify feature behavior
- [ ] Check performance counters

### Post-Activation

- [ ] Monitor KPIs for 24+ hours
- [ ] Validate user experience impact
- [ ] Document any deviations
- [ ] Update network documentation

## Common Parameter Configurations

### INTERNAL_PROC_MIMO_SLEEP_SWITCHED

Used in 12 features

### INTERNAL_EVENT_MIMO_SLEEP_DETECTED

Used in 8 features

### FeatureState.featureState

Used in 2 features

### EUtranCellTDD.transmissionMode

Used in 1 features

### Uen.AC

Used in 1 features

### ENodeBFunction.initPreschedulingEnable

Used in 1 features

### PreschedProfile.preschedulingDataSize  PreschedulingProfile.preschedulingDataSize

Used in 1 features

### PreschedProfile.preschedulingPeriod  PreschedulingProfile.preschedulingPeriod

Used in 1 features

### PreschedProfile.preschedulingDuration  PreschedulingProfile.preschedulingDuration

Used in 1 features

### EUtranCellFDD.prescheduling  EUtranCellTDD.prescheduling

Used in 1 features

### Uen.AY

Used in 1 features

### PreschedProfile.preschedulingDataSize

Used in 1 features

### PreschedulingProfile.preschedulingDataSize

Used in 1 features

### PreschedProfile.preschedulingPeriod

Used in 1 features

### PreschedulingProfile.preschedulingPeriod

Used in 1 features

### PreschedProfile.preschedulingDuration

Used in 1 features

### PreschedulingProfile.preschedulingDuration

Used in 1 features

### ENodeBFunction.locationReportForPSCell

Used in 1 features

### Uen.AS

Used in 1 features

### A.a

Used in 1 features

## Quick Troubleshooting Steps

### Feature Not Activating

1. Check CXC code is correct
2. Verify prerequisites are met
3. Check parameter values
4. Review event logs
5. Validate MO instance exists

### Unexpected Performance Impact

1. Review counter trends before/after activation
2. Check related feature interactions
3. Verify parameter configurations
4. Consider traffic pattern changes
5. Plan rollback if needed
