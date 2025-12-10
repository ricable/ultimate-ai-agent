# Troubleshooting Guide

Common issues and solutions for Ericsson RAN features.

## Common Issues

### Feature Not Activating

**Possible Causes**:
- CXC code not found or incorrect
- Prerequisites not met
- Feature license missing
- MO instance doesn't exist
- Parameter conflicts

**Troubleshooting Steps**:
1. Verify CXC code in feature documentation
2. Check feature prerequisites are activated
3. Confirm license is available and valid
4. Validate MO instance exists: `moget -class FeatureState`
5. Review parameter configurations for conflicts

### Performance Degradation After Activation

**Possible Causes**:
- Parameter settings not optimized
- Feature conflicts with existing configuration
- Network conditions changed
- Hardware capacity limitations
- Feature interaction issues

**Troubleshooting Steps**:
1. Compare performance counters before/after activation
2. Review parameter recommendations in guidelines
3. Check for related feature conflicts
4. Analyze traffic patterns and network load
5. Consider partial rollback or parameter adjustment

### Counters Not Updating

**Possible Causes**:
- Feature not properly activated
- Counter collection not enabled
- Reporting interval too long
- PM collection configuration issue
- Feature not generating expected events

**Troubleshooting Steps**:
1. Verify feature activation status
2. Check PM configuration for counter collection
3. Review reporting intervals
4. Monitor events in real-time
5. Validate feature is generating expected activity

## Feature-Specific Troubleshooting

### Carrier Aggregation Issues

#### TM8 Mode Switching

**CXC Code**: CXC4011996
**Common Issues**:
- Review specific parameter configurations
- Check feature interactions within category
- Verify network conditions support feature
- Monitor relevant performance counters

#### Cell ID-Based Location Support

**CXC Code**: CXC4010841
**Common Issues**:
- Review specific parameter configurations
- Check feature interactions within category
- Verify network conditions support feature
- Monitor relevant performance counters

#### Dynamic PUCCH

**CXC Code**: CXC4011955
**Common Issues**:
- Review specific parameter configurations
- Check feature interactions within category
- Verify network conditions support feature
- Monitor relevant performance counters

### Other Features Issues

#### Prescheduling

**CXC Code**: CXC4011715
**Common Issues**:
- Review specific parameter configurations
- Check feature interactions within category
- Verify network conditions support feature
- Monitor relevant performance counters

### Energy Efficiency Issues

#### MIMO Sleep Mode

**CXC Code**: CXC4011808
**Common Issues**:
- Review specific parameter configurations
- Check feature interactions within category
- Verify network conditions support feature
- Monitor relevant performance counters
