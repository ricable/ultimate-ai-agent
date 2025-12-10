# Performance Counter Analysis

Guide to analyzing performance counters for troubleshooting.

## Counter Analysis Approach

1. **Baseline Establishment**
   - Establish normal performance baselines
   - Identify key performance indicators
   - Set monitoring thresholds

2. **Trend Analysis**
   - Monitor counter trends over time
   - Identify gradual degradation
   - Correlate with network changes

3. **Threshold Monitoring**
   - Set appropriate alert thresholds
   - Monitor for counter spikes
   - Investigate threshold breaches

## Key Counter Categories

### MIMO Performance
- **pmMimoSleepTime**: Time spent in MIMO sleep mode
- **pmRadioTxRankDistr**: Distribution of MIMO ranks
- **pmMimoSleepOppTime**: Opportunities for MIMO sleep

### Energy Efficiency
- **pmTxOffTime**: Time with TX off
- **pmTxOffRatio**: Percentage of TX off time
- **pmPdcpPktDiscDlAqm**: PDCP packet discard due to AQM

### Mobility Performance
- **pmHoSuccessRate**: Handover success rate
- **pmRrcConnEstabSuccessRate**: RRC connection establishment success
- **pmUeThpTimeDl**: UE throughput time

