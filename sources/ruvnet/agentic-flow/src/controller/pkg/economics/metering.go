package economics

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// MeteringEngine handles usage tracking and metering
type MeteringEngine struct {
	config  *MeteringConfig
	buffer  chan *UsageRecord
	store   UsageStore
	mu      sync.RWMutex
	stopped chan struct{}
}

// UsageStore interface for persistence
type UsageStore interface {
	SaveUsageRecord(ctx context.Context, record *UsageRecord) error
	GetUsageSummary(ctx context.Context, userID string, period string) (*UsageSummary, error)
	ListUsageRecords(ctx context.Context, userID string, start, end time.Time) ([]*UsageRecord, error)
}

// NewMeteringEngine creates a new metering engine
func NewMeteringEngine(config *MeteringConfig, store UsageStore) *MeteringEngine {
	me := &MeteringEngine{
		config:  config,
		buffer:  make(chan *UsageRecord, config.BufferSize),
		store:   store,
		stopped: make(chan struct{}),
	}

	// Start background workers
	go me.processBuffer()
	go me.aggregateUsage()

	return me
}

// RecordUsage records a usage event
func (me *MeteringEngine) RecordUsage(ctx context.Context, record *UsageRecord) error {
	if !me.config.Enabled {
		return nil
	}

	// Validate metric type
	if !me.isMetricEnabled(record.MetricType) {
		return fmt.Errorf("metric not enabled: %s", record.MetricType)
	}

	// Set timestamp if not provided
	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now()
	}

	// Set billing period
	record.BillingPeriod = me.getBillingPeriod(record.Timestamp)

	// Buffer the record
	select {
	case me.buffer <- record:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		// Buffer full, save directly
		return me.store.SaveUsageRecord(ctx, record)
	}
}

// processBuffer processes buffered usage records
func (me *MeteringEngine) processBuffer() {
	ticker := time.NewTicker(me.config.FlushInterval)
	defer ticker.Stop()

	batch := make([]*UsageRecord, 0, 100)

	for {
		select {
		case record := <-me.buffer:
			batch = append(batch, record)
			if len(batch) >= 100 {
				me.flushBatch(batch)
				batch = batch[:0]
			}

		case <-ticker.C:
			if len(batch) > 0 {
				me.flushBatch(batch)
				batch = batch[:0]
			}

		case <-me.stopped:
			// Flush remaining records
			for len(me.buffer) > 0 {
				batch = append(batch, <-me.buffer)
			}
			if len(batch) > 0 {
				me.flushBatch(batch)
			}
			return
		}
	}
}

// flushBatch saves a batch of records
func (me *MeteringEngine) flushBatch(batch []*UsageRecord) {
	ctx := context.Background()
	for _, record := range batch {
		if err := me.store.SaveUsageRecord(ctx, record); err != nil {
			// Log error but continue
			fmt.Printf("Error saving usage record: %v\n", err)
		}
	}
}

// aggregateUsage periodically aggregates usage data
func (me *MeteringEngine) aggregateUsage() {
	ticker := time.NewTicker(me.config.AggregationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Aggregation logic here
			// This would roll up detailed records into summaries

		case <-me.stopped:
			return
		}
	}
}

// GetUsageSummary returns usage summary for a period
func (me *MeteringEngine) GetUsageSummary(ctx context.Context, userID string, period string) (*UsageSummary, error) {
	return me.store.GetUsageSummary(ctx, userID, period)
}

// CheckQuota checks if user is within quota limits
func (me *MeteringEngine) CheckQuota(ctx context.Context, userID string, metric MetricType, amount float64, limits *UsageLimits) (*QuotaCheck, error) {
	// Get current period usage
	period := me.getCurrentPeriod()
	summary, err := me.store.GetUsageSummary(ctx, userID, period)
	if err != nil {
		return nil, err
	}

	// Get limit for metric
	limit := me.getLimitForMetric(limits, metric)
	if limit < 0 {
		// Unlimited
		return &QuotaCheck{
			Allowed:       true,
			Current:       summary.Metrics[metric],
			Limit:         -1,
			Remaining:     -1,
			PercentUsed:   0,
			WillExceed:    false,
		}, nil
	}

	current := summary.Metrics[metric]
	remaining := limit - current
	percentUsed := (current / limit) * 100
	willExceed := (current + amount) > limit

	return &QuotaCheck{
		Allowed:       !willExceed,
		Current:       current,
		Limit:         limit,
		Remaining:     remaining,
		PercentUsed:   percentUsed,
		WillExceed:    willExceed,
		EstimatedCost: me.calculateCost(metric, amount, limits),
	}, nil
}

// QuotaCheck represents quota check result
type QuotaCheck struct {
	Allowed       bool
	Current       float64
	Limit         float64
	Remaining     float64
	PercentUsed   float64
	WillExceed    bool
	EstimatedCost float64
}

// Stop stops the metering engine
func (me *MeteringEngine) Stop() {
	close(me.stopped)
}

// Helper methods

func (me *MeteringEngine) isMetricEnabled(metric MetricType) bool {
	for _, m := range me.config.Metrics {
		if m == metric {
			return true
		}
	}
	return false
}

func (me *MeteringEngine) getBillingPeriod(t time.Time) string {
	return fmt.Sprintf("%d-%02d", t.Year(), t.Month())
}

func (me *MeteringEngine) getCurrentPeriod() string {
	return me.getBillingPeriod(time.Now())
}

func (me *MeteringEngine) getLimitForMetric(limits *UsageLimits, metric MetricType) float64 {
	switch metric {
	case MetricAgentHours:
		return float64(limits.MaxAgentHours)
	case MetricDeployments:
		return float64(limits.MaxDeployments)
	case MetricAPIRequests:
		return float64(limits.MaxAPIRequests)
	case MetricStorageGB:
		return limits.MaxStorageGB
	case MetricNetworkGB:
		return limits.MaxNetworkGB
	case MetricCPUHours:
		return limits.MaxCPUHours
	case MetricMemoryGBHours:
		return limits.MaxMemoryGBHours
	case MetricGPUHours:
		return limits.MaxGPUHours
	default:
		return -1
	}
}

func (me *MeteringEngine) calculateCost(metric MetricType, amount float64, limits *UsageLimits) float64 {
	// This would integrate with pricing to calculate cost
	// Simplified for now
	rates := map[MetricType]float64{
		MetricAgentHours:  0.08,
		MetricDeployments: 0.50,
		MetricAPIRequests: 0.00005,
		MetricStorageGB:   0.10,
		MetricNetworkGB:   0.03,
		MetricGPUHours:    0.40,
	}

	if rate, ok := rates[metric]; ok {
		return amount * rate
	}
	return 0
}
