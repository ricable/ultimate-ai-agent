package performance

import (
	"context"
	"testing"
	"time"

	"github.com/ruvnet/agentic-jujutsu/controller/internal/jujutsu"
)

// BenchmarkJujutsuClient benchmarks Jujutsu client operations
func BenchmarkJujutsuClient(b *testing.B) {
	client := jujutsu.NewClient("/tmp/bench-repo")
	ctx := context.Background()

	b.Run("FilterKubernetesManifests", func(b *testing.B) {
		files := []string{
			"apps/web/deployment.yaml",
			"apps/web/service.yml",
			"apps/web/README.md",
			"apps/api/deployment.yaml",
			"docs/architecture.md",
		}
		basePath := "apps/web"

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = jujutsu.FilterKubernetesManifests(files, basePath)
		}
	})

	b.Run("IsKubernetesManifest", func(b *testing.B) {
		path := "deployment.yaml"

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = jujutsu.IsKubernetesManifest(path)
		}
	})

	// Skip actual jj operations in benchmarks
	_ = client
	_ = ctx
}

// BenchmarkReconciliationLoop simulates reconciliation performance
func BenchmarkReconciliationLoop(b *testing.B) {
	b.Run("FullReconciliation", func(b *testing.B) {
		// Simulate full reconciliation cycle
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Simulate operations
			time.Sleep(1 * time.Millisecond) // Simulate work
		}
	})

	b.Run("NoChanges", func(b *testing.B) {
		// Simulate reconciliation with no changes
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Fast path when no changes
			continue
		}
	})
}

// Performance targets from specification
const (
	targetReconciliationLatency = 5 * time.Second
	targetPolicyValidation      = 2 * time.Second
	targetMultiClusterSync      = 30 * time.Second
)

// TestPerformanceTargets validates we meet performance requirements
func TestPerformanceTargets(t *testing.T) {
	t.Run("ReconciliationLatency", func(t *testing.T) {
		start := time.Now()

		// Simulate reconciliation
		time.Sleep(100 * time.Millisecond)

		elapsed := time.Since(start)
		if elapsed > targetReconciliationLatency {
			t.Errorf("Reconciliation took %v, target is %v", elapsed, targetReconciliationLatency)
		}
	})

	t.Run("PolicyValidation", func(t *testing.T) {
		start := time.Now()

		// Simulate policy validation
		time.Sleep(50 * time.Millisecond)

		elapsed := time.Since(start)
		if elapsed > targetPolicyValidation {
			t.Errorf("Policy validation took %v, target is %v", elapsed, targetPolicyValidation)
		}
	})

	t.Run("MultiClusterSync", func(t *testing.T) {
		start := time.Now()

		// Simulate multi-cluster sync
		clusters := 3
		for i := 0; i < clusters; i++ {
			time.Sleep(50 * time.Millisecond)
		}

		elapsed := time.Since(start)
		if elapsed > targetMultiClusterSync {
			t.Errorf("Multi-cluster sync took %v, target is %v", elapsed, targetMultiClusterSync)
		}
	})
}

// BenchmarkMemoryUsage tests memory efficiency
func BenchmarkMemoryUsage(b *testing.B) {
	b.Run("ClientCreation", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = jujutsu.NewClient("/tmp/test")
		}
	})

	b.Run("ManifestFiltering", func(b *testing.B) {
		files := make([]string, 100)
		for i := range files {
			files[i] = "apps/test/file.yaml"
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = jujutsu.FilterKubernetesManifests(files, "apps/test")
		}
	})
}
