package unit

import (
	"context"
	"testing"

	"github.com/ruvnet/agentic-jujutsu/controller/internal/jujutsu"
)

func TestJujutsuClient(t *testing.T) {
	t.Run("FilterKubernetesManifests", func(t *testing.T) {
		files := []string{
			"apps/web/deployment.yaml",
			"apps/web/service.yml",
			"apps/web/README.md",
			"apps/api/deployment.yaml",
			"docs/architecture.md",
		}

		basePath := "apps/web"
		manifests := jujutsu.FilterKubernetesManifests(files, basePath)

		expected := 2
		if len(manifests) != expected {
			t.Errorf("Expected %d manifests, got %d", expected, len(manifests))
		}

		// Check that only web manifests are included
		for _, m := range manifests {
			if m != "apps/web/deployment.yaml" && m != "apps/web/service.yml" {
				t.Errorf("Unexpected manifest in results: %s", m)
			}
		}
	})

	t.Run("IsKubernetesManifest", func(t *testing.T) {
		testCases := []struct {
			path     string
			expected bool
		}{
			{"deployment.yaml", true},
			{"service.yml", true},
			{"configmap.YAML", true},
			{"README.md", false},
			{"main.go", false},
			{"", false},
		}

		for _, tc := range testCases {
			result := jujutsu.IsKubernetesManifest(tc.path)
			if result != tc.expected {
				t.Errorf("IsKubernetesManifest(%s) = %v, expected %v", tc.path, result, tc.expected)
			}
		}
	})
}

func TestChangeFiltering(t *testing.T) {
	ctx := context.Background()
	_ = ctx // Use context in actual tests

	// Note: These are unit tests, integration tests would use actual jj commands
	t.Run("NewClient", func(t *testing.T) {
		client := jujutsu.NewClient("/tmp/test-repo")
		if client == nil {
			t.Error("Expected non-nil client")
		}
	})
}
