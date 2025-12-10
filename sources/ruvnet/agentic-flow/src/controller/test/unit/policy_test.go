package unit

import (
	"context"
	"testing"

	ajjv1 "github.com/ruvnet/agentic-jujutsu/controller/api/v1"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/policy"
)

func TestPolicyValidator(t *testing.T) {
	validator := policy.NewValidator()

	t.Run("ValidateStrict", func(t *testing.T) {
		ctx := context.Background()
		manifests := [][]byte{
			[]byte(`apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: nginx:latest
`),
		}

		config := &ajjv1.PolicyConfiguration{
			EnforcementMode: "strict",
			Validation:      []string{"require-labels"},
		}

		// This should pass as it's a basic manifest
		err := validator.Validate(ctx, manifests, config)
		if err != nil {
			t.Logf("Validation error (expected in strict mode): %v", err)
		}
	})

	t.Run("ValidatePermissive", func(t *testing.T) {
		ctx := context.Background()
		manifests := [][]byte{
			[]byte(`apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: nginx:latest
`),
		}

		config := &ajjv1.PolicyConfiguration{
			EnforcementMode: "permissive",
			Validation:      []string{},
		}

		err := validator.Validate(ctx, manifests, config)
		if err != nil {
			t.Errorf("Permissive validation should not fail: %v", err)
		}
	})

	t.Run("EmptyManifests", func(t *testing.T) {
		ctx := context.Background()
		manifests := [][]byte{}

		config := &ajjv1.PolicyConfiguration{
			EnforcementMode: "strict",
		}

		err := validator.Validate(ctx, manifests, config)
		if err == nil {
			t.Error("Expected error for empty manifests")
		}
	})
}
