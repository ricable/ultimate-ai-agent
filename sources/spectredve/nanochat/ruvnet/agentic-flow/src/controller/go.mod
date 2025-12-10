module github.com/ruvnet/agentic-jujutsu/controller

go 1.21

require (
	github.com/go-logr/logr v1.3.0
	github.com/onsi/ginkgo/v2 v2.13.2
	github.com/onsi/gomega v1.30.0
	k8s.io/api v0.28.4
	k8s.io/apimachinery v0.28.4
	k8s.io/client-go v0.28.4
	sigs.k8s.io/controller-runtime v0.16.3
	sigs.k8s.io/yaml v1.4.0
)

require (
	github.com/prometheus/client_golang v1.17.0
	k8s.io/klog/v2 v2.110.1
)
