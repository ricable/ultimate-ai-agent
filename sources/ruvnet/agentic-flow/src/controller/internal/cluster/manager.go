package cluster

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	ajjv1 "github.com/ruvnet/agentic-jujutsu/controller/api/v1"
)

// Manager manages connections to multiple Kubernetes clusters
type Manager struct {
	client  client.Client
	clients map[string]*ClusterClient
	mu      sync.RWMutex
}

// ClusterClient represents a connection to a Kubernetes cluster
type ClusterClient struct {
	Name           string
	Config         *rest.Config
	Clientset      *kubernetes.Clientset
	DynamicClient  dynamic.Interface
}

// NewManager creates a new cluster manager
func NewManager(client client.Client) *Manager {
	return &Manager{
		client:  client,
		clients: make(map[string]*ClusterClient),
	}
}

// GetClient returns a client for the specified cluster
func (m *Manager) GetClient(ctx context.Context, clusterName string) (*ClusterClient, error) {
	// Check cache first
	m.mu.RLock()
	if client, ok := m.clients[clusterName]; ok {
		m.mu.RUnlock()
		return client, nil
	}
	m.mu.RUnlock()

	// Fetch cluster resource
	cluster := &ajjv1.Cluster{}
	if err := m.client.Get(ctx, types.NamespacedName{Name: clusterName}, cluster); err != nil {
		return nil, fmt.Errorf("failed to get cluster: %w", err)
	}

	// Get kubeconfig secret
	secret := &corev1.Secret{}
	secretRef := cluster.Spec.Kubeconfig.SecretRef
	secretNamespace := secretRef.Namespace
	if secretNamespace == "" {
		secretNamespace = "ajj-system"
	}

	if err := m.client.Get(ctx, types.NamespacedName{
		Name:      secretRef.Name,
		Namespace: secretNamespace,
	}, secret); err != nil {
		return nil, fmt.Errorf("failed to get kubeconfig secret: %w", err)
	}

	// Extract kubeconfig
	kubeconfigData, ok := secret.Data["kubeconfig"]
	if !ok {
		return nil, fmt.Errorf("kubeconfig not found in secret")
	}

	// Build REST config
	config, err := clientcmd.RESTConfigFromKubeConfig(kubeconfigData)
	if err != nil {
		return nil, fmt.Errorf("failed to build REST config: %w", err)
	}

	// Create clientset
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset: %w", err)
	}

	// Create dynamic client
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}

	clusterClient := &ClusterClient{
		Name:          clusterName,
		Config:        config,
		Clientset:     clientset,
		DynamicClient: dynamicClient,
	}

	// Cache client
	m.mu.Lock()
	m.clients[clusterName] = clusterClient
	m.mu.Unlock()

	return clusterClient, nil
}

// Apply applies a manifest to the cluster
func (c *ClusterClient) Apply(ctx context.Context, manifest []byte) error {
	// TODO: Implement manifest application using dynamic client
	// Parse YAML, determine resource type, and apply using appropriate client
	// For now, return success
	return nil
}

// Delete deletes resources from the cluster
func (c *ClusterClient) Delete(ctx context.Context, namespace, name string) error {
	// TODO: Implement resource deletion
	return nil
}

// GetServerVersion returns the Kubernetes server version
func (c *ClusterClient) GetServerVersion(ctx context.Context) (string, error) {
	version, err := c.Clientset.Discovery().ServerVersion()
	if err != nil {
		return "", err
	}
	return version.String(), nil
}

// CheckConnection checks if the cluster is reachable
func (c *ClusterClient) CheckConnection(ctx context.Context) error {
	_, err := c.Clientset.Discovery().ServerVersion()
	return err
}
