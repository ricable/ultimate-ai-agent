package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ClusterSpec defines the desired state of Cluster
type ClusterSpec struct {
	// Kubeconfig references the kubeconfig secret
	Kubeconfig KubeconfigReference `json:"kubeconfig"`

	// Environment is the cluster environment (development, staging, production)
	// +optional
	Environment string `json:"environment,omitempty"`

	// Region is the cluster region
	// +optional
	Region string `json:"region,omitempty"`

	// Sync configures synchronization settings
	// +optional
	Sync *ClusterSync `json:"sync,omitempty"`

	// Labels are additional labels for the cluster
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
}

// KubeconfigReference references a kubeconfig secret
type KubeconfigReference struct {
	// SecretRef references the secret containing the kubeconfig
	SecretRef SecretReference `json:"secretRef"`
}

// SecretReference references a secret
type SecretReference struct {
	// Name is the secret name
	Name string `json:"name"`

	// Namespace is the secret namespace
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// ClusterSync defines synchronization configuration
type ClusterSync struct {
	// Interval is the sync interval
	// +optional
	Interval string `json:"interval,omitempty"`

	// Timeout is the sync timeout
	// +optional
	Timeout string `json:"timeout,omitempty"`

	// Retries is the number of retries
	// +optional
	Retries int32 `json:"retries,omitempty"`
}

// ClusterStatus defines the observed state of Cluster
type ClusterStatus struct {
	// Phase is the cluster phase
	Phase ClusterPhase `json:"phase,omitempty"`

	// ConnectionStatus contains connection status
	// +optional
	ConnectionStatus *ConnectionStatus `json:"connectionStatus,omitempty"`

	// ServerVersion is the Kubernetes server version
	// +optional
	ServerVersion string `json:"serverVersion,omitempty"`

	// Conditions represent the latest available observations
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// ClusterPhase represents cluster phase
type ClusterPhase string

const (
	ClusterPhaseConnected    ClusterPhase = "Connected"
	ClusterPhaseDisconnected ClusterPhase = "Disconnected"
	ClusterPhaseUnknown      ClusterPhase = "Unknown"
)

// ConnectionStatus contains connection status information
type ConnectionStatus struct {
	// Connected indicates if the cluster is reachable
	Connected bool `json:"connected"`

	// LastChecked is the last time the connection was checked
	// +optional
	LastChecked *metav1.Time `json:"lastChecked,omitempty"`

	// Message provides additional information
	// +optional
	Message string `json:"message,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster,shortName=cls
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Environment",type=string,JSONPath=`.spec.environment`
// +kubebuilder:printcolumn:name="Region",type=string,JSONPath=`.spec.region`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// Cluster is the Schema for the clusters API
type Cluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ClusterSpec   `json:"spec,omitempty"`
	Status ClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ClusterList contains a list of Cluster
type ClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Cluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Cluster{}, &ClusterList{})
}
