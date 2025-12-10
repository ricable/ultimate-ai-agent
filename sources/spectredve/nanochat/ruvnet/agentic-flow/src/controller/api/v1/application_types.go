package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ApplicationSpec defines the desired state of Application
type ApplicationSpec struct {
	// Source defines where the application manifests come from
	Source ApplicationSource `json:"source"`

	// Destination defines where the application should be deployed
	Destination ApplicationDestination `json:"destination"`

	// SyncPolicy controls how the application is synced
	// +optional
	SyncPolicy *SyncPolicy `json:"syncPolicy,omitempty"`

	// ProgressiveDelivery defines progressive delivery settings
	// +optional
	ProgressiveDelivery *ProgressiveDelivery `json:"progressiveDelivery,omitempty"`

	// Infrastructure defines infrastructure requirements
	// +optional
	Infrastructure *InfrastructureRequirements `json:"infrastructure,omitempty"`

	// Policies defines policy enforcement configuration
	// +optional
	Policies *PolicyConfiguration `json:"policies,omitempty"`
}

// ApplicationSource defines the source of application manifests
type ApplicationSource struct {
	// RepoPath is the path within the Jujutsu repository
	RepoPath string `json:"repoPath"`

	// TargetRevision is the Jujutsu change ID or bookmark to sync to
	TargetRevision string `json:"targetRevision"`

	// Path is the path within RepoPath to the manifests
	// +optional
	Path string `json:"path,omitempty"`
}

// ApplicationDestination defines deployment targets
type ApplicationDestination struct {
	// Clusters is the list of target clusters
	Clusters []string `json:"clusters"`
}

// SyncPolicy controls synchronization behavior
type SyncPolicy struct {
	// Automated enables automatic synchronization
	// +optional
	Automated *AutomatedSync `json:"automated,omitempty"`

	// Retry configures retry behavior
	// +optional
	Retry *RetryPolicy `json:"retry,omitempty"`

	// SyncOptions provides additional sync options
	// +optional
	SyncOptions []string `json:"syncOptions,omitempty"`
}

// AutomatedSync configures automatic syncing
type AutomatedSync struct {
	// Prune enables automatic pruning of resources
	Prune bool `json:"prune,omitempty"`

	// SelfHeal enables automatic correction of drift
	SelfHeal bool `json:"selfHeal,omitempty"`

	// AllowEmpty allows syncing when no resources are found
	AllowEmpty bool `json:"allowEmpty,omitempty"`
}

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	// Limit is the maximum number of retry attempts
	Limit int64 `json:"limit,omitempty"`

	// Backoff configures exponential backoff
	// +optional
	Backoff *Backoff `json:"backoff,omitempty"`
}

// Backoff configures exponential backoff
type Backoff struct {
	// Duration is the initial duration
	Duration string `json:"duration,omitempty"`

	// Factor is the exponential backoff factor
	Factor int64 `json:"factor,omitempty"`

	// MaxDuration is the maximum backoff duration
	MaxDuration string `json:"maxDuration,omitempty"`
}

// ProgressiveDelivery defines progressive delivery configuration
type ProgressiveDelivery struct {
	// Enabled enables progressive delivery
	Enabled bool `json:"enabled"`

	// Strategy defines the rollout strategy (canary, blue-green, ab-test)
	Strategy string `json:"strategy"`

	// Steps defines the rollout steps
	// +optional
	Steps []RolloutStep `json:"steps,omitempty"`

	// TrafficRouting defines traffic routing configuration
	// +optional
	TrafficRouting *TrafficRouting `json:"trafficRouting,omitempty"`

	// AutoRollback configures automatic rollback
	// +optional
	AutoRollback *AutoRollback `json:"autoRollback,omitempty"`
}

// RolloutStep defines a single rollout step
type RolloutStep struct {
	// SetWeight sets the canary weight
	// +optional
	SetWeight *int32 `json:"setWeight,omitempty"`

	// Pause pauses the rollout
	// +optional
	Pause *RolloutPause `json:"pause,omitempty"`

	// Analysis runs analysis
	// +optional
	Analysis *RolloutAnalysis `json:"analysis,omitempty"`
}

// RolloutPause defines a pause in the rollout
type RolloutPause struct {
	// Duration is the pause duration
	Duration string `json:"duration,omitempty"`
}

// RolloutAnalysis defines analysis configuration
type RolloutAnalysis struct {
	// Templates is the list of analysis template names
	Templates []AnalysisTemplateRef `json:"templates,omitempty"`
}

// AnalysisTemplateRef references an analysis template
type AnalysisTemplateRef struct {
	// TemplateName is the name of the analysis template
	TemplateName string `json:"templateName"`

	// ClusterScope indicates if the template is cluster-scoped
	// +optional
	ClusterScope bool `json:"clusterScope,omitempty"`
}

// TrafficRouting defines traffic routing configuration
type TrafficRouting struct {
	// ServiceMesh defines the service mesh to use (istio, linkerd)
	ServiceMesh string `json:"serviceMesh"`

	// VirtualService defines the virtual service configuration
	// +optional
	VirtualService *VirtualServiceConfig `json:"virtualService,omitempty"`

	// DestinationRule defines the destination rule configuration
	// +optional
	DestinationRule *DestinationRuleConfig `json:"destinationRule,omitempty"`
}

// VirtualServiceConfig defines virtual service configuration
type VirtualServiceConfig struct {
	// Name is the virtual service name
	Name string `json:"name"`

	// Routes is the list of route names
	// +optional
	Routes []string `json:"routes,omitempty"`
}

// DestinationRuleConfig defines destination rule configuration
type DestinationRuleConfig struct {
	// Name is the destination rule name
	Name string `json:"name"`

	// CanarySubsetName is the canary subset name
	CanarySubsetName string `json:"canarySubsetName"`

	// StableSubsetName is the stable subset name
	StableSubsetName string `json:"stableSubsetName"`
}

// AutoRollback defines automatic rollback configuration
type AutoRollback struct {
	// Enabled enables automatic rollback
	Enabled bool `json:"enabled"`

	// OnFailure enables rollback on deployment failure
	// +optional
	OnFailure bool `json:"onFailure,omitempty"`

	// OnPolicyViolation enables rollback on policy violation
	// +optional
	OnPolicyViolation bool `json:"onPolicyViolation,omitempty"`
}

// InfrastructureRequirements defines infrastructure requirements
type InfrastructureRequirements struct {
	// Required is the list of required infrastructure components
	Required []InfrastructureComponent `json:"required,omitempty"`
}

// InfrastructureComponent defines a single infrastructure component
type InfrastructureComponent struct {
	// Name is the component name
	Name string `json:"name"`

	// Type is the component type (database, cache, storage, queue)
	Type string `json:"type"`

	// Composition is the Crossplane composition name
	Composition string `json:"composition"`

	// Parameters are component-specific parameters
	// +optional
	Parameters map[string]string `json:"parameters,omitempty"`

	// ConnectionSecret defines where to store connection details
	// +optional
	ConnectionSecret *ConnectionSecretRef `json:"connectionSecret,omitempty"`
}

// ConnectionSecretRef references a connection secret
type ConnectionSecretRef struct {
	// Name is the secret name
	Name string `json:"name"`

	// Namespace is the secret namespace
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// PolicyConfiguration defines policy enforcement configuration
type PolicyConfiguration struct {
	// Validation is the list of validation policies
	// +optional
	Validation []string `json:"validation,omitempty"`

	// EnforcementMode defines how policies are enforced (strict, permissive, audit)
	EnforcementMode string `json:"enforcementMode,omitempty"`
}

// ApplicationStatus defines the observed state of Application
type ApplicationStatus struct {
	// Phase is the current application phase
	Phase ApplicationPhase `json:"phase,omitempty"`

	// SyncStatus contains sync status information
	// +optional
	SyncStatus *SyncStatus `json:"syncStatus,omitempty"`

	// Health contains health status information
	// +optional
	Health *HealthStatus `json:"health,omitempty"`

	// RolloutStatus contains rollout status information
	// +optional
	RolloutStatus *RolloutStatus `json:"rolloutStatus,omitempty"`

	// Conditions represent the latest available observations
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration is the generation observed by the controller
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// ApplicationPhase represents the current phase
type ApplicationPhase string

const (
	ApplicationPhasePending   ApplicationPhase = "Pending"
	ApplicationPhaseSyncing   ApplicationPhase = "Syncing"
	ApplicationPhaseHealthy   ApplicationPhase = "Healthy"
	ApplicationPhaseDegraded  ApplicationPhase = "Degraded"
	ApplicationPhaseFailed    ApplicationPhase = "Failed"
	ApplicationPhaseDeleting  ApplicationPhase = "Deleting"
)

// SyncStatus contains sync status information
type SyncStatus struct {
	// Status is the sync status
	Status SyncStatusCode `json:"status"`

	// Revision is the current Jujutsu change ID
	// +optional
	Revision string `json:"revision,omitempty"`

	// LastSyncTime is the last sync time
	// +optional
	LastSyncTime *metav1.Time `json:"lastSyncTime,omitempty"`

	// Message provides additional information
	// +optional
	Message string `json:"message,omitempty"`
}

// SyncStatusCode represents sync status
type SyncStatusCode string

const (
	SyncStatusCodeSynced    SyncStatusCode = "Synced"
	SyncStatusCodeOutOfSync SyncStatusCode = "OutOfSync"
	SyncStatusCodeUnknown   SyncStatusCode = "Unknown"
)

// HealthStatus contains health status information
type HealthStatus struct {
	// Status is the health status
	Status HealthStatusCode `json:"status"`

	// Message provides additional information
	// +optional
	Message string `json:"message,omitempty"`
}

// HealthStatusCode represents health status
type HealthStatusCode string

const (
	HealthStatusCodeHealthy    HealthStatusCode = "Healthy"
	HealthStatusCodeProgressing HealthStatusCode = "Progressing"
	HealthStatusCodeDegraded   HealthStatusCode = "Degraded"
	HealthStatusCodeSuspended  HealthStatusCode = "Suspended"
	HealthStatusCodeUnknown    HealthStatusCode = "Unknown"
)

// RolloutStatus contains rollout status information
type RolloutStatus struct {
	// Phase is the rollout phase
	Phase RolloutPhase `json:"phase"`

	// CanaryWeight is the current canary weight
	// +optional
	CanaryWeight int32 `json:"canaryWeight,omitempty"`

	// Message provides additional information
	// +optional
	Message string `json:"message,omitempty"`

	// CurrentStepIndex is the current step index
	// +optional
	CurrentStepIndex int32 `json:"currentStepIndex,omitempty"`
}

// RolloutPhase represents rollout phase
type RolloutPhase string

const (
	RolloutPhaseInitializing RolloutPhase = "Initializing"
	RolloutPhasePromoting    RolloutPhase = "Promoting"
	RolloutPhasePaused       RolloutPhase = "Paused"
	RolloutPhaseStable       RolloutPhase = "Stable"
	RolloutPhaseAborted      RolloutPhase = "Aborted"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=app;apps
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Sync",type=string,JSONPath=`.status.syncStatus.status`
// +kubebuilder:printcolumn:name="Health",type=string,JSONPath=`.status.health.status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// Application is the Schema for the applications API
type Application struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ApplicationSpec   `json:"spec,omitempty"`
	Status ApplicationStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ApplicationList contains a list of Application
type ApplicationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Application `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Application{}, &ApplicationList{})
}
