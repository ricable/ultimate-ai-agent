package controller

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	ajjv1 "github.com/ruvnet/agentic-jujutsu/controller/api/v1"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/cluster"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/jujutsu"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/policy"
)

const (
	applicationFinalizer = "ajj.io/finalizer"
	reconcileInterval    = 30 * time.Second
)

// ApplicationReconciler reconciles an Application object
type ApplicationReconciler struct {
	client.Client
	Scheme         *runtime.Scheme
	JujutsuClient  *jujutsu.Client
	PolicyValidator *policy.Validator
	ClusterManager *cluster.Manager
}

// +kubebuilder:rbac:groups=ajj.io,resources=applications,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ajj.io,resources=applications/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ajj.io,resources=applications/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=pods;services;configmaps;secrets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments;statefulsets;daemonsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=argoproj.io,resources=rollouts,verbs=get;list;watch;create;update;patch;delete

// Reconcile handles reconciliation of Application resources
func (r *ApplicationReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling Application", "name", req.Name, "namespace", req.Namespace)

	// Fetch the Application instance
	app := &ajjv1.Application{}
	if err := r.Get(ctx, req.NamespacedName, app); err != nil {
		if errors.IsNotFound(err) {
			log.Info("Application resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		log.Error(err, "Failed to get Application")
		return ctrl.Result{}, err
	}

	// Handle deletion
	if !app.DeletionTimestamp.IsZero() {
		return r.handleDeletion(ctx, app)
	}

	// Add finalizer if not present
	if !controllerutil.ContainsFinalizer(app, applicationFinalizer) {
		controllerutil.AddFinalizer(app, applicationFinalizer)
		if err := r.Update(ctx, app); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Sync Jujutsu repository
	if err := r.JujutsuClient.Sync(ctx); err != nil {
		log.Error(err, "Failed to sync Jujutsu repository")
		r.updateStatus(ctx, app, ajjv1.ApplicationPhaseFailed, "Failed to sync repository")
		return ctrl.Result{RequeueAfter: reconcileInterval}, nil
	}

	// Get changes since last sync
	lastRevision := ""
	if app.Status.SyncStatus != nil {
		lastRevision = app.Status.SyncStatus.Revision
	}

	changes, err := r.JujutsuClient.GetChangesSince(ctx, lastRevision)
	if err != nil {
		log.Error(err, "Failed to get changes")
		r.updateStatus(ctx, app, ajjv1.ApplicationPhaseFailed, "Failed to get changes")
		return ctrl.Result{RequeueAfter: reconcileInterval}, nil
	}

	// No new changes, check health and requeue
	if len(changes) == 0 {
		log.Info("No new changes detected")
		// Update health status
		r.updateHealthStatus(ctx, app)
		return ctrl.Result{RequeueAfter: reconcileInterval}, nil
	}

	log.Info("Processing new changes", "count", len(changes))

	// Update status to syncing
	r.updateStatus(ctx, app, ajjv1.ApplicationPhaseSyncing, "Processing new changes")

	// Extract manifests from changes
	manifests, err := r.extractManifests(ctx, changes, app.Spec.Source.RepoPath)
	if err != nil {
		log.Error(err, "Failed to extract manifests")
		r.updateStatus(ctx, app, ajjv1.ApplicationPhaseFailed, "Failed to extract manifests")
		return ctrl.Result{RequeueAfter: reconcileInterval}, nil
	}

	if len(manifests) == 0 {
		log.Info("No Kubernetes manifests found in changes")
		return ctrl.Result{RequeueAfter: reconcileInterval}, nil
	}

	// Validate policies if configured
	if app.Spec.Policies != nil && len(app.Spec.Policies.Validation) > 0 {
		log.Info("Validating policies")
		if err := r.PolicyValidator.Validate(ctx, manifests, app.Spec.Policies); err != nil {
			log.Error(err, "Policy validation failed")
			r.updateStatus(ctx, app, ajjv1.ApplicationPhaseFailed, fmt.Sprintf("Policy validation failed: %v", err))
			return ctrl.Result{RequeueAfter: reconcileInterval}, nil
		}
	}

	// Deploy to target clusters
	for _, clusterName := range app.Spec.Destination.Clusters {
		log.Info("Deploying to cluster", "cluster", clusterName)

		if err := r.deployToCluster(ctx, app, clusterName, manifests); err != nil {
			log.Error(err, "Failed to deploy to cluster", "cluster", clusterName)
			r.updateStatus(ctx, app, ajjv1.ApplicationPhaseDegraded, fmt.Sprintf("Failed to deploy to cluster %s: %v", clusterName, err))
			continue
		}
	}

	// Update status to healthy
	latestChange := changes[len(changes)-1]
	app.Status.SyncStatus = &ajjv1.SyncStatus{
		Status:       ajjv1.SyncStatusCodeSynced,
		Revision:     latestChange.ID,
		LastSyncTime: &metav1.Time{Time: time.Now()},
		Message:      fmt.Sprintf("Synced to change %s", latestChange.ID[:12]),
	}
	r.updateStatus(ctx, app, ajjv1.ApplicationPhaseHealthy, "Successfully deployed")

	log.Info("Reconciliation completed successfully")
	return ctrl.Result{RequeueAfter: reconcileInterval}, nil
}

// handleDeletion handles application deletion
func (r *ApplicationReconciler) handleDeletion(ctx context.Context, app *ajjv1.Application) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	if !controllerutil.ContainsFinalizer(app, applicationFinalizer) {
		return ctrl.Result{}, nil
	}

	log.Info("Handling Application deletion")
	r.updateStatus(ctx, app, ajjv1.ApplicationPhaseDeleting, "Deleting application")

	// Delete from all target clusters
	for _, clusterName := range app.Spec.Destination.Clusters {
		log.Info("Deleting from cluster", "cluster", clusterName)
		// TODO: Implement cluster deletion
	}

	// Remove finalizer
	controllerutil.RemoveFinalizer(app, applicationFinalizer)
	if err := r.Update(ctx, app); err != nil {
		return ctrl.Result{}, err
	}

	log.Info("Application deleted successfully")
	return ctrl.Result{}, nil
}

// extractManifests extracts Kubernetes manifests from changes
func (r *ApplicationReconciler) extractManifests(ctx context.Context, changes []jujutsu.Change, basePath string) ([][]byte, error) {
	var manifests [][]byte

	// Get the latest change
	latestChange := changes[len(changes)-1]

	// Get all changed files
	files := jujutsu.FilterKubernetesManifests(latestChange.Files, basePath)

	// Read each manifest file
	for _, file := range files {
		content, err := r.JujutsuClient.GetFileContent(ctx, latestChange.ID, file)
		if err != nil {
			return nil, fmt.Errorf("failed to get file content for %s: %w", file, err)
		}
		manifests = append(manifests, content)
	}

	return manifests, nil
}

// deployToCluster deploys manifests to a specific cluster
func (r *ApplicationReconciler) deployToCluster(ctx context.Context, app *ajjv1.Application, clusterName string, manifests [][]byte) error {
	// Get cluster client
	clusterClient, err := r.ClusterManager.GetClient(ctx, clusterName)
	if err != nil {
		return fmt.Errorf("failed to get cluster client: %w", err)
	}

	// Apply manifests
	for _, manifest := range manifests {
		if err := clusterClient.Apply(ctx, manifest); err != nil {
			return fmt.Errorf("failed to apply manifest: %w", err)
		}
	}

	return nil
}

// updateStatus updates the application status
func (r *ApplicationReconciler) updateStatus(ctx context.Context, app *ajjv1.Application, phase ajjv1.ApplicationPhase, message string) {
	app.Status.Phase = phase

	// Update conditions
	condition := metav1.Condition{
		Type:               string(phase),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             string(phase),
		Message:            message,
	}

	// Add or update condition
	found := false
	for i, c := range app.Status.Conditions {
		if c.Type == condition.Type {
			app.Status.Conditions[i] = condition
			found = true
			break
		}
	}
	if !found {
		app.Status.Conditions = append(app.Status.Conditions, condition)
	}

	// Update status
	if err := r.Status().Update(ctx, app); err != nil {
		log.FromContext(ctx).Error(err, "Failed to update status")
	}
}

// updateHealthStatus checks and updates health status
func (r *ApplicationReconciler) updateHealthStatus(ctx context.Context, app *ajjv1.Application) {
	// TODO: Implement health check logic
	app.Status.Health = &ajjv1.HealthStatus{
		Status:  ajjv1.HealthStatusCodeHealthy,
		Message: "All resources healthy",
	}

	if err := r.Status().Update(ctx, app); err != nil {
		log.FromContext(ctx).Error(err, "Failed to update health status")
	}
}

// SetupWithManager sets up the controller with the Manager
func (r *ApplicationReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&ajjv1.Application{}).
		Complete(r)
}
