package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	ajjv1 "github.com/ruvnet/agentic-jujutsu/controller/api/v1"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/cluster"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/controller"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/jujutsu"
	"github.com/ruvnet/agentic-jujutsu/controller/internal/policy"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(ajjv1.AddToScheme(scheme))
}

func main() {
	var (
		metricsAddr          string
		enableLeaderElection bool
		probeAddr            string
		repoPath             string
		reconcileInterval    string
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&repoPath, "repo-path", "/tmp/gitops-repo", "Path to Jujutsu repository")
	flag.StringVar(&reconcileInterval, "reconcile-interval", "30s", "Reconciliation interval")

	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
		Metrics: metricsserver.Options{
			BindAddress: metricsAddr,
		},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "ajj-controller-leader-election",
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// Initialize clients
	jujutsuClient := jujutsu.NewClient(repoPath)
	policyValidator := policy.NewValidator()
	clusterManager := cluster.NewManager(mgr.GetClient())

	// Setup Application controller
	if err = (&controller.ApplicationReconciler{
		Client:          mgr.GetClient(),
		Scheme:          mgr.GetScheme(),
		JujutsuClient:   jujutsuClient,
		PolicyValidator: policyValidator,
		ClusterManager:  clusterManager,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Application")
		os.Exit(1)
	}

	// Add health and ready checks
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
