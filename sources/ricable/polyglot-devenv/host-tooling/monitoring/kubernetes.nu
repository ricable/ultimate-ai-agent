#!/usr/bin/env nu

# Kubernetes management utilities for the polyglot development environment
# Adapted from the sophisticated DevOps scripts in nushell.md
# Usage: nu scripts/kubernetes.nu <command> [options]

use ../common.nu *

# Main kubernetes command dispatcher
def main [command?: string] {
    if ($command | is-empty) {
        show-help
    } else {
        match $command {
            "status" => { main status }
            "setup" => { main setup }
            "deploy" => { main deploy }
            "logs" => { main logs }
            "port-forward" => { main port-forward }
            "scale" => { main scale }
            "restart" => { main restart }
            "cleanup" => { main cleanup }
            _ => {
                log error $"Unknown command: ($command)"
                show-help
                exit 1
            }
        }
    }
}

def show-help [] {
    log info "Kubernetes utilities for polyglot development environment"
    print ""
    print "Commands:"
    print "  status        - Show cluster and deployment status"
    print "  setup         - Setup development namespace and resources"
    print "  deploy        - Deploy applications from environment configs"
    print "  logs          - View application logs"
    print "  port-forward  - Setup port forwarding for local development"
    print "  scale         - Scale deployments"
    print "  restart       - Restart deployments"
    print "  cleanup       - Clean up development resources"
    print ""
    print "Examples:"
    print "  nu scripts/kubernetes.nu status"
    print "  nu scripts/kubernetes.nu setup --namespace dev"
    print "  nu scripts/kubernetes.nu deploy --app python-app"
}

# Check cluster status and connection
def "main status" [
    --namespace: string = "default"
] {
    log info "Checking Kubernetes cluster status..."
    
    # Check kubectl availability
    if not (cmd exists "kubectl") {
        log error "kubectl not found. Please install kubectl."
        exit 1
    }
    
    # Check cluster connection
    try {
        let cluster_info = kubectl cluster-info | str trim
        log success "✅ Connected to cluster"
        log info $"Cluster info: ($cluster_info | lines | first)"
    } catch {
        log error "❌ Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    # Check namespace
    try {
        kubectl get namespace $namespace | ignore
        log success $"✅ Namespace '($namespace)' exists"
    } catch {
        log warn $"⚠️  Namespace '($namespace)' not found"
    }
    
    # Show resource summary
    show-resource-summary $namespace
}

def show-resource-summary [namespace: string] {
    log info $"Resource summary for namespace '($namespace)':"
    
    let resources = ["pods", "services", "deployments", "configmaps", "secrets"]
    
    for resource in $resources {
        try {
            let count = kubectl get $resource --namespace $namespace --no-headers | lines | length
            log info $"  ($resource): ($count)"
        } catch {
            log info $"  ($resource): 0"
        }
    }
}

# Setup development namespace and common resources
def "main setup" [
    --namespace: string = "polyglot-dev"
    --create-secrets = true
] {
    log info $"Setting up development environment in namespace '($namespace)'..."
    
    # Create namespace
    try {
        kubectl create namespace $namespace
        log success $"✅ Created namespace '($namespace)'"
    } catch {
        log info $"Namespace '($namespace)' already exists"
    }
    
    # Setup common resources
    if $create_secrets {
        setup-secrets $namespace
    }
    
    setup-configmaps $namespace
    
    log success "Development environment setup completed!"
}

def setup-secrets [namespace: string] {
    log info $"Setting up secrets in namespace '($namespace)'..."
    
    # Check if .env file exists for secrets
    if (".env" | path exists) {
        let env_vars = open .env | lines | where ($it | str contains "=") and not ($it | str starts-with "#")
        
        if ($env_vars | length) > 0 {
            # Create a generic secret from .env file
            kubectl create secret generic app-secrets --namespace $namespace --from-env-file .env | ignore
            log success "✅ Created app-secrets from .env file"
        }
    }
    
    # Setup registry credentials if available
    if (env get-or-prompt "REGISTRY_USER" "Container registry username (optional)" | is-not-empty) {
        let registry_user = $env.REGISTRY_USER
        let registry_password = env get-or-prompt "REGISTRY_PASSWORD" "Container registry password" --secret
        let registry_email = env get-or-prompt "REGISTRY_EMAIL" "Container registry email"
        let registry_server = env get-or-prompt "REGISTRY_SERVER" "Container registry server" | default "ghcr.io"
        
        try {
            kubectl create secret docker-registry regcred 
                --namespace $namespace
                $"--docker-server=($registry_server)"
                $"--docker-username=($registry_user)"
                $"--docker-password=($registry_password)" 
                $"--docker-email=($registry_email)"
            log success "✅ Created registry credentials"
        } catch {
            log warn "Registry credentials already exist or failed to create"
        }
    }
}

def setup-configmaps [namespace: string] {
    log info $"Setting up ConfigMaps in namespace '($namespace)'..."
    
    # Create a basic config map for the polyglot environment
    let config_data = {
        "python_version": "3.12"
        "node_version": "20"
        "rust_version": "1.70"
        "go_version": "1.22"
        "environment": "development"
    }
    
    # Write config to temporary file
    $config_data | to json | save tmp/app-config.json --force
    
    try {
        kubectl create configmap app-config --namespace $namespace --from-file tmp/app-config.json
        log success "✅ Created app-config ConfigMap"
    } catch {
        log warn "ConfigMap already exists or failed to create"
    }
    
    # Clean up temp file
    rm tmp/app-config.json
}

# Deploy applications based on environment
def "main deploy" [
    --app: string = "all"
    --namespace: string = "polyglot-dev"
    --dry-run = false
] {
    log info $"Deploying applications to namespace '($namespace)'..."
    
    let environments = get-deployable-environments
    
    for env in $environments {
        if $app == "all" or $app == $env.name {
            deploy-environment $env $namespace $dry_run
        }
    }
}

def get-deployable-environments [] {
    [
        {name: "python", dir: "python-env", port: 8000},
        {name: "typescript", dir: "typescript-env", port: 3000},
        {name: "rust", dir: "rust-env", port: 8080},
        {name: "go", dir: "go-env", port: 8090}
    ]
}

def deploy-environment [env: record, namespace: string, dry_run: bool] {
    if not ($env.dir | path exists) {
        log warn $"Environment directory not found: ($env.dir)"
        return
    }
    
    log info $"Deploying ($env.name) application..."
    
    # Generate basic Kubernetes manifests
    let manifests = generate-k8s-manifests $env $namespace
    
    if $dry_run {
        log info $"Dry run - would deploy ($env.name) with the following manifests:"
        $manifests | each { |manifest| print $manifest.content }
    } else {
        # Apply manifests
        for manifest in $manifests {
            $manifest.content | save $"tmp/($manifest.name)" --force
            
            try {
                kubectl apply --filename $"tmp/($manifest.name)"
                log success $"✅ Applied ($manifest.name)"
            } catch { |e|
                log error $"❌ Failed to apply ($manifest.name): ($e.msg)"
            }
            
            rm $"tmp/($manifest.name)"
        }
    }
}

def generate-k8s-manifests [env: record, namespace: string] {
    let app_name = $"($env.name)-app"
    
    [
        {
            name: $"($app_name)-deployment.yaml"
            content: generate-deployment-manifest $env $namespace
        }
        {
            name: $"($app_name)-service.yaml" 
            content: generate-service-manifest $env $namespace
        }
    ]
}

def generate-deployment-manifest [env: record, namespace: string] {
    let app_name = $"($env.name)-app"
    
    {
        apiVersion: "apps/v1"
        kind: "Deployment"
        metadata: {
            name: $app_name
            namespace: $namespace
        }
        spec: {
            replicas: 1
            selector: {
                matchLabels: {
                    app: $app_name
                }
            }
            template: {
                metadata: {
                    labels: {
                        app: $app_name
                    }
                }
                spec: {
                    containers: [{
                        name: $app_name
                        image: $"($app_name):latest"
                        ports: [{
                            containerPort: $env.port
                        }]
                        envFrom: [{
                            secretRef: {
                                name: "app-secrets"
                            }
                        }]
                    }]
                }
            }
        }
    } | to yaml
}

def generate-service-manifest [env: record, namespace: string] {
    let app_name = $"($env.name)-app"
    
    {
        apiVersion: "v1"
        kind: "Service"
        metadata: {
            name: $app_name
            namespace: $namespace
        }
        spec: {
            selector: {
                app: $app_name
            }
            ports: [{
                port: $env.port
                targetPort: $env.port
            }]
            type: "ClusterIP"
        }
    } | to yaml
}

# View logs from applications
def "main logs" [
    --app: string = ""
    --namespace: string = "polyglot-dev"
    --follow = false
] {
    if ($app | is-empty) {
        log error "Please specify an app with --app"
        exit 1
    }
    
    let app_name = $"($app)-app"
    
    log info $"Viewing logs for ($app_name) in namespace ($namespace)..."
    
    let follow_flag = if $follow { "--follow" } else { "" }
    
    kubectl logs $"deployment/($app_name)" --namespace $namespace $follow_flag
}

# Setup port forwarding for local development
def "main port-forward" [
    --app: string = ""
    --namespace: string = "polyglot-dev"
    --local-port: int = 0
] {
    if ($app | is-empty) {
        log error "Please specify an app with --app"
        exit 1
    }
    
    let environments = get-deployable-environments
    let env = $environments | where name == $app | first
    
    if ($env | is-empty) {
        log error $"Unknown app: ($app)"
        exit 1
    }
    
    let target_port = if $local_port == 0 { $env.port } else { $local_port }
    let app_name = $"($app)-app"
    
    log info $"Setting up port forwarding: localhost:($target_port) -> ($app_name):($env.port)"
    
    kubectl port-forward $"deployment/($app_name)" --namespace $namespace $"($target_port):($env.port)"
}

# Scale deployments
def "main scale" [
    --app: string = ""
    --replicas: int = 1
    --namespace: string = "polyglot-dev"
] {
    if ($app | is-empty) {
        log error "Please specify an app with --app"
        exit 1
    }
    
    let app_name = $"($app)-app"
    
    log info $"Scaling ($app_name) to ($replicas) replicas..."
    
    kubectl scale deployment $app_name --replicas $replicas --namespace $namespace
    
    log success $"✅ Scaled ($app_name) to ($replicas) replicas"
}

# Restart deployments
def "main restart" [
    --app: string = "all"
    --namespace: string = "polyglot-dev"
] {
    log info $"Restarting deployments in namespace ($namespace)..."
    
    if $app == "all" {
        let environments = get-deployable-environments
        for env in $environments {
            restart-deployment $env.name $namespace
        }
    } else {
        restart-deployment $app $namespace
    }
}

def restart-deployment [app: string, namespace: string] {
    let app_name = $"($app)-app"
    
    try {
        kubectl rollout restart $"deployment/($app_name)" --namespace $namespace
        log success $"✅ Restarted ($app_name)"
    } catch { |e|
        log error $"❌ Failed to restart ($app_name): ($e.msg)"
    }
}

# Clean up development resources
def "main cleanup" [
    --namespace: string = "polyglot-dev"
    --confirm = false
] {
    if not $confirm {
        let response = input $"Are you sure you want to delete namespace '($namespace)' and all resources? (yes/no): "
        if $response != "yes" {
            log info "Cleanup cancelled"
            return
        }
    }
    
    log warn $"Cleaning up namespace '($namespace)'..."
    
    try {
        kubectl delete namespace $namespace
        log success $"✅ Deleted namespace '($namespace)'"
    } catch { |e|
        log error $"❌ Failed to delete namespace: ($e.msg)"
    }
}