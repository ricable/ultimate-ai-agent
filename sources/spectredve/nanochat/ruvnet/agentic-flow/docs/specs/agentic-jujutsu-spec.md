# SPARC Specification: Agentic-Jujutsu GitOps Platform

## Executive Summary

**agentic-jujutsu** is a next-generation GitOps platform that leverages Jujutsu's advanced version control capabilities to deliver state-of-the-art Kubernetes deployments with policy-first enforcement, progressive delivery, and unified control plane management.

**Key Innovation**: Jujutsu's change-centric model, conflict-free merging, and operation history provide superior GitOps workflows compared to traditional Git-based systems.

---

## S - SPECIFICATION

### 1. Project Objectives

**Primary Goal**: Build a complete GitOps platform using Jujutsu (jj) as the source of truth, integrating policy enforcement, progressive delivery, and platform abstractions into a unified deployment system.

**Success Metrics**:
- Deploy applications to Kubernetes with zero-touch GitOps reconciliation
- Enforce policy-as-code with 100% pre-deployment validation
- Achieve progressive delivery with automatic rollback on SLO violations
- Reduce deployment complexity by 70% through platform abstraction
- Support multi-cluster, multi-tenant environments
- Enable infrastructure-as-code through unified control plane

### 2. Core Requirements

#### 2.1 Jujutsu Integration (VCS Layer)
- Replace Git with Jujutsu as the source of truth
- Leverage Jujutsu's unique features:
  - **Change-centric workflow**: Every change has a unique ID
  - **Automatic conflict resolution**: Better merge strategies
  - **Operation log**: Complete audit trail of all operations
  - **Working copy as a commit**: Continuous snapshot capability
  - **Colocated branches**: Simplified branch management
- Bi-directional sync with Git repositories (for tool compatibility)
- Webhook integration for change notifications

#### 2.2 GitOps Controllers
- **Primary**: Argo CD integration with Jujutsu adapter
- **Secondary**: Flux CD compatibility layer
- Custom Jujutsu reconciliation controller:
  - Watch Jujutsu repository changes
  - Convert change IDs to deployment artifacts
  - Continuous reconciliation loop
  - Multi-cluster sync capability

#### 2.3 Progressive Delivery
- **Argo Rollouts** integration:
  - Canary deployments with traffic shifting
  - Blue-green deployments
  - A/B testing with header-based routing
- **Flagger** compatibility:
  - Service mesh integration (Istio, Linkerd)
  - Automated promotion/rollback
  - SLO-based decision making
- Metric analysis and health checks:
  - Prometheus query integration
  - OpenTelemetry trace analysis
  - Custom SLI/SLO definitions

#### 2.4 Policy-as-Code Enforcement
- **Kyverno** integration:
  - Pre-deployment validation
  - Mutation policies for standardization
  - Generation policies for automation
  - Verification for supply chain security
- **OPA/Gatekeeper** support:
  - Rego policy engine
  - Constraint templates
  - Audit and enforcement modes
- Policy testing framework:
  - Unit tests for policies
  - Integration tests with sample workloads
  - Policy coverage metrics

#### 2.5 Platform Engineering Layer
- Developer self-service portal
- Template system for common patterns:
  - Application scaffolding
  - Database-backed services
  - Event-driven microservices
  - ML inference endpoints
- GitOps-driven platform configuration
- Multi-tenancy with isolation guarantees

#### 2.6 Unified Control Plane
- **Crossplane** integration:
  - Cloud resource provisioning (AWS, GCP, Azure)
  - Database and queue management
  - Identity and access management
  - Network infrastructure
- **Composition functions** for reusable infrastructure patterns
- Dependency graph management

#### 2.7 Supply Chain Security
- **Sigstore** integration:
  - Image signing with Cosign
  - Keyless signing with Fulcio
  - Transparency log with Rekor
- **SBOM** generation and verification
- Policy enforcement for signed artifacts only
- Vulnerability scanning in the deployment pipeline

#### 2.8 Multi-Cluster Management
- Fleet management for cluster groups
- Per-environment GitOps repositories
- Cluster-specific overrides with Kustomize
- Progressive rollout across clusters
- Disaster recovery and failover

#### 2.9 Observability Integration
- **Prometheus** metrics collection
- **OpenTelemetry** distributed tracing
- Log aggregation (Loki, Elasticsearch)
- SLO monitoring and alerting
- Rollout metrics dashboard

### 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Developer Experience                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  CLI Tool    │  │  MCP Server  │  │  Web Portal  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Jujutsu VCS Layer                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Jujutsu Repository (Source of Truth)                    │  │
│  │  - Change tracking with unique IDs                       │  │
│  │  - Operation history for audit                           │  │
│  │  - Conflict-free merging                                 │  │
│  │  - Git co-location for tool compatibility               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GitOps Control Plane                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Argo CD     │  │  Flux CD     │  │  Custom JJ   │         │
│  │  (Adapter)   │  │  (Bridge)    │  │  Controller  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Policy Enforcement Layer                     │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │  Kyverno                 │  │  OPA Gatekeeper          │    │
│  │  - Validation            │  │  - Rego policies         │    │
│  │  - Mutation              │  │  - Constraint templates  │    │
│  │  - Generation            │  │  - Audit mode            │    │
│  │  - Verification (Sigstore)│ │                          │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Progressive Delivery Layer                    │
│  ┌──────────────────────────┐  ┌──────────────────────────┐    │
│  │  Argo Rollouts           │  │  Flagger                 │    │
│  │  - Canary                │  │  - Service mesh          │    │
│  │  - Blue-Green            │  │  - Automated rollback    │    │
│  │  - A/B Testing           │  │  - SLO analysis          │    │
│  └──────────────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Control Plane                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Crossplane                                              │  │
│  │  - Cloud resources (AWS, GCP, Azure)                    │  │
│  │  - Databases, queues, storage                           │  │
│  │  - Identity and networking                              │  │
│  │  - Composition functions                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Clusters                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Production  │  │  Staging     │  │  Development │         │
│  │  Cluster     │  │  Cluster     │  │  Cluster     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Observability Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Prometheus  │  │ OpenTelemetry│  │  Loki/ELK    │         │
│  │  (Metrics)   │  │  (Traces)    │  │  (Logs)      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Component Breakdown

#### 4.1 CLI Tool (`ajj` command)

**Core Commands**:
```bash
# Repository management
ajj init <repo-url>                    # Initialize Jujutsu GitOps repo
ajj clone <source> <dest>              # Clone GitOps repository
ajj status                             # Show deployment status across clusters

# Application management
ajj app create <name> [--template]     # Create new application
ajj app deploy <name> [--env]          # Deploy application
ajj app rollback <name> <change-id>    # Rollback to specific change
ajj app delete <name>                  # Delete application

# Progressive delivery
ajj rollout start <name> --strategy    # Start progressive rollout
ajj rollout promote <name>             # Promote canary
ajj rollout abort <name>               # Abort rollout
ajj rollout status <name>              # Check rollout status

# Policy management
ajj policy apply <file>                # Apply policy
ajj policy validate <manifest>         # Validate against policies
ajj policy test <policy-dir>           # Test policies
ajj policy audit                       # Audit cluster compliance

# Cluster management
ajj cluster add <name> <kubeconfig>    # Add cluster to fleet
ajj cluster sync <name>                # Force sync cluster
ajj cluster diff <name>                # Show drift
ajj cluster list                       # List all clusters

# Infrastructure (Crossplane)
ajj infra apply <file>                 # Apply infrastructure definition
ajj infra status <resource>            # Check infrastructure status
ajj infra delete <resource>            # Delete infrastructure

# Observability
ajj metrics <app>                      # Show application metrics
ajj logs <app> [--follow]              # Stream logs
ajj traces <app> <trace-id>            # Show trace details

# Security
ajj sign <image>                       # Sign container image
ajj verify <image>                     # Verify image signature
ajj sbom <image>                       # Generate SBOM

# Change management (Jujutsu-specific)
ajj change create <description>        # Create new change
ajj change list                        # List all changes
ajj change diff <id1> <id2>            # Compare changes
ajj change history <app>               # Show change history
ajj change squash <id>                 # Squash changes
```

#### 4.2 MCP Server (`mcp__agentic-jujutsu`)

**MCP Tools**:
```typescript
// Repository operations
jj_init(repo_url, config)
jj_clone(source, destination)
jj_status(repo_path)
jj_change_create(description, files)
jj_change_list(repo_path, filters)
jj_change_diff(change_id_1, change_id_2)

// GitOps operations
gitops_app_create(name, template, config)
gitops_app_deploy(name, environment, options)
gitops_app_sync(name, prune, dry_run)
gitops_app_rollback(name, change_id)
gitops_cluster_add(name, kubeconfig, labels)
gitops_cluster_sync(cluster_name, namespace)

// Policy operations
policy_validate(manifest, policy_set)
policy_apply(policy_file, cluster)
policy_test(policy_dir, test_cases)
policy_audit(cluster, namespace)

// Progressive delivery
rollout_create(name, strategy, steps)
rollout_promote(name, step)
rollout_abort(name, reason)
rollout_status(name)
rollout_metrics(name, time_range)

// Infrastructure operations
infra_apply(composition, parameters)
infra_status(resource_name)
infra_delete(resource_name, wait)
infra_claim_create(class, parameters)

// Observability
metrics_query(query, time_range)
traces_query(service, operation, time_range)
logs_query(app, level, time_range)
slo_status(app, slo_name)

// Security operations
security_sign_image(image, key)
security_verify_image(image)
security_sbom_generate(image, format)
security_scan_image(image, severity)
security_attestation_create(artifact, predicate)
```

#### 4.3 Custom Jujutsu GitOps Controller

**Reconciliation Loop**:
```
1. Watch Jujutsu repository for changes
2. Detect new change IDs
3. Extract Kubernetes manifests
4. Validate against policies (Kyverno/OPA)
5. Apply to target cluster(s)
6. Monitor deployment health
7. Update status in Jujutsu repo
8. Trigger progressive delivery if configured
9. Record metrics and audit trail
```

**Key Features**:
- Change-based reconciliation (vs commit-based)
- Automatic conflict resolution using jj's merge strategies
- Operation log for complete audit trail
- Multi-cluster synchronization
- Dependency ordering
- Health assessment
- Auto-healing

### 5. Jujutsu Advantages for GitOps

#### 5.1 Change-Centric Model
**Traditional Git**: Commits are snapshots, branches are pointers
**Jujutsu**: Changes are first-class objects with stable IDs

**GitOps Benefit**:
- Track deployment changes across branches without merge confusion
- Cherry-pick fixes between environments reliably
- Parallel environment changes without branch management overhead

#### 5.2 Conflict-Free Operations
**Traditional Git**: Manual merge conflict resolution
**Jujutsu**: Automatic conflict markers, multiple conflict resolution strategies

**GitOps Benefit**:
- Automated environment synchronization
- Parallel team deployments without blocking
- Safer automated reconciliation

#### 5.3 Operation History
**Traditional Git**: Reflog (local only, pruned)
**Jujutsu**: Complete operation log (persisted, replicated)

**GitOps Benefit**:
- Complete audit trail for compliance
- Rollback any operation, not just commits
- Debugging deployment issues with full history

#### 5.4 Working Copy as Commit
**Traditional Git**: Working directory is separate from history
**Jujutsu**: Working copy is always a commit (snapshots)

**GitOps Benefit**:
- No lost work from failed deployments
- Continuous backup of configuration changes
- Easier disaster recovery

#### 5.5 Colocated Branches
**Traditional Git**: Branch switching changes working directory
**Jujutsu**: Multiple branches in same working copy

**GitOps Benefit**:
- Compare environment configurations side-by-side
- Faster environment promotion workflows
- Reduced context switching for operators

---

## P - PSEUDOCODE

### Core Algorithms

#### Algorithm 1: Jujutsu Change Reconciliation

```
FUNCTION reconcile_jujutsu_changes(repo_path, cluster_config):
    // Initialize Jujutsu watcher
    watcher = JujutsuWatcher(repo_path)
    last_processed_change = load_checkpoint()

    WHILE true:
        // Detect new changes
        new_changes = watcher.get_changes_since(last_processed_change)

        FOR EACH change IN new_changes:
            // Extract change metadata
            change_id = change.id
            description = change.description
            affected_files = change.files

            // Filter for Kubernetes manifests
            manifests = filter_kubernetes_manifests(affected_files)

            IF manifests.is_empty():
                CONTINUE

            // Validate against policies
            validation_result = validate_policies(manifests)

            IF NOT validation_result.passed:
                log_error(change_id, validation_result.errors)
                update_status(change_id, "POLICY_FAILED")
                CONTINUE

            // Determine target clusters
            target_clusters = determine_clusters(change, cluster_config)

            // Apply to clusters
            FOR EACH cluster IN target_clusters:
                TRY:
                    // Check if progressive delivery is required
                    IF requires_progressive_delivery(manifests):
                        rollout_id = create_progressive_rollout(
                            cluster, manifests, change_id
                        )
                        monitor_rollout(rollout_id)
                    ELSE:
                        apply_manifests(cluster, manifests)
                        wait_for_ready(cluster, manifests)

                    update_status(change_id, cluster, "SYNCED")
                    record_metrics(change_id, cluster, "SUCCESS")

                CATCH deployment_error:
                    log_error(change_id, deployment_error)
                    update_status(change_id, cluster, "FAILED")

                    // Automatic rollback if configured
                    IF cluster_config.auto_rollback:
                        rollback_to_previous_change(cluster, change_id)

            // Update checkpoint
            last_processed_change = change_id
            save_checkpoint(last_processed_change)

        // Sleep before next poll
        sleep(reconciliation_interval)
END FUNCTION
```

#### Algorithm 2: Policy Validation Pipeline

```
FUNCTION validate_policies(manifests):
    validation_results = []

    // Phase 1: Kyverno validation
    FOR EACH manifest IN manifests:
        kyverno_result = kyverno.validate(manifest)

        IF kyverno_result.violations:
            validation_results.append({
                "engine": "kyverno",
                "manifest": manifest.name,
                "violations": kyverno_result.violations
            })

    // Phase 2: OPA/Gatekeeper validation
    FOR EACH manifest IN manifests:
        opa_result = opa_gatekeeper.check(manifest)

        IF opa_result.denials:
            validation_results.append({
                "engine": "opa",
                "manifest": manifest.name,
                "denials": opa_result.denials
            })

    // Phase 3: Supply chain verification
    FOR EACH manifest IN manifests:
        IF manifest.kind == "Deployment" OR manifest.kind == "StatefulSet":
            containers = extract_containers(manifest)

            FOR EACH container IN containers:
                // Verify image signature
                signature_valid = verify_image_signature(container.image)

                IF NOT signature_valid:
                    validation_results.append({
                        "engine": "sigstore",
                        "manifest": manifest.name,
                        "container": container.name,
                        "error": "Image not signed or signature invalid"
                    })

                // Check for vulnerabilities
                vuln_scan = scan_vulnerabilities(container.image)

                IF vuln_scan.critical_count > 0:
                    validation_results.append({
                        "engine": "vulnerability_scanner",
                        "manifest": manifest.name,
                        "container": container.name,
                        "critical_vulns": vuln_scan.critical_count
                    })

    // Phase 4: Custom policy checks
    custom_checks = run_custom_policies(manifests)
    validation_results.extend(custom_checks)

    RETURN {
        "passed": validation_results.is_empty(),
        "errors": validation_results
    }
END FUNCTION
```

#### Algorithm 3: Progressive Delivery with SLO Monitoring

```
FUNCTION progressive_delivery(app_name, new_version, strategy):
    // Initialize rollout
    rollout = Rollout(app_name, new_version, strategy)
    baseline_metrics = capture_baseline_metrics(app_name)

    // Define traffic steps based on strategy
    IF strategy == "canary":
        traffic_steps = [10%, 25%, 50%, 75%, 100%]
    ELSE IF strategy == "blue-green":
        traffic_steps = [0%, 100%]
    ELSE IF strategy == "ab-test":
        traffic_steps = [50%, 50%]  // Parallel traffic

    FOR EACH step IN traffic_steps:
        // Shift traffic
        shift_traffic(app_name, new_version, step)

        // Analysis window
        analysis_duration = strategy.analysis_duration  // e.g., 5 minutes
        sleep(analysis_duration)

        // Collect metrics during analysis
        current_metrics = collect_metrics(app_name, new_version, analysis_duration)

        // Evaluate SLOs
        slo_evaluation = evaluate_slos(baseline_metrics, current_metrics)

        IF NOT slo_evaluation.passed:
            // SLO violation detected
            log_warning(f"SLO violation at {step}% traffic: {slo_evaluation.violations}")

            // Automatic rollback
            rollback_deployment(app_name, new_version)
            shift_traffic(app_name, old_version, 100%)

            RETURN {
                "status": "FAILED",
                "reason": "SLO_VIOLATION",
                "violations": slo_evaluation.violations,
                "failed_at_step": step
            }

        // Check error rate
        error_rate = current_metrics.error_rate
        IF error_rate > strategy.max_error_rate:
            log_warning(f"High error rate at {step}%: {error_rate}")
            rollback_deployment(app_name, new_version)

            RETURN {
                "status": "FAILED",
                "reason": "HIGH_ERROR_RATE",
                "error_rate": error_rate
            }

        // Step passed, continue
        log_info(f"Step {step}% passed SLO checks")

    // All steps passed
    finalize_deployment(app_name, new_version)

    RETURN {
        "status": "SUCCESS",
        "version": new_version,
        "duration": rollout.duration()
    }
END FUNCTION

FUNCTION evaluate_slos(baseline, current):
    violations = []

    // SLO: Latency P95 < 500ms and not >20% increase
    IF current.latency_p95 > 500 OR
       current.latency_p95 > baseline.latency_p95 * 1.2:
        violations.append({
            "slo": "latency_p95",
            "threshold": "500ms or +20%",
            "actual": current.latency_p95
        })

    // SLO: Error rate < 1% and not >2x baseline
    IF current.error_rate > 0.01 OR
       current.error_rate > baseline.error_rate * 2:
        violations.append({
            "slo": "error_rate",
            "threshold": "1% or 2x baseline",
            "actual": current.error_rate
        })

    // SLO: Success rate > 99.5%
    IF current.success_rate < 0.995:
        violations.append({
            "slo": "success_rate",
            "threshold": "99.5%",
            "actual": current.success_rate
        })

    // Custom SLOs from app config
    FOR EACH slo IN app_config.custom_slos:
        result = evaluate_custom_slo(slo, baseline, current)
        IF NOT result.passed:
            violations.append(result)

    RETURN {
        "passed": violations.is_empty(),
        "violations": violations
    }
END FUNCTION
```

#### Algorithm 4: Multi-Cluster Sync with Dependency Management

```
FUNCTION sync_multi_cluster(change_id, clusters):
    // Extract dependency graph from manifests
    manifests = get_manifests_for_change(change_id)
    dependency_graph = build_dependency_graph(manifests)

    // Topological sort for ordered deployment
    deployment_order = topological_sort(dependency_graph)

    // Group clusters by environment
    cluster_groups = group_by_environment(clusters)

    // Deploy to environments in order (dev -> staging -> prod)
    environment_order = ["development", "staging", "production"]

    FOR EACH env IN environment_order:
        env_clusters = cluster_groups[env]

        // Parallel deployment within environment
        deploy_tasks = []

        FOR EACH cluster IN env_clusters:
            task = async_deploy_to_cluster(
                cluster,
                deployment_order,
                change_id
            )
            deploy_tasks.append(task)

        // Wait for all clusters in environment
        results = await_all(deploy_tasks)

        // Check if all deployments succeeded
        all_success = all(r.status == "SUCCESS" for r in results)

        IF NOT all_success:
            failed_clusters = [r.cluster for r in results if r.status != "SUCCESS"]
            log_error(f"Deployment failed in {env}: {failed_clusters}")

            // Rollback all clusters in environment
            FOR EACH cluster IN env_clusters:
                rollback_cluster(cluster, change_id)

            // Stop promotion to next environment
            RETURN {
                "status": "FAILED",
                "failed_environment": env,
                "failed_clusters": failed_clusters
            }

        // Environment deployment successful
        log_info(f"Environment {env} deployed successfully")

        // Wait before next environment (soak time)
        IF env != "production":
            soak_time = get_soak_time(env)
            sleep(soak_time)

    RETURN {
        "status": "SUCCESS",
        "change_id": change_id,
        "clusters_deployed": len(clusters)
    }
END FUNCTION

FUNCTION build_dependency_graph(manifests):
    graph = DependencyGraph()

    FOR EACH manifest IN manifests:
        graph.add_node(manifest)

        // Extract dependencies
        IF manifest.kind == "Deployment":
            // Depends on ConfigMaps
            IF manifest.spec.volumes:
                FOR EACH volume IN manifest.spec.volumes:
                    IF volume.configMap:
                        configmap = find_manifest(manifests, "ConfigMap", volume.configMap.name)
                        graph.add_edge(configmap, manifest)

            // Depends on Secrets
            IF manifest.spec.containers:
                FOR EACH container IN manifest.spec.containers:
                    IF container.envFrom:
                        FOR EACH envFrom IN container.envFrom:
                            IF envFrom.secretRef:
                                secret = find_manifest(manifests, "Secret", envFrom.secretRef.name)
                                graph.add_edge(secret, manifest)

        // Service depends on Deployment
        IF manifest.kind == "Service":
            deployment = find_deployment_for_service(manifests, manifest)
            IF deployment:
                graph.add_edge(deployment, manifest)

        // Ingress depends on Service
        IF manifest.kind == "Ingress":
            FOR EACH rule IN manifest.spec.rules:
                FOR EACH path IN rule.http.paths:
                    service = find_manifest(manifests, "Service", path.backend.service.name)
                    graph.add_edge(service, manifest)

    RETURN graph
END FUNCTION
```

#### Algorithm 5: Crossplane Infrastructure Provisioning

```
FUNCTION provision_infrastructure(app_config):
    // Parse infrastructure requirements
    infra_requirements = parse_infrastructure(app_config)

    // Create Crossplane compositions
    compositions = []

    FOR EACH requirement IN infra_requirements:
        IF requirement.type == "database":
            composition = create_database_composition(requirement)
        ELSE IF requirement.type == "queue":
            composition = create_queue_composition(requirement)
        ELSE IF requirement.type == "storage":
            composition = create_storage_composition(requirement)
        ELSE IF requirement.type == "network":
            composition = create_network_composition(requirement)

        compositions.append(composition)

    // Create claims with dependencies
    claims = []

    FOR EACH composition IN compositions:
        // Check if dependencies are ready
        dependencies_ready = check_dependencies(composition.dependencies)

        IF NOT dependencies_ready:
            log_info(f"Waiting for dependencies: {composition.name}")
            wait_for_dependencies(composition.dependencies)

        // Create claim
        claim = create_crossplane_claim(composition)
        claims.append(claim)

        // Wait for infrastructure to be ready
        wait_for_ready(claim, timeout=600)

        // Extract connection details
        connection_details = get_connection_details(claim)

        // Store in Kubernetes secret
        create_secret(app_config.namespace, connection_details)

    RETURN {
        "status": "SUCCESS",
        "claims": claims,
        "connection_secrets": [c.name + "-connection" for c in claims]
    }
END FUNCTION

FUNCTION create_database_composition(requirement):
    // Determine cloud provider
    provider = requirement.provider  // aws, gcp, azure

    IF provider == "aws":
        resource = {
            "apiVersion": "database.aws.crossplane.io/v1beta1",
            "kind": "RDSInstance",
            "spec": {
                "forProvider": {
                    "region": requirement.region,
                    "dbInstanceClass": requirement.instance_class,
                    "engine": requirement.engine,
                    "engineVersion": requirement.version,
                    "allocatedStorage": requirement.storage_gb,
                    "storageEncrypted": true,
                    "storageType": "gp3",
                    "multiAZ": requirement.high_availability,
                    "backupRetentionPeriod": requirement.backup_retention_days
                },
                "writeConnectionSecretToRef": {
                    "name": requirement.name + "-connection",
                    "namespace": requirement.namespace
                }
            }
        }

    RETURN Composition(
        name=requirement.name,
        type="database",
        resources=[resource],
        dependencies=requirement.dependencies
    )
END FUNCTION
```

---

## A - ARCHITECTURE

### 1. System Components

#### 1.1 Core Components

**A. Jujutsu Integration Layer**
- **JJ Repository Manager**: Clone, init, sync operations
- **Change Tracker**: Monitor repository for new changes
- **Operation Log Reader**: Audit trail access
- **Git Bridge**: Bi-directional sync with Git for tool compatibility

**B. GitOps Control Plane**
- **Custom JJ Controller**: Kubernetes operator watching Jujutsu repos
- **Argo CD Adapter**: Bridge between Argo CD and Jujutsu
- **Flux Bridge**: Optional Flux compatibility
- **Multi-Cluster Sync Engine**: Fleet management

**C. Policy Enforcement**
- **Kyverno Integration**: Validation, mutation, generation policies
- **OPA Integration**: Rego policy evaluation
- **Policy Test Framework**: Unit/integration testing
- **Admission Controller Proxy**: Pre-deployment validation

**D. Progressive Delivery**
- **Rollout Manager**: Orchestrate progressive strategies
- **Argo Rollouts Integration**: Canary, blue-green, A/B
- **Flagger Integration**: Service mesh-based delivery
- **SLO Evaluator**: Metrics analysis for promotion decisions
- **Traffic Manager**: Service mesh configuration

**E. Unified Control Plane**
- **Crossplane Operator**: Infrastructure provisioning
- **Composition Manager**: Reusable infrastructure patterns
- **Dependency Resolver**: Order infrastructure creation
- **Connection Secret Manager**: Inject credentials into apps

**F. Observability**
- **Metrics Collector**: Prometheus integration
- **Trace Analyzer**: OpenTelemetry processing
- **Log Aggregator**: Loki/ELK integration
- **SLO Monitor**: Track and alert on objectives
- **Rollout Dashboard**: Visualize progressive delivery

**G. Security**
- **Image Signer**: Sigstore/Cosign integration
- **Signature Verifier**: Admission webhook
- **SBOM Generator**: Software bill of materials
- **Vulnerability Scanner**: CVE detection
- **Attestation Manager**: Supply chain metadata

#### 1.2 Interface Components

**A. CLI Tool (`ajj`)**
- Command-line interface for all operations
- Interactive prompts for complex operations
- Rich terminal UI with progress indicators
- Configuration management
- Plugin system for extensibility

**B. MCP Server**
- MCP protocol implementation
- Tool definitions for all operations
- Streaming results for long operations
- Session management
- Authentication and authorization

**C. Web Portal** (Future)
- Dashboard for fleet overview
- Application catalog
- Policy editor
- Rollout visualization
- Audit log viewer

### 2. Data Flow Diagrams

#### 2.1 Application Deployment Flow

```
Developer               CLI/MCP              Jujutsu Repo         Controller
   │                      │                       │                    │
   ├─(1)─ ajj app create ─>                      │                    │
   │                      ├─(2)─ jj new ────────>│                    │
   │                      ├─(3)─ generate manifests                   │
   │                      ├─(4)─ jj describe ───>│                    │
   │                      │                       ├─(5)─ change event──>
   │                      │                       │                    │
   │                      │                       │         ┌──────────┴──────────┐
   │                      │                       │         │ Fetch change        │
   │                      │                       │         │ Extract manifests   │
   │                      │                       │         │ Validate policies   │
   │                      │                       │         └──────────┬──────────┘
   │                      │                       │                    │
   │                      │                       │                    ▼
   │                      │                       │         ┌──────────────────────┐
   │                      │                       │         │ Apply to Kubernetes │
   │                      │                       │         │ Monitor health      │
   │                      │                       │         └──────────┬──────────┘
   │                      │                       │                    │
   │                      │                       │<─(6)─ update status┤
   │<─(7)─ deployment success ────────────────────│                    │
   │                      │                       │                    │
```

#### 2.2 Progressive Rollout Flow

```
Controller          Rollout Manager      Service Mesh      Prometheus
   │                      │                    │                │
   ├─(1)─ start rollout ─>│                    │                │
   │                      ├─(2)─ shift 10% ───>│                │
   │                      │                    ├─(3)─ apply────>│
   │                      ├─(4)─ wait analysis │                │
   │                      ├─(5)─ query metrics ────────────────>│
   │                      │<─(6)─ return metrics ───────────────┤
   │                      │                    │                │
   │                      ├─(7)─ evaluate SLOs │                │
   │                      │ [SLO passed]       │                │
   │                      │                    │                │
   │                      ├─(8)─ shift 25% ───>│                │
   │                      ├─(9)─ wait analysis │                │
   │                      ├─(10) query metrics ────────────────>│
   │                      │<─(11) return metrics ───────────────┤
   │                      │                    │                │
   │                      ├─(12) evaluate SLOs │                │
   │                      │ [SLO failed]       │                │
   │                      │                    │                │
   │                      ├─(13) rollback ────>│                │
   │<─(14) rollout failed ┤                    │                │
```

### 3. Data Models

#### 3.1 Jujutsu GitOps Repository Structure

```
gitops-repo/
├── .jj/                           # Jujutsu metadata
├── clusters/                      # Cluster definitions
│   ├── production/
│   │   ├── cluster-config.yaml    # Cluster metadata
│   │   └── kubeconfig             # Access credentials
│   ├── staging/
│   └── development/
├── apps/                          # Applications
│   ├── web-frontend/
│   │   ├── base/                  # Base manifests
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── kustomization.yaml
│   │   ├── overlays/              # Environment overlays
│   │   │   ├── development/
│   │   │   ├── staging/
│   │   │   └── production/
│   │   ├── rollout.yaml           # Progressive delivery config
│   │   └── app-config.yaml        # Application metadata
│   └── api-backend/
├── infrastructure/                # Crossplane compositions
│   ├── databases/
│   │   ├── postgres-composition.yaml
│   │   └── postgres-claim.yaml
│   ├── queues/
│   │   └── sqs-composition.yaml
│   └── storage/
│       └── s3-composition.yaml
├── policies/                      # Policy-as-code
│   ├── kyverno/
│   │   ├── require-labels.yaml
│   │   ├── require-resources.yaml
│   │   ├── verify-images.yaml
│   │   └── generate-network-policy.yaml
│   ├── opa/
│   │   ├── constraints/
│   │   └── constraint-templates/
│   └── tests/                     # Policy tests
│       ├── kyverno-tests/
│       └── opa-tests/
├── platform/                      # Platform templates
│   ├── templates/
│   │   ├── web-app-template/
│   │   ├── api-template/
│   │   └── worker-template/
│   └── compositions/              # Reusable patterns
└── config/
    ├── ajj-config.yaml            # Agentic-Jujutsu configuration
    ├── argocd-config.yaml         # Argo CD settings
    └── rollout-defaults.yaml      # Default rollout strategies
```

#### 3.2 Configuration Schema

**Application Config (`app-config.yaml`)**:
```yaml
apiVersion: ajj.io/v1
kind: Application
metadata:
  name: web-frontend
  namespace: default
spec:
  source:
    path: apps/web-frontend/overlays
    targetRevision: "@main"  # Jujutsu change/branch

  destination:
    clusters:
      - development
      - staging
      - production

  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2

  progressiveDelivery:
    enabled: true
    strategy: canary
    steps:
      - setWeight: 10
      - pause: {duration: 5m}
      - analysis:
          templates:
            - name: error-rate
            - name: latency-p95
      - setWeight: 25
      - pause: {duration: 5m}
      - analysis:
          templates:
            - name: error-rate
            - name: latency-p95
      - setWeight: 50
      - pause: {duration: 10m}
      - setWeight: 100

    trafficRouting:
      serviceMesh: istio
      virtualService: web-frontend-vsvc

  slos:
    - name: error-rate
      query: |
        sum(rate(http_requests_total{status=~"5.."}[5m])) /
        sum(rate(http_requests_total[5m]))
      threshold: 0.01  # 1%

    - name: latency-p95
      query: |
        histogram_quantile(0.95,
          sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
      threshold: 0.5  # 500ms

  infrastructure:
    required:
      - name: postgres-db
        type: database
        composition: postgresql-cluster
        parameters:
          instanceClass: db.t3.medium
          storageGB: 100
          highAvailability: true

      - name: redis-cache
        type: cache
        composition: redis-cluster
        parameters:
          nodeType: cache.t3.medium
          numNodes: 3

  policies:
    validation:
      - require-resource-limits
      - require-labels
      - require-non-root
      - verify-image-signature

    enforcementMode: strict  # strict or permissive

  observability:
    metrics:
      enabled: true
      scrapeInterval: 30s

    tracing:
      enabled: true
      samplingRate: 0.1

    logging:
      enabled: true
      level: info
```

**Rollout Strategy Config**:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate
spec:
  metrics:
    - name: error-rate
      initialDelay: 1m
      interval: 1m
      count: 5
      successCondition: result < 0.01
      failureCondition: result >= 0.05
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{app="web-frontend",status=~"5.."}[2m])) /
            sum(rate(http_requests_total{app="web-frontend"}[2m]))
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: latency-p95
spec:
  metrics:
    - name: latency-p95
      initialDelay: 1m
      interval: 1m
      count: 5
      successCondition: result < 0.5
      failureCondition: result >= 1.0
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            histogram_quantile(0.95,
              sum(rate(http_request_duration_seconds_bucket{app="web-frontend"}[2m])) by (le))
```

#### 3.3 MCP Protocol Schema

**Tool Definition Example**:
```json
{
  "name": "jj_change_create",
  "description": "Create a new Jujutsu change for GitOps deployment",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo_path": {
        "type": "string",
        "description": "Path to Jujutsu repository"
      },
      "description": {
        "type": "string",
        "description": "Change description"
      },
      "files": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
          }
        }
      },
      "metadata": {
        "type": "object",
        "properties": {
          "app": {"type": "string"},
          "environment": {"type": "string"},
          "type": {"type": "string", "enum": ["deployment", "config", "policy"]}
        }
      }
    },
    "required": ["repo_path", "description", "files"]
  }
}
```

### 4. Technology Stack

#### 4.1 Core Technologies

**Language**: TypeScript/Node.js
- Primary implementation language
- Rich ecosystem for CLI and MCP
- Excellent Kubernetes client libraries

**Version Control**: Jujutsu (`jj`)
- Source of truth for GitOps
- Change-centric model
- Operation history

**Kubernetes**: v1.28+
- Target platform
- Custom resource definitions (CRDs)
- Operator SDK

#### 4.2 GitOps Components

**Argo CD**: v2.9+
- GitOps controller (with Jujutsu adapter)
- Multi-cluster management
- Web UI

**Flux CD**: v2.2+ (optional)
- Alternative/complementary controller
- Lighter weight

**Custom Controller**:
- Language: Go (for Kubernetes operator best practices)
- Framework: Kubebuilder/Operator SDK
- Watches: Jujutsu repositories

#### 4.3 Policy Enforcement

**Kyverno**: v1.11+
- Kubernetes-native policy engine
- YAML-based policies
- Validation, mutation, generation, verification

**OPA/Gatekeeper**: v3.14+
- General-purpose policy engine
- Rego language
- Constraint framework

#### 4.4 Progressive Delivery

**Argo Rollouts**: v1.6+
- Advanced deployment strategies
- Traffic management
- Analysis and metrics

**Flagger**: v1.35+
- Service mesh integration
- Automated canary analysis
- Multi-provider support

#### 4.5 Service Mesh

**Istio**: v1.20+ (primary)
- Traffic management
- Observability
- Security

**Linkerd**: v2.14+ (alternative)
- Lightweight option
- Simpler configuration

#### 4.6 Infrastructure as Code

**Crossplane**: v1.14+
- Universal control plane
- Cloud provider integrations
- Composition functions

#### 4.7 Observability

**Prometheus**: v2.48+
- Metrics collection
- PromQL queries
- Alerting

**OpenTelemetry**: v1.22+
- Distributed tracing
- Unified observability
- Vendor-neutral

**Grafana**: v10.2+
- Visualization
- Dashboards
- Alerts

**Loki**: v2.9+
- Log aggregation
- LogQL queries

#### 4.8 Security

**Sigstore**:
- **Cosign**: Image signing
- **Fulcio**: Certificate authority
- **Rekor**: Transparency log

**Trivy**: Vulnerability scanning

#### 4.9 Development Tools

**CLI Framework**: oclif
- TypeScript-based
- Plugin architecture
- Auto-documentation

**MCP SDK**: `@modelcontextprotocol/sdk`
- Official MCP implementation
- TypeScript support

**Testing**:
- **Jest**: Unit testing
- **Testcontainers**: Integration testing
- **k3d**: Local Kubernetes clusters

**Build**:
- **TypeScript**: Compilation
- **esbuild**: Bundling
- **npm**: Package management

---

## R - REFINEMENT

### 1. Implementation Phases

#### Phase 1: Foundation (Weeks 1-2)
- Set up project structure
- Implement Jujutsu integration layer
- Create basic CLI scaffolding
- Set up MCP server framework
- Establish testing infrastructure

#### Phase 2: Core GitOps (Weeks 3-4)
- Build custom Jujutsu controller
- Implement change reconciliation loop
- Create Argo CD adapter
- Add basic policy validation (Kyverno)
- Multi-cluster sync engine

#### Phase 3: Progressive Delivery (Weeks 5-6)
- Integrate Argo Rollouts
- Implement SLO evaluation
- Add traffic management
- Create rollout dashboard
- Automated rollback logic

#### Phase 4: Policy & Security (Week 7)
- Complete Kyverno integration
- Add OPA/Gatekeeper support
- Implement Sigstore signing/verification
- SBOM generation
- Vulnerability scanning

#### Phase 5: Infrastructure Control Plane (Week 8)
- Crossplane integration
- Composition library
- Dependency management
- Connection secret injection

#### Phase 6: Observability (Week 9)
- Prometheus integration
- OpenTelemetry tracing
- Log aggregation
- SLO monitoring
- Rollout metrics

#### Phase 7: Platform Features (Week 10)
- Application templates
- Self-service portal
- Multi-tenancy
- RBAC integration

#### Phase 8: Polish & Documentation (Weeks 11-12)
- Performance optimization
- Error handling
- Comprehensive documentation
- Example repositories
- Video tutorials

### 2. Testing Strategy

#### 2.1 Unit Tests
- All core functions with >90% coverage
- Mock Kubernetes API
- Mock Jujutsu operations
- Policy validation logic
- SLO evaluation algorithms

#### 2.2 Integration Tests
- Local Kubernetes cluster (k3d)
- Real Jujutsu repositories
- End-to-end deployment flows
- Multi-cluster scenarios
- Rollback procedures

#### 2.3 Policy Tests
- Kyverno policy unit tests
- OPA policy tests with Conftest
- Example manifests (valid/invalid)
- Edge cases

#### 2.4 Performance Tests
- Reconciliation loop latency
- Multi-cluster sync performance
- Large manifest processing
- Policy validation overhead

#### 2.5 Security Tests
- Image signature verification
- SBOM generation
- Vulnerability detection
- RBAC enforcement

### 3. Performance Optimization

#### 3.1 Reconciliation
- Incremental processing (only changed files)
- Parallel cluster sync
- Efficient Jujutsu operations
- Caching of validated manifests

#### 3.2 Policy Validation
- Pre-compile OPA policies
- Cache validation results
- Parallel policy checks
- Early termination on failure

#### 3.3 Progressive Delivery
- Efficient metrics queries
- Caching of baseline metrics
- Parallel SLO evaluation
- Optimized traffic shifting

### 4. Error Handling

#### 4.1 Deployment Errors
- Automatic rollback on failure
- Detailed error messages
- Retry with exponential backoff
- Circuit breaker for failing clusters

#### 4.2 Policy Violations
- Clear violation messages
- Suggestions for fixes
- Dry-run mode
- Policy exception workflow

#### 4.3 Infrastructure Errors
- Dependency resolution failures
- Cloud provider API errors
- Timeout handling
- Resource cleanup on failure

### 5. Security Considerations

#### 5.1 Authentication
- Kubeconfig-based auth
- OIDC integration
- Service account tokens
- MCP authentication

#### 5.2 Authorization
- Kubernetes RBAC
- Namespace isolation
- Policy-based access control
- Audit logging

#### 5.3 Secrets Management
- Encrypted secrets in Jujutsu (git-crypt)
- External secret operators (ESO, Sealed Secrets)
- Vault integration
- Secret rotation

#### 5.4 Supply Chain
- Mandatory image signing
- SBOM requirements
- Vulnerability blocking
- Provenance attestations

### 6. Configuration Management

#### 6.1 Global Configuration
```yaml
# ~/.config/ajj/config.yaml
repositories:
  default: ~/gitops-repo
  cache: ~/.cache/ajj

clusters:
  contexts:
    - name: production
      kubeconfig: ~/.kube/production
    - name: staging
      kubeconfig: ~/.kube/staging

policies:
  enforcementMode: strict
  engines:
    - kyverno
    - opa

progressiveDelivery:
  defaultStrategy: canary
  analysisInterval: 5m
  autoPromotion: false

observability:
  prometheus:
    url: http://prometheus:9090
  jaeger:
    url: http://jaeger:16686

security:
  requireSignedImages: true
  cosign:
    publicKeyPath: ~/.config/ajj/cosign.pub
```

#### 6.2 Repository Configuration
```yaml
# gitops-repo/config/ajj-config.yaml
apiVersion: ajj.io/v1
kind: Configuration
metadata:
  name: gitops-config

spec:
  reconciliation:
    interval: 30s
    timeout: 5m
    retries: 3

  clusters:
    - name: production
      environment: production
      sync:
        automated: true
        prune: true
        selfHeal: true

    - name: staging
      environment: staging
      sync:
        automated: true
        prune: true

  policies:
    kyverno:
      enabled: true
      policyPath: policies/kyverno
    opa:
      enabled: true
      policyPath: policies/opa

  progressiveDelivery:
    defaults:
      strategy: canary
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - setWeight: 25
        - pause: {duration: 5m}
        - setWeight: 50
        - pause: {duration: 10m}
        - setWeight: 100

  observability:
    metrics:
      prometheus:
        url: http://prometheus:9090
    tracing:
      jaeger:
        url: http://jaeger:16686
```

---

## C - COMPLETION

### 1. Deliverables

#### 1.1 Core Package
- **NPM Package**: `agentic-jujutsu`
- **CLI Binary**: `ajj`
- **MCP Server**: Included in package

#### 1.2 Additional Packages
- `@agentic-jujutsu/core`: Core library
- `@agentic-jujutsu/cli`: CLI implementation
- `@agentic-jujutsu/mcp`: MCP server
- `@agentic-jujutsu/policies`: Policy library
- `@agentic-jujutsu/templates`: Application templates

#### 1.3 Kubernetes Operators
- `jujutsu-gitops-controller`: Custom controller
- Helm chart for installation

#### 1.4 Documentation
- README.md: Quick start
- Architecture guide
- CLI reference
- MCP tool reference
- Policy writing guide
- Template creation guide
- Migration guide (from Git-based GitOps)
- Troubleshooting guide

#### 1.5 Examples
- Sample GitOps repository
- Application templates
- Policy examples
- Infrastructure compositions
- Rollout strategies

### 2. Installation & Setup

#### 2.1 Install CLI
```bash
npm install -g agentic-jujutsu

# Or with specific version
npm install -g agentic-jujutsu@latest

# Verify installation
ajj --version
ajj --help
```

#### 2.2 Install MCP Server
```bash
# Add to Claude Desktop or other MCP client
claude mcp add agentic-jujutsu npx agentic-jujutsu mcp start

# Or manual configuration in config file
{
  "mcpServers": {
    "agentic-jujutsu": {
      "command": "npx",
      "args": ["agentic-jujutsu", "mcp", "start"]
    }
  }
}
```

#### 2.3 Install Kubernetes Controller
```bash
# Add Helm repository
helm repo add agentic-jujutsu https://charts.agentic-jujutsu.io
helm repo update

# Install controller
helm install ajj-controller agentic-jujutsu/jujutsu-gitops-controller \
  --namespace ajj-system \
  --create-namespace \
  --set repoUrl=https://github.com/org/gitops-repo \
  --set reconciliation.interval=30s

# Verify installation
kubectl get pods -n ajj-system
```

#### 2.4 Install Dependencies

**Kyverno**:
```bash
kubectl create -f https://github.com/kyverno/kyverno/releases/download/v1.11.0/install.yaml
```

**Argo Rollouts**:
```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

**Crossplane**:
```bash
helm repo add crossplane-stable https://charts.crossplane.io/stable
helm install crossplane crossplane-stable/crossplane \
  --namespace crossplane-system \
  --create-namespace
```

**Prometheus** (if not already installed):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### 3. Quick Start Guide

#### Step 1: Initialize GitOps Repository
```bash
# Create new Jujutsu repository
ajj init https://github.com/org/gitops-repo

# Or clone existing
ajj clone https://github.com/org/gitops-repo ~/gitops-repo
cd ~/gitops-repo
```

#### Step 2: Add Cluster
```bash
# Add Kubernetes cluster
ajj cluster add production ~/.kube/production-config

# Verify cluster
ajj cluster list
```

#### Step 3: Create Application
```bash
# Create from template
ajj app create web-frontend --template=web-app

# This creates:
# - apps/web-frontend/base/
# - apps/web-frontend/overlays/
# - apps/web-frontend/app-config.yaml
```

#### Step 4: Deploy Application
```bash
# Deploy to development
ajj app deploy web-frontend --env=development

# Check status
ajj app status web-frontend

# View logs
ajj logs web-frontend --follow
```

#### Step 5: Apply Policies
```bash
# Apply policy set
ajj policy apply policies/kyverno/require-labels.yaml

# Validate application
ajj policy validate apps/web-frontend/base/deployment.yaml
```

#### Step 6: Progressive Rollout
```bash
# Update application (new image)
# Edit apps/web-frontend/base/deployment.yaml

# Start rollout
ajj rollout start web-frontend --strategy=canary

# Monitor rollout
ajj rollout status web-frontend

# Promote if successful
ajj rollout promote web-frontend
```

### 4. Integration Examples

#### 4.1 CI/CD Pipeline Integration

**GitHub Actions**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install agentic-jujutsu
        run: npm install -g agentic-jujutsu

      - name: Configure cluster
        run: |
          echo "${{ secrets.KUBECONFIG }}" > /tmp/kubeconfig
          ajj cluster add production /tmp/kubeconfig

      - name: Validate policies
        run: ajj policy validate apps/web-frontend/overlays/production

      - name: Sign images
        run: |
          echo "${{ secrets.COSIGN_KEY }}" > /tmp/cosign.key
          ajj sign ${{ steps.build.outputs.image }} --key=/tmp/cosign.key

      - name: Deploy application
        run: ajj app deploy web-frontend --env=production

      - name: Monitor rollout
        run: ajj rollout status web-frontend --wait
```

#### 4.2 MCP Integration

**Claude Desktop Integration**:
```typescript
// Claude can now use MCP tools
// Example conversation:

User: "Deploy the web-frontend application to production with a canary rollout"