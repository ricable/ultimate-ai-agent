# UAP Infrastructure as Code - Main Configuration
# Multi-cloud deployment with auto-scaling and failover

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Remote state configuration for team collaboration
  backend "s3" {
    bucket         = "uap-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "uap-terraform-locks"
    
    # State versioning and backup
    versioning = true
  }
}

# Local variables for configuration
locals {
  environment = var.environment
  project     = "uap"
  
  # Common tags for all resources
  common_tags = {
    Project     = local.project
    Environment = local.environment
    ManagedBy   = "terraform"
    Owner       = "uap-devops"
    CreatedAt   = timestamp()
  }
  
  # Multi-cloud configuration
  primary_cloud   = var.primary_cloud
  secondary_cloud = var.secondary_cloud
  
  # Resource naming convention
  name_prefix = "${local.project}-${local.environment}"
  
  # Network configuration
  vpc_cidr = var.vpc_cidr
  availability_zones = var.availability_zones
  
  # Auto-scaling configuration
  min_instances = var.min_instances
  max_instances = var.max_instances
  target_cpu_utilization = var.target_cpu_utilization
  
  # Database configuration
  db_instance_class = var.db_instance_class
  db_storage_size   = var.db_storage_size
  db_backup_retention = var.db_backup_retention
  
  # Load balancing configuration
  enable_cross_zone_load_balancing = var.enable_cross_zone_load_balancing
  load_balancer_type = var.load_balancer_type
  
  # Monitoring configuration
  enable_detailed_monitoring = var.enable_detailed_monitoring
  log_retention_days = var.log_retention_days
  
  # Security configuration
  enable_encryption_at_rest = var.enable_encryption_at_rest
  enable_encryption_in_transit = var.enable_encryption_in_transit
  enable_waf = var.enable_waf
}

# Data sources for existing resources
data "aws_caller_identity" "current" {
  count = local.primary_cloud == "aws" ? 1 : 0
}

data "aws_region" "current" {
  count = local.primary_cloud == "aws" ? 1 : 0
}

data "google_client_config" "current" {
  count = local.primary_cloud == "gcp" ? 1 : 0
}

data "azurerm_client_config" "current" {
  count = local.primary_cloud == "azure" ? 1 : 0
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
  
  lifecycle {
    ignore_changes = [length, special]
  }
}

# AWS Infrastructure Module
module "aws_infrastructure" {
  count  = local.primary_cloud == "aws" || local.secondary_cloud == "aws" ? 1 : 0
  source = "./modules/aws"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  tags        = local.common_tags
  
  # Network configuration
  vpc_cidr           = local.vpc_cidr
  availability_zones = local.availability_zones
  
  # Compute configuration
  min_instances              = local.min_instances
  max_instances              = local.max_instances
  target_cpu_utilization     = local.target_cpu_utilization
  instance_type              = var.aws_instance_type
  key_pair_name              = var.aws_key_pair_name
  
  # Database configuration
  db_instance_class    = local.db_instance_class
  db_storage_size      = local.db_storage_size
  db_backup_retention  = local.db_backup_retention
  db_password          = random_password.db_password.result
  db_multi_az          = var.aws_db_multi_az
  
  # Load balancing
  load_balancer_type               = local.load_balancer_type
  enable_cross_zone_load_balancing = local.enable_cross_zone_load_balancing
  
  # Security configuration
  enable_encryption_at_rest     = local.enable_encryption_at_rest
  enable_encryption_in_transit  = local.enable_encryption_in_transit
  enable_waf                    = local.enable_waf
  kms_key_id                    = var.aws_kms_key_id
  
  # Monitoring configuration
  enable_detailed_monitoring = local.enable_detailed_monitoring
  log_retention_days         = local.log_retention_days
  sns_topic_arn              = var.aws_sns_topic_arn
  
  # Storage configuration
  s3_bucket_prefix = var.aws_s3_bucket_prefix
  enable_s3_versioning = var.aws_enable_s3_versioning
  s3_lifecycle_rules = var.aws_s3_lifecycle_rules
}

# GCP Infrastructure Module
module "gcp_infrastructure" {
  count  = local.primary_cloud == "gcp" || local.secondary_cloud == "gcp" ? 1 : 0
  source = "./modules/gcp"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  labels      = local.common_tags
  
  # GCP-specific configuration
  gcp_project    = var.gcp_project_id
  gcp_region     = var.gcp_region
  gcp_zones      = var.gcp_zones
  
  # Network configuration
  vpc_cidr = local.vpc_cidr
  
  # Compute configuration
  min_instances              = local.min_instances
  max_instances              = local.max_instances
  target_cpu_utilization     = local.target_cpu_utilization
  machine_type               = var.gcp_machine_type
  
  # Database configuration
  db_tier             = var.gcp_db_tier
  db_storage_size     = local.db_storage_size
  db_backup_retention = local.db_backup_retention
  db_password         = random_password.db_password.result
  db_high_availability = var.gcp_db_high_availability
  
  # Security configuration
  enable_encryption_at_rest = local.enable_encryption_at_rest
  kms_key_ring             = var.gcp_kms_key_ring
  kms_crypto_key           = var.gcp_kms_crypto_key
  
  # Monitoring configuration
  enable_detailed_monitoring = local.enable_detailed_monitoring
  log_retention_days         = local.log_retention_days
  
  # Storage configuration
  gcs_bucket_prefix = var.gcp_gcs_bucket_prefix
  enable_gcs_versioning = var.gcp_enable_gcs_versioning
}

# Azure Infrastructure Module
module "azure_infrastructure" {
  count  = local.primary_cloud == "azure" || local.secondary_cloud == "azure" ? 1 : 0
  source = "./modules/azure"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  tags        = local.common_tags
  
  # Azure-specific configuration
  resource_group_name = var.azure_resource_group_name
  location           = var.azure_location
  
  # Network configuration
  vpc_cidr = local.vpc_cidr
  
  # Compute configuration
  min_instances              = local.min_instances
  max_instances              = local.max_instances
  target_cpu_utilization     = local.target_cpu_utilization
  vm_size                    = var.azure_vm_size
  
  # Database configuration
  db_sku_name         = var.azure_db_sku_name
  db_storage_mb       = local.db_storage_size * 1024
  db_backup_retention = local.db_backup_retention
  db_password         = random_password.db_password.result
  db_geo_redundant_backup = var.azure_db_geo_redundant_backup
  
  # Security configuration
  enable_encryption_at_rest = local.enable_encryption_at_rest
  key_vault_name           = var.azure_key_vault_name
  
  # Monitoring configuration
  enable_detailed_monitoring = local.enable_detailed_monitoring
  log_retention_days         = local.log_retention_days
  
  # Storage configuration
  storage_account_prefix = var.azure_storage_account_prefix
  enable_blob_versioning = var.azure_enable_blob_versioning
}

# Kubernetes Infrastructure Module
module "kubernetes_infrastructure" {
  count  = var.enable_kubernetes ? 1 : 0
  source = "./modules/kubernetes"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  
  # Cluster configuration
  cluster_version    = var.k8s_cluster_version
  node_count        = var.k8s_node_count
  node_instance_type = var.k8s_node_instance_type
  
  # Primary cloud cluster endpoint
  cluster_endpoint = (
    local.primary_cloud == "aws" && length(module.aws_infrastructure) > 0 ? 
      module.aws_infrastructure[0].eks_cluster_endpoint :
    local.primary_cloud == "gcp" && length(module.gcp_infrastructure) > 0 ?
      module.gcp_infrastructure[0].gke_cluster_endpoint :
    local.primary_cloud == "azure" && length(module.azure_infrastructure) > 0 ?
      module.azure_infrastructure[0].aks_cluster_endpoint : ""
  )
  
  cluster_ca_certificate = (
    local.primary_cloud == "aws" && length(module.aws_infrastructure) > 0 ? 
      module.aws_infrastructure[0].eks_cluster_ca_certificate :
    local.primary_cloud == "gcp" && length(module.gcp_infrastructure) > 0 ?
      module.gcp_infrastructure[0].gke_cluster_ca_certificate :
    local.primary_cloud == "azure" && length(module.azure_infrastructure) > 0 ?
      module.azure_infrastructure[0].aks_cluster_ca_certificate : ""
  )
  
  # Depends on cloud infrastructure
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Monitoring Infrastructure Module
module "monitoring_infrastructure" {
  count  = var.enable_monitoring ? 1 : 0
  source = "./modules/monitoring"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  
  # Monitoring configuration
  enable_prometheus = var.enable_prometheus
  enable_grafana   = var.enable_grafana
  enable_alertmanager = var.enable_alertmanager
  enable_jaeger    = var.enable_jaeger
  
  # Log retention
  log_retention_days = local.log_retention_days
  
  # Alerting configuration
  alert_channels = var.alert_channels
  critical_alert_channels = var.critical_alert_channels
  
  # Depends on infrastructure
  depends_on = [
    module.kubernetes_infrastructure
  ]
}

# Security Infrastructure Module
module "security_infrastructure" {
  count  = var.enable_security_scanning ? 1 : 0
  source = "./modules/security"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  
  # Security scanning configuration
  enable_vulnerability_scanning = var.enable_vulnerability_scanning
  enable_compliance_scanning   = var.enable_compliance_scanning
  enable_malware_scanning      = var.enable_malware_scanning
  
  # Compliance frameworks
  compliance_frameworks = var.compliance_frameworks
  
  # Security policies
  security_policies = var.security_policies
  
  # Depends on infrastructure
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Cost Optimization Module
module "cost_optimization" {
  count  = var.enable_cost_optimization ? 1 : 0
  source = "./modules/cost-optimization"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  
  # Cost optimization configuration
  enable_right_sizing     = var.enable_right_sizing
  enable_spot_instances   = var.enable_spot_instances
  enable_reserved_instances = var.enable_reserved_instances
  
  # Budget configuration
  monthly_budget_limit = var.monthly_budget_limit
  budget_alert_thresholds = var.budget_alert_thresholds
  
  # Cost allocation tags
  cost_allocation_tags = var.cost_allocation_tags
  
  # Depends on infrastructure
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Disaster Recovery Module
module "disaster_recovery" {
  count  = var.enable_disaster_recovery ? 1 : 0
  source = "./modules/disaster-recovery"
  
  # Basic configuration
  environment = local.environment
  project     = local.project
  name_prefix = local.name_prefix
  
  # DR configuration
  backup_schedule = var.backup_schedule
  backup_retention_days = var.backup_retention_days
  cross_region_backup = var.cross_region_backup
  
  # Recovery objectives
  rto_minutes = var.rto_minutes
  rpo_minutes = var.rpo_minutes
  
  # Failover configuration
  enable_automatic_failover = var.enable_automatic_failover
  failover_threshold = var.failover_threshold
  
  # Depends on infrastructure
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}