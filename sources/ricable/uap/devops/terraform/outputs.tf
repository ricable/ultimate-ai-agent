# UAP Infrastructure Outputs
# Export important resource information for use by other systems

# Basic Information
output "environment" {
  description = "Deployment environment"
  value       = var.environment
}

output "primary_cloud" {
  description = "Primary cloud provider"
  value       = var.primary_cloud
}

output "secondary_cloud" {
  description = "Secondary cloud provider"
  value       = var.secondary_cloud
}

# AWS Outputs
output "aws_vpc_id" {
  description = "AWS VPC ID"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].vpc_id : null
}

output "aws_subnet_ids" {
  description = "AWS subnet IDs"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].subnet_ids : []
}

output "aws_security_group_id" {
  description = "AWS security group ID"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].security_group_id : null
}

output "aws_load_balancer_dns" {
  description = "AWS load balancer DNS name"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].load_balancer_dns : null
}

output "aws_load_balancer_zone_id" {
  description = "AWS load balancer zone ID"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].load_balancer_zone_id : null
}

output "aws_auto_scaling_group_name" {
  description = "AWS auto scaling group name"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].auto_scaling_group_name : null
}

output "aws_database_endpoint" {
  description = "AWS RDS database endpoint"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].database_endpoint : null
  sensitive   = true
}

output "aws_database_port" {
  description = "AWS RDS database port"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].database_port : null
}

output "aws_s3_bucket_name" {
  description = "AWS S3 bucket name"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].s3_bucket_name : null
}

output "aws_cloudwatch_log_group" {
  description = "AWS CloudWatch log group name"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cloudwatch_log_group : null
}

output "aws_eks_cluster_name" {
  description = "AWS EKS cluster name"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].eks_cluster_name : null
}

output "aws_eks_cluster_endpoint" {
  description = "AWS EKS cluster endpoint"
  value       = length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].eks_cluster_endpoint : null
}

# GCP Outputs
output "gcp_vpc_name" {
  description = "GCP VPC network name"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].vpc_name : null
}

output "gcp_subnet_names" {
  description = "GCP subnet names"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].subnet_names : []
}

output "gcp_load_balancer_ip" {
  description = "GCP load balancer IP address"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].load_balancer_ip : null
}

output "gcp_instance_group_manager_name" {
  description = "GCP instance group manager name"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].instance_group_manager_name : null
}

output "gcp_database_connection_name" {
  description = "GCP Cloud SQL connection name"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_connection_name : null
  sensitive   = true
}

output "gcp_database_ip_address" {
  description = "GCP Cloud SQL IP address"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_ip_address : null
  sensitive   = true
}

output "gcp_gcs_bucket_name" {
  description = "GCP Cloud Storage bucket name"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].gcs_bucket_name : null
}

output "gcp_gke_cluster_name" {
  description = "GCP GKE cluster name"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].gke_cluster_name : null
}

output "gcp_gke_cluster_endpoint" {
  description = "GCP GKE cluster endpoint"
  value       = length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].gke_cluster_endpoint : null
}

# Azure Outputs
output "azure_resource_group_name" {
  description = "Azure resource group name"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].resource_group_name : null
}

output "azure_vnet_name" {
  description = "Azure virtual network name"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].vnet_name : null
}

output "azure_subnet_ids" {
  description = "Azure subnet IDs"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].subnet_ids : []
}

output "azure_load_balancer_ip" {
  description = "Azure load balancer public IP"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].load_balancer_ip : null
}

output "azure_vmss_name" {
  description = "Azure virtual machine scale set name"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].vmss_name : null
}

output "azure_database_fqdn" {
  description = "Azure Database FQDN"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].database_fqdn : null
  sensitive   = true
}

output "azure_storage_account_name" {
  description = "Azure Storage Account name"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].storage_account_name : null
}

output "azure_aks_cluster_name" {
  description = "Azure AKS cluster name"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].aks_cluster_name : null
}

output "azure_aks_cluster_fqdn" {
  description = "Azure AKS cluster FQDN"
  value       = length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].aks_cluster_fqdn : null
}

# Kubernetes Outputs
output "kubernetes_cluster_name" {
  description = "Kubernetes cluster name"
  value       = length(module.kubernetes_infrastructure) > 0 ? module.kubernetes_infrastructure[0].cluster_name : null
}

output "kubernetes_cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = length(module.kubernetes_infrastructure) > 0 ? module.kubernetes_infrastructure[0].cluster_endpoint : null
}

output "kubernetes_cluster_ca_certificate" {
  description = "Kubernetes cluster CA certificate"
  value       = length(module.kubernetes_infrastructure) > 0 ? module.kubernetes_infrastructure[0].cluster_ca_certificate : null
  sensitive   = true
}

output "kubernetes_config_path" {
  description = "Path to Kubernetes config file"
  value       = length(module.kubernetes_infrastructure) > 0 ? module.kubernetes_infrastructure[0].config_path : null
}

# Monitoring Outputs
output "monitoring_prometheus_endpoint" {
  description = "Prometheus monitoring endpoint"
  value       = length(module.monitoring_infrastructure) > 0 ? module.monitoring_infrastructure[0].prometheus_endpoint : null
}

output "monitoring_grafana_endpoint" {
  description = "Grafana dashboard endpoint"
  value       = length(module.monitoring_infrastructure) > 0 ? module.monitoring_infrastructure[0].grafana_endpoint : null
}

output "monitoring_alertmanager_endpoint" {
  description = "Alertmanager endpoint"
  value       = length(module.monitoring_infrastructure) > 0 ? module.monitoring_infrastructure[0].alertmanager_endpoint : null
}

output "monitoring_jaeger_endpoint" {
  description = "Jaeger tracing endpoint"
  value       = length(module.monitoring_infrastructure) > 0 ? module.monitoring_infrastructure[0].jaeger_endpoint : null
}

# Security Outputs
output "security_vulnerability_scanner_endpoint" {
  description = "Vulnerability scanner endpoint"
  value       = length(module.security_infrastructure) > 0 ? module.security_infrastructure[0].vulnerability_scanner_endpoint : null
}

output "security_compliance_dashboard_endpoint" {
  description = "Compliance dashboard endpoint"
  value       = length(module.security_infrastructure) > 0 ? module.security_infrastructure[0].compliance_dashboard_endpoint : null
}

output "security_policy_violations" {
  description = "Current security policy violations"
  value       = length(module.security_infrastructure) > 0 ? module.security_infrastructure[0].policy_violations : []
}

# Cost Optimization Outputs
output "cost_optimization_recommendations" {
  description = "Current cost optimization recommendations"
  value       = length(module.cost_optimization) > 0 ? module.cost_optimization[0].recommendations : []
}

output "cost_optimization_current_spend" {
  description = "Current monthly spend"
  value       = length(module.cost_optimization) > 0 ? module.cost_optimization[0].current_spend : null
}

output "cost_optimization_projected_savings" {
  description = "Projected monthly savings"
  value       = length(module.cost_optimization) > 0 ? module.cost_optimization[0].projected_savings : null
}

output "cost_optimization_budget_utilization" {
  description = "Current budget utilization percentage"
  value       = length(module.cost_optimization) > 0 ? module.cost_optimization[0].budget_utilization : null
}

# Disaster Recovery Outputs
output "disaster_recovery_backup_locations" {
  description = "Backup storage locations"
  value       = length(module.disaster_recovery) > 0 ? module.disaster_recovery[0].backup_locations : []
}

output "disaster_recovery_last_backup_time" {
  description = "Timestamp of last successful backup"
  value       = length(module.disaster_recovery) > 0 ? module.disaster_recovery[0].last_backup_time : null
}

output "disaster_recovery_failover_endpoint" {
  description = "Disaster recovery failover endpoint"
  value       = length(module.disaster_recovery) > 0 ? module.disaster_recovery[0].failover_endpoint : null
}

output "disaster_recovery_rto_status" {
  description = "Current RTO status"
  value       = length(module.disaster_recovery) > 0 ? module.disaster_recovery[0].rto_status : null
}

output "disaster_recovery_rpo_status" {
  description = "Current RPO status"
  value       = length(module.disaster_recovery) > 0 ? module.disaster_recovery[0].rpo_status : null
}

# Database Connection Information
output "database_connection_string" {
  description = "Database connection string"
  value = (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].database_connection_string :
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].database_connection_string :
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].database_connection_string : null
  )
  sensitive = true
}

# Load Balancer Information
output "load_balancer_endpoint" {
  description = "Primary load balancer endpoint"
  value = (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].load_balancer_dns :
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].load_balancer_ip :
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].load_balancer_ip : null
  )
}

# Storage Information
output "storage_bucket_name" {
  description = "Primary storage bucket name"
  value = (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].s3_bucket_name :
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].gcs_bucket_name :
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].storage_account_name : null
  )
}

# Consolidated Status Information
output "infrastructure_status" {
  description = "Overall infrastructure deployment status"
  value = {
    environment    = var.environment
    primary_cloud  = var.primary_cloud
    secondary_cloud = var.secondary_cloud
    
    aws_deployed    = length(module.aws_infrastructure) > 0
    gcp_deployed    = length(module.gcp_infrastructure) > 0
    azure_deployed  = length(module.azure_infrastructure) > 0
    
    kubernetes_enabled = var.enable_kubernetes
    monitoring_enabled = var.enable_monitoring
    security_enabled   = var.enable_security_scanning
    cost_optimization_enabled = var.enable_cost_optimization
    disaster_recovery_enabled = var.enable_disaster_recovery
    
    deployment_timestamp = timestamp()
  }
}

# Deployment URLs
output "application_urls" {
  description = "Application access URLs"
  value = {
    primary_endpoint = (
      length(module.aws_infrastructure) > 0 ? "https://${module.aws_infrastructure[0].load_balancer_dns}" :
      length(module.gcp_infrastructure) > 0 ? "https://${module.gcp_infrastructure[0].load_balancer_ip}" :
      length(module.azure_infrastructure) > 0 ? "https://${module.azure_infrastructure[0].load_balancer_ip}" : null
    )
    
    monitoring_dashboard = length(module.monitoring_infrastructure) > 0 ? module.monitoring_infrastructure[0].grafana_endpoint : null
    
    security_dashboard = length(module.security_infrastructure) > 0 ? module.security_infrastructure[0].compliance_dashboard_endpoint : null
    
    cost_dashboard = length(module.cost_optimization) > 0 ? module.cost_optimization[0].dashboard_endpoint : null
  }
}

# Health Check Information
output "health_check_endpoints" {
  description = "Health check endpoints for monitoring"
  value = {
    primary_health_check = (
      length(module.aws_infrastructure) > 0 ? "${module.aws_infrastructure[0].load_balancer_dns}/health" :
      length(module.gcp_infrastructure) > 0 ? "${module.gcp_infrastructure[0].load_balancer_ip}/health" :
      length(module.azure_infrastructure) > 0 ? "${module.azure_infrastructure[0].load_balancer_ip}/health" : null
    )
    
    database_health_check = "/api/health/database"
    api_health_check = "/api/health"
    metrics_endpoint = "/metrics"
  }
}