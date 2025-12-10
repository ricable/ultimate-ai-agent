# UAP Infrastructure Variables
# Configuration variables for multi-cloud deployment

# Basic Configuration
variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  default     = "development"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "primary_cloud" {
  description = "Primary cloud provider (aws, gcp, azure)"
  type        = string
  default     = "aws"
  
  validation {
    condition     = contains(["aws", "gcp", "azure"], var.primary_cloud)
    error_message = "Primary cloud must be aws, gcp, or azure."
  }
}

variable "secondary_cloud" {
  description = "Secondary cloud provider for disaster recovery"
  type        = string
  default     = ""
  
  validation {
    condition = var.secondary_cloud == "" || contains(["aws", "gcp", "azure"], var.secondary_cloud)
    error_message = "Secondary cloud must be empty or one of: aws, gcp, azure."
  }
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

# Auto-scaling Configuration
variable "min_instances" {
  description = "Minimum number of instances in auto-scaling group"
  type        = number
  default     = 2
  
  validation {
    condition     = var.min_instances >= 1 && var.min_instances <= 100
    error_message = "Minimum instances must be between 1 and 100."
  }
}

variable "max_instances" {
  description = "Maximum number of instances in auto-scaling group"
  type        = number
  default     = 10
  
  validation {
    condition     = var.max_instances >= var.min_instances && var.max_instances <= 1000
    error_message = "Maximum instances must be greater than minimum instances and less than 1000."
  }
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization percentage for auto-scaling"
  type        = number
  default     = 70
  
  validation {
    condition     = var.target_cpu_utilization >= 10 && var.target_cpu_utilization <= 90
    error_message = "Target CPU utilization must be between 10 and 90 percent."
  }
}

# Database Configuration
variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_storage_size" {
  description = "Database storage size in GB"
  type        = number
  default     = 100
  
  validation {
    condition     = var.db_storage_size >= 20 && var.db_storage_size <= 65536
    error_message = "Database storage size must be between 20 and 65536 GB."
  }
}

variable "db_backup_retention" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
  
  validation {
    condition     = var.db_backup_retention >= 1 && var.db_backup_retention <= 35
    error_message = "Database backup retention must be between 1 and 35 days."
  }
}

# Load Balancing Configuration
variable "load_balancer_type" {
  description = "Load balancer type (application, network)"
  type        = string
  default     = "application"
  
  validation {
    condition     = contains(["application", "network"], var.load_balancer_type)
    error_message = "Load balancer type must be application or network."
  }
}

variable "enable_cross_zone_load_balancing" {
  description = "Enable cross-zone load balancing"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.log_retention_days >= 1 && var.log_retention_days <= 3653
    error_message = "Log retention days must be between 1 and 3653."
  }
}

# Security Configuration
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable Web Application Firewall"
  type        = bool
  default     = true
}

# AWS-specific Variables
variable "aws_instance_type" {
  description = "AWS EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "aws_key_pair_name" {
  description = "AWS key pair name for EC2 instances"
  type        = string
  default     = ""
}

variable "aws_db_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "aws_kms_key_id" {
  description = "AWS KMS key ID for encryption"
  type        = string
  default     = ""
}

variable "aws_sns_topic_arn" {
  description = "AWS SNS topic ARN for alerts"
  type        = string
  default     = ""
}

variable "aws_s3_bucket_prefix" {
  description = "S3 bucket name prefix"
  type        = string
  default     = "uap"
}

variable "aws_enable_s3_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "aws_s3_lifecycle_rules" {
  description = "S3 lifecycle rules"
  type = list(object({
    id     = string
    status = string
    transitions = list(object({
      days          = number
      storage_class = string
    }))
  }))
  default = [
    {
      id     = "standard_to_ia"
      status = "Enabled"
      transitions = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]
    }
  ]
}

# GCP-specific Variables
variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_zones" {
  description = "GCP zones"
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

variable "gcp_machine_type" {
  description = "GCP machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "gcp_db_tier" {
  description = "GCP Cloud SQL database tier"
  type        = string
  default     = "db-g1-small"
}

variable "gcp_db_high_availability" {
  description = "Enable GCP Cloud SQL high availability"
  type        = bool
  default     = true
}

variable "gcp_kms_key_ring" {
  description = "GCP KMS key ring name"
  type        = string
  default     = "uap-keyring"
}

variable "gcp_kms_crypto_key" {
  description = "GCP KMS crypto key name"
  type        = string
  default     = "uap-key"
}

variable "gcp_gcs_bucket_prefix" {
  description = "GCS bucket name prefix"
  type        = string
  default     = "uap"
}

variable "gcp_enable_gcs_versioning" {
  description = "Enable GCS bucket versioning"
  type        = bool
  default     = true
}

# Azure-specific Variables
variable "azure_resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = "uap-rg"
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "West US 2"
}

variable "azure_vm_size" {
  description = "Azure VM size"
  type        = string
  default     = "Standard_B2s"
}

variable "azure_db_sku_name" {
  description = "Azure Database SKU name"
  type        = string
  default     = "GP_Gen5_2"
}

variable "azure_db_geo_redundant_backup" {
  description = "Enable Azure Database geo-redundant backup"
  type        = bool
  default     = true
}

variable "azure_key_vault_name" {
  description = "Azure Key Vault name"
  type        = string
  default     = "uap-keyvault"
}

variable "azure_storage_account_prefix" {
  description = "Azure Storage Account name prefix"
  type        = string
  default     = "uap"
}

variable "azure_enable_blob_versioning" {
  description = "Enable Azure Blob versioning"
  type        = bool
  default     = true
}

# Kubernetes Configuration
variable "enable_kubernetes" {
  description = "Enable Kubernetes cluster deployment"
  type        = bool
  default     = true
}

variable "k8s_cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.27"
}

variable "k8s_node_count" {
  description = "Number of Kubernetes nodes"
  type        = number
  default     = 3
  
  validation {
    condition     = var.k8s_node_count >= 1 && var.k8s_node_count <= 100
    error_message = "Kubernetes node count must be between 1 and 100."
  }
}

variable "k8s_node_instance_type" {
  description = "Kubernetes node instance type"
  type        = string
  default     = "t3.medium"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable monitoring stack deployment"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_alertmanager" {
  description = "Enable Alertmanager for alerts"
  type        = bool
  default     = true
}

variable "enable_jaeger" {
  description = "Enable Jaeger distributed tracing"
  type        = bool
  default     = true
}

variable "alert_channels" {
  description = "Alert notification channels"
  type = list(object({
    type   = string
    config = map(string)
  }))
  default = [
    {
      type = "slack"
      config = {
        webhook_url = ""
        channel     = "#alerts"
      }
    }
  ]
}

variable "critical_alert_channels" {
  description = "Critical alert notification channels"
  type = list(object({
    type   = string
    config = map(string)
  }))
  default = [
    {
      type = "pagerduty"
      config = {
        service_key = ""
      }
    }
  ]
}

# Security Configuration
variable "enable_security_scanning" {
  description = "Enable security scanning and compliance"
  type        = bool
  default     = true
}

variable "enable_vulnerability_scanning" {
  description = "Enable vulnerability scanning"
  type        = bool
  default     = true
}

variable "enable_compliance_scanning" {
  description = "Enable compliance scanning"
  type        = bool
  default     = true
}

variable "enable_malware_scanning" {
  description = "Enable malware scanning"
  type        = bool
  default     = true
}

variable "compliance_frameworks" {
  description = "Compliance frameworks to monitor"
  type        = list(string)
  default     = ["SOC2", "GDPR", "HIPAA", "ISO27001"]
}

variable "security_policies" {
  description = "Security policies to enforce"
  type = list(object({
    name        = string
    description = string
    rules       = list(string)
  }))
  default = [
    {
      name        = "encryption_policy"
      description = "Enforce encryption at rest and in transit"
      rules       = ["require_encryption_at_rest", "require_ssl_tls"]
    }
  ]
}

# Cost Optimization Configuration
variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "enable_right_sizing" {
  description = "Enable resource right-sizing recommendations"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Enable spot instance usage"
  type        = bool
  default     = true
}

variable "enable_reserved_instances" {
  description = "Enable reserved instance recommendations"
  type        = bool
  default     = true
}

variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 10000
  
  validation {
    condition     = var.monthly_budget_limit > 0
    error_message = "Monthly budget limit must be greater than 0."
  }
}

variable "budget_alert_thresholds" {
  description = "Budget alert thresholds as percentages"
  type        = list(number)
  default     = [50, 80, 90, 100]
}

variable "cost_allocation_tags" {
  description = "Tags for cost allocation and tracking"
  type        = map(string)
  default = {
    "CostCenter" = "Engineering"
    "Team"       = "DevOps"
    "Purpose"    = "UAP-Platform"
  }
}

# Disaster Recovery Configuration
variable "enable_disaster_recovery" {
  description = "Enable disaster recovery features"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Backup schedule in cron format"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition     = var.backup_retention_days >= 7 && var.backup_retention_days <= 365
    error_message = "Backup retention days must be between 7 and 365."
  }
}

variable "cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

variable "rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
  
  validation {
    condition     = var.rto_minutes >= 15 && var.rto_minutes <= 1440
    error_message = "RTO must be between 15 minutes and 24 hours."
  }
}

variable "rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
  
  validation {
    condition     = var.rpo_minutes >= 5 && var.rpo_minutes <= 1440
    error_message = "RPO must be between 5 minutes and 24 hours."
  }
}

variable "enable_automatic_failover" {
  description = "Enable automatic failover to secondary region"
  type        = bool
  default     = false
}

variable "failover_threshold" {
  description = "Threshold for automatic failover (error rate percentage)"
  type        = number
  default     = 5
  
  validation {
    condition     = var.failover_threshold >= 1 && var.failover_threshold <= 50
    error_message = "Failover threshold must be between 1 and 50 percent."
  }
}