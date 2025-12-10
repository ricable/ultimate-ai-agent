# AWS Module Variables

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project" {
  description = "Project name"
  type        = string
}

variable "name_prefix" {
  description = "Name prefix for resources"
  type        = string
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

# Compute Configuration
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_pair_name" {
  description = "EC2 key pair name"
  type        = string
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
}

variable "target_cpu_utilization" {
  description = "Target CPU utilization for auto-scaling"
  type        = number
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
}

variable "db_storage_size" {
  description = "RDS storage size in GB"
  type        = number
}

variable "db_backup_retention" {
  description = "RDS backup retention in days"
  type        = number
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_multi_az" {
  description = "Enable RDS Multi-AZ"
  type        = bool
}

# Load Balancer Configuration
variable "load_balancer_type" {
  description = "Load balancer type"
  type        = string
}

variable "enable_cross_zone_load_balancing" {
  description = "Enable cross-zone load balancing"
  type        = bool
}

# Security Configuration
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest"
  type        = bool
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
}

variable "enable_waf" {
  description = "Enable WAF"
  type        = bool
}

variable "kms_key_id" {
  description = "KMS key ID for encryption"
  type        = string
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring"
  type        = bool
}

variable "log_retention_days" {
  description = "Log retention in days"
  type        = number
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  type        = string
}

# Storage Configuration
variable "s3_bucket_prefix" {
  description = "S3 bucket name prefix"
  type        = string
}

variable "enable_s3_versioning" {
  description = "Enable S3 versioning"
  type        = bool
}

variable "s3_lifecycle_rules" {
  description = "S3 lifecycle rules"
  type = list(object({
    id     = string
    status = string
    transitions = list(object({
      days          = number
      storage_class = string
    }))
  }))
}

# Kubernetes Configuration
variable "enable_kubernetes" {
  description = "Enable EKS cluster"
  type        = bool
  default     = false
}

variable "k8s_cluster_version" {
  description = "EKS cluster version"
  type        = string
  default     = "1.27"
}