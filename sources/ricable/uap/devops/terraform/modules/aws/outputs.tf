# AWS Module Outputs

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "subnet_ids" {
  description = "Subnet IDs"
  value       = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "security_group_id" {
  description = "Web security group ID"
  value       = aws_security_group.web.id
}

output "database_security_group_id" {
  description = "Database security group ID"
  value       = aws_security_group.database.id
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Load balancer zone ID"
  value       = aws_lb.main.zone_id
}

output "load_balancer_arn" {
  description = "Load balancer ARN"
  value       = aws_lb.main.arn
}

output "auto_scaling_group_name" {
  description = "Auto scaling group name"
  value       = aws_autoscaling_group.web.name
}

output "auto_scaling_group_arn" {
  description = "Auto scaling group ARN"
  value       = aws_autoscaling_group.web.arn
}

output "launch_template_id" {
  description = "Launch template ID"
  value       = aws_launch_template.web.id
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS database port"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "RDS database name"
  value       = aws_db_instance.main.db_name
}

output "database_connection_string" {
  description = "Database connection string"
  value       = "postgresql://uapuser:${var.db_password}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${aws_db_instance.main.db_name}"
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.main.bucket
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.main.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.main.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN"
  value       = aws_cloudwatch_log_group.main.arn
}

# EKS Outputs (conditional)
output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = var.enable_kubernetes ? aws_eks_cluster.main[0].name : null
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = var.enable_kubernetes ? aws_eks_cluster.main[0].endpoint : null
}

output "eks_cluster_ca_certificate" {
  description = "EKS cluster CA certificate"
  value       = var.enable_kubernetes ? base64decode(aws_eks_cluster.main[0].certificate_authority[0].data) : null
  sensitive   = true
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = var.enable_kubernetes ? aws_eks_cluster.main[0].arn : null
}

output "eks_cluster_version" {
  description = "EKS cluster version"
  value       = var.enable_kubernetes ? aws_eks_cluster.main[0].version : null
}

# IAM Outputs
output "web_iam_role_arn" {
  description = "Web server IAM role ARN"
  value       = aws_iam_role.web.arn
}

output "web_iam_instance_profile_name" {
  description = "Web server IAM instance profile name"
  value       = aws_iam_instance_profile.web.name
}

output "eks_iam_role_arn" {
  description = "EKS IAM role ARN"
  value       = var.enable_kubernetes ? aws_iam_role.eks[0].arn : null
}

# Monitoring Outputs
output "scale_up_policy_arn" {
  description = "Scale up policy ARN"
  value       = aws_autoscaling_policy.scale_up.arn
}

output "scale_down_policy_arn" {
  description = "Scale down policy ARN"
  value       = aws_autoscaling_policy.scale_down.arn
}

output "cpu_high_alarm_arn" {
  description = "CPU high alarm ARN"
  value       = aws_cloudwatch_metric_alarm.cpu_high.arn
}

output "cpu_low_alarm_arn" {
  description = "CPU low alarm ARN"
  value       = aws_cloudwatch_metric_alarm.cpu_low.arn
}

# Network Outputs
output "internet_gateway_id" {
  description = "Internet gateway ID"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "NAT gateway IDs"
  value       = aws_nat_gateway.main[*].id
}

output "route_table_ids" {
  description = "Route table IDs"
  value = {
    public  = aws_route_table.public.id
    private = aws_route_table.private[*].id
  }
}

# Resource Summary
output "resource_summary" {
  description = "Summary of created AWS resources"
  value = {
    vpc_id                = aws_vpc.main.id
    public_subnets        = length(aws_subnet.public)
    private_subnets       = length(aws_subnet.private)
    nat_gateways         = length(aws_nat_gateway.main)
    load_balancer_dns    = aws_lb.main.dns_name
    auto_scaling_group   = aws_autoscaling_group.web.name
    database_endpoint    = aws_db_instance.main.endpoint
    s3_bucket           = aws_s3_bucket.main.bucket
    eks_enabled         = var.enable_kubernetes
    eks_cluster_name    = var.enable_kubernetes ? aws_eks_cluster.main[0].name : null
  }
}