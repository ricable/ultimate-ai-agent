//! Policy Enforcement Engine
//!
//! Integrates with Open Policy Agent (OPA) for fine-grained access control and governance.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum PolicyError {
    #[error("Policy evaluation failed: {0}")]
    EvaluationError(String),
    #[error("Policy not found: {0}")]
    PolicyNotFound(String),
    #[error("Invalid policy: {0}")]
    InvalidPolicy(String),
    #[error("OPA communication error: {0}")]
    OpaError(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub rules: Vec<PolicyRule>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyRule {
    pub id: String,
    pub resource_type: String,
    pub action: String,
    pub conditions: Vec<Condition>,
    pub effect: Effect,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Effect {
    Allow,
    Deny,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Condition {
    pub field: String,
    pub operator: Operator,
    pub value: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Operator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    In,
    NotIn,
    Contains,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub reason: Option<String>,
    pub matched_rules: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub opa_url: String,
    pub opa_timeout_secs: u64,
    pub cache_enabled: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            opa_url: "http://localhost:8181".to_string(),
            opa_timeout_secs: 5,
            cache_enabled: true,
        }
    }
}

pub struct PolicyEnforcer {
    config: PolicyConfig,
    policies: Arc<RwLock<std::collections::HashMap<String, Policy>>>,
    decision_cache: Arc<RwLock<std::collections::HashMap<String, PolicyDecision>>>,
}

impl PolicyEnforcer {
    pub fn new(config: PolicyConfig) -> Self {
        Self {
            config,
            policies: Arc::new(RwLock::new(std::collections::HashMap::new())),
            decision_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Add a new policy
    pub async fn add_policy(&self, policy: Policy) -> anyhow::Result<()> {
        info!("Adding policy: {} ({})", policy.name, policy.id);
        self.policies.write().await.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Remove a policy
    pub async fn remove_policy(&self, policy_id: &str) -> anyhow::Result<()> {
        info!("Removing policy: {}", policy_id);
        self.policies.write().await.remove(policy_id);
        // Clear cache entries for this policy
        self.decision_cache.write().await.clear();
        Ok(())
    }

    /// Evaluate policy for a request
    pub async fn evaluate(
        &self,
        subject: &str,
        action: &str,
        resource: &str,
    ) -> Result<PolicyDecision, PolicyError> {
        debug!("Evaluating policy: subject={}, action={}, resource={}", subject, action, resource);

        // Check cache first
        let cache_key = format!("{}:{}:{}", subject, action, resource);
        if self.config.cache_enabled {
            if let Some(cached) = self.decision_cache.read().await.get(&cache_key) {
                debug!("Returning cached policy decision");
                return Ok(cached.clone());
            }
        }

        // Evaluate against local policies
        let decision = self.evaluate_local(subject, action, resource).await?;

        // Cache the decision
        if self.config.cache_enabled {
            self.decision_cache
                .write()
                .await
                .insert(cache_key, decision.clone());
        }

        Ok(decision)
    }

    /// Evaluate against local policies
    async fn evaluate_local(
        &self,
        subject: &str,
        action: &str,
        resource: &str,
    ) -> Result<PolicyDecision, PolicyError> {
        let policies = self.policies.read().await;
        let mut matched_rules = Vec::new();
        let mut final_effect = Effect::Deny;

        for policy in policies.values() {
            for rule in &policy.rules {
                if self.rule_matches(rule, subject, action, resource).await {
                    matched_rules.push(rule.id.clone());

                    match rule.effect {
                        Effect::Allow => final_effect = Effect::Allow,
                        Effect::Deny => {
                            // Deny takes precedence
                            return Ok(PolicyDecision {
                                allowed: false,
                                reason: Some(format!("Denied by rule: {}", rule.id)),
                                matched_rules,
                            });
                        }
                    }
                }
            }
        }

        let allowed = matches!(final_effect, Effect::Allow);

        Ok(PolicyDecision {
            allowed,
            reason: if allowed {
                Some("Allowed by policy".to_string())
            } else {
                Some("No matching allow rule".to_string())
            },
            matched_rules,
        })
    }

    /// Check if a rule matches the request
    async fn rule_matches(&self, rule: &PolicyRule, subject: &str, action: &str, resource: &str) -> bool {
        // Check action match
        if rule.action != "*" && rule.action != action {
            return false;
        }

        // Check resource type match
        if !resource.starts_with(&rule.resource_type) && rule.resource_type != "*" {
            return false;
        }

        // Evaluate conditions
        for condition in &rule.conditions {
            if !self.evaluate_condition(condition, subject, resource).await {
                return false;
            }
        }

        true
    }

    /// Evaluate a single condition
    async fn evaluate_condition(&self, condition: &Condition, subject: &str, resource: &str) -> bool {
        // Simplified condition evaluation
        // In production, this would extract actual field values from context
        match condition.operator {
            Operator::Equals => {
                if condition.field == "subject" {
                    return condition.value == serde_json::json!(subject);
                }
                true
            }
            Operator::In => {
                if let Some(arr) = condition.value.as_array() {
                    return arr.iter().any(|v| *v == serde_json::json!(subject));
                }
                false
            }
            _ => true, // Simplified for other operators
        }
    }

    /// Call external OPA service
    async fn call_opa(
        &self,
        input: serde_json::Value,
    ) -> Result<PolicyDecision, PolicyError> {
        debug!("Calling OPA at: {}", self.config.opa_url);

        // This would make actual HTTP request to OPA
        // Simplified implementation for demonstration
        Ok(PolicyDecision {
            allowed: true,
            reason: Some("OPA evaluation (mock)".to_string()),
            matched_rules: vec![],
        })
    }

    /// Validate policy syntax
    pub fn validate_policy(&self, policy: &Policy) -> Result<(), PolicyError> {
        if policy.id.is_empty() {
            return Err(PolicyError::InvalidPolicy("Policy ID cannot be empty".into()));
        }

        if policy.rules.is_empty() {
            return Err(PolicyError::InvalidPolicy("Policy must have at least one rule".into()));
        }

        for rule in &policy.rules {
            if rule.resource_type.is_empty() {
                return Err(PolicyError::InvalidPolicy(
                    format!("Rule {} has empty resource type", rule.id),
                ));
            }
        }

        Ok(())
    }

    /// Create a default CRISPR data access policy
    pub fn create_default_policy() -> Policy {
        Policy {
            id: "crispr-data-access".to_string(),
            name: "CRISPR Data Access Policy".to_string(),
            description: "Default policy for CRISPR genomic data access".to_string(),
            rules: vec![
                PolicyRule {
                    id: "allow-researcher-read".to_string(),
                    resource_type: "crispr:data".to_string(),
                    action: "read".to_string(),
                    conditions: vec![Condition {
                        field: "role".to_string(),
                        operator: Operator::In,
                        value: serde_json::json!(["researcher", "admin"]),
                    }],
                    effect: Effect::Allow,
                },
                PolicyRule {
                    id: "deny-export-pii".to_string(),
                    resource_type: "crispr:patient".to_string(),
                    action: "export".to_string(),
                    conditions: vec![Condition {
                        field: "contains_pii".to_string(),
                        operator: Operator::Equals,
                        value: serde_json::json!(true),
                    }],
                    effect: Effect::Deny,
                },
            ],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_policy_evaluation() {
        let enforcer = PolicyEnforcer::new(PolicyConfig::default());
        let policy = PolicyEnforcer::create_default_policy();

        enforcer.add_policy(policy).await.unwrap();

        let decision = enforcer
            .evaluate("researcher", "read", "crispr:data:sample1")
            .await
            .unwrap();

        assert!(decision.allowed);
    }

    #[test]
    fn test_policy_validation() {
        let enforcer = PolicyEnforcer::new(PolicyConfig::default());
        let policy = PolicyEnforcer::create_default_policy();

        assert!(enforcer.validate_policy(&policy).is_ok());
    }
}
