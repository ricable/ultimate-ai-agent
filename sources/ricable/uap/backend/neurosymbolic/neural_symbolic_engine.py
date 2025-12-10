"""
Agent 35: Neuro-Symbolic AI System
Combines symbolic reasoning with neural networks for explainable AI,
knowledge graph integration, and causal inference.
"""

import asyncio
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import uuid4
from enum import Enum
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """Logical operators for symbolic reasoning"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    EXISTS = "∃"
    FORALL = "∀"


class NodeType(Enum):
    """Types of nodes in knowledge graphs"""
    ENTITY = "entity"
    CONCEPT = "concept"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    RULE = "rule"


@dataclass
class SymbolicRule:
    """Represents a symbolic logical rule"""
    id: str
    premise: List[str]  # Antecedent conditions
    conclusion: str     # Consequent
    confidence: float   # Rule confidence (0-1)
    support: int       # Number of instances supporting this rule
    operator: LogicalOperator
    domain: str
    created_at: datetime
    learned_from: str  # "neural", "symbolic", "hybrid"


@dataclass
class KnowledgeGraphNode:
    """Node in the knowledge graph"""
    id: str
    label: str
    node_type: NodeType
    properties: Dict[str, Any]
    embedding: Optional[np.ndarray]
    certainty: float


@dataclass
class KnowledgeGraphEdge:
    """Edge in the knowledge graph"""
    id: str
    source: str
    target: str
    relation: str
    weight: float
    properties: Dict[str, Any]
    certainty: float


@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    id: str
    cause: str
    effect: str
    strength: float  # Causal strength (0-1)
    mechanism: str   # Description of causal mechanism
    confounders: List[str]  # Potential confounding variables
    evidence: List[str]     # Supporting evidence
    confidence: float


@dataclass
class CounterfactualQuery:
    """Represents a counterfactual reasoning query"""
    id: str
    original_scenario: Dict[str, Any]
    counterfactual_scenario: Dict[str, Any]
    predicted_outcome: Any
    confidence: float
    explanation: str


class SymbolicReasoner:
    """Handles symbolic logical reasoning"""
    
    def __init__(self):
        self.rules: Dict[str, SymbolicRule] = {}
        self.facts: Set[str] = set()
        self.predicates: Dict[str, List[str]] = {}
        
    def add_rule(self, rule: SymbolicRule) -> bool:
        """Add a symbolic rule to the reasoner"""
        try:
            self.rules[rule.id] = rule
            logger.info(f"Added rule: {rule.premise} → {rule.conclusion}")
            return True
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
            return False
    
    def add_fact(self, fact: str) -> bool:
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
        return True
    
    async def forward_chain(self, max_iterations: int = 100) -> List[str]:
        """Forward chaining inference to derive new facts"""
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            facts_added = False
            
            for rule in self.rules.values():
                if self._rule_applicable(rule):
                    if rule.conclusion not in self.facts:
                        self.facts.add(rule.conclusion)
                        new_facts.append(rule.conclusion)
                        facts_added = True
                        logger.info(f"Derived new fact: {rule.conclusion}")
            
            if not facts_added:
                break
                
            iteration += 1
        
        return new_facts
    
    async def backward_chain(self, goal: str) -> Tuple[bool, List[str]]:
        """Backward chaining to prove a goal"""
        if goal in self.facts:
            return True, [goal]
        
        proof_chain = []
        
        # Find rules that conclude the goal
        applicable_rules = [rule for rule in self.rules.values() 
                          if rule.conclusion == goal]
        
        for rule in applicable_rules:
            can_prove_premises = True
            premise_proofs = []
            
            for premise in rule.premise:
                can_prove, premise_proof = await self.backward_chain(premise)
                if can_prove:
                    premise_proofs.extend(premise_proof)
                else:
                    can_prove_premises = False
                    break
            
            if can_prove_premises:
                proof_chain = premise_proofs + [goal]
                return True, proof_chain
        
        return False, []
    
    def _rule_applicable(self, rule: SymbolicRule) -> bool:
        """Check if a rule is applicable given current facts"""
        if rule.operator == LogicalOperator.AND:
            return all(premise in self.facts for premise in rule.premise)
        elif rule.operator == LogicalOperator.OR:
            return any(premise in self.facts for premise in rule.premise)
        else:
            # For other operators, default to AND logic
            return all(premise in self.facts for premise in rule.premise)
    
    async def explain_inference(self, conclusion: str) -> Dict[str, Any]:
        """Generate explanation for how a conclusion was derived"""
        can_prove, proof_chain = await self.backward_chain(conclusion)
        
        if can_prove:
            explanation = {
                "conclusion": conclusion,
                "provable": True,
                "proof_chain": proof_chain,
                "rules_used": [],
                "facts_used": []
            }
            
            # Find which rules were used
            for rule in self.rules.values():
                if rule.conclusion in proof_chain:
                    explanation["rules_used"].append({
                        "rule_id": rule.id,
                        "premise": rule.premise,
                        "conclusion": rule.conclusion,
                        "confidence": rule.confidence
                    })
            
            # Find which facts were used
            explanation["facts_used"] = [fact for fact in proof_chain if fact in self.facts]
            
            return explanation
        else:
            return {
                "conclusion": conclusion,
                "provable": False,
                "explanation": "No valid proof chain found"
            }


class NeuralSymbolicNetwork(nn.Module):
    """Neural network that integrates symbolic reasoning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_rules: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_rules = num_rules
        
        # Neural components
        self.neural_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Symbolic reasoning layer
        self.rule_weights = nn.Parameter(torch.randn(num_rules, hidden_dim))
        self.rule_biases = nn.Parameter(torch.zeros(num_rules))
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim + num_rules, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for rule selection
        self.rule_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, x: torch.Tensor, symbolic_rules: Optional[torch.Tensor] = None):
        """Forward pass combining neural and symbolic components"""
        # Neural encoding
        neural_features = self.neural_encoder(x)
        
        # Symbolic reasoning
        if symbolic_rules is not None:
            # Apply rules to neural features
            rule_activations = torch.sigmoid(
                torch.matmul(neural_features, self.rule_weights.t()) + self.rule_biases
            )
        else:
            rule_activations = torch.zeros(x.size(0), self.num_rules)
        
        # Attention-based rule selection
        attended_features, attention_weights = self.rule_attention(
            neural_features.unsqueeze(0),
            neural_features.unsqueeze(0), 
            neural_features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        
        # Integrate neural and symbolic information
        combined_features = torch.cat([attended_features, rule_activations], dim=1)
        output = self.integration_layer(combined_features)
        
        return output, attention_weights, rule_activations


class KnowledgeGraphEngine:
    """Manages knowledge graph construction and reasoning"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: Dict[str, KnowledgeGraphEdge] = {}
        self.embeddings_dim = 512
        
    def add_node(self, node: KnowledgeGraphNode) -> bool:
        """Add a node to the knowledge graph"""
        try:
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **asdict(node))
            return True
        except Exception as e:
            logger.error(f"Error adding node {node.id}: {e}")
            return False
    
    def add_edge(self, edge: KnowledgeGraphEdge) -> bool:
        """Add an edge to the knowledge graph"""
        try:
            self.edges[edge.id] = edge
            self.graph.add_edge(edge.source, edge.target, **asdict(edge))
            return True
        except Exception as e:
            logger.error(f"Error adding edge {edge.id}: {e}")
            return False
    
    async def find_path(self, source: str, target: str, 
                       max_length: int = 5) -> List[List[str]]:
        """Find paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            return paths[:10]  # Return top 10 paths
        except nx.NetworkXNoPath:
            return []
    
    async def find_related_entities(self, entity_id: str, 
                                  relation_type: str = None,
                                  max_distance: int = 2) -> List[Tuple[str, float]]:
        """Find entities related to a given entity"""
        related = []
        
        # Direct neighbors
        neighbors = list(self.graph.neighbors(entity_id))
        
        for neighbor in neighbors:
            # Get edge data
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if edge_data:
                # Take the first edge if multiple exist
                edge_info = list(edge_data.values())[0]
                if relation_type is None or edge_info.get('relation') == relation_type:
                    weight = edge_info.get('weight', 0.5)
                    related.append((neighbor, weight))
        
        # Second-degree neighbors if max_distance > 1
        if max_distance > 1:
            for neighbor in neighbors:
                second_neighbors = list(self.graph.neighbors(neighbor))
                for second_neighbor in second_neighbors:
                    if second_neighbor != entity_id and second_neighbor not in [r[0] for r in related]:
                        # Reduce weight for second-degree connections
                        related.append((second_neighbor, 0.3))
        
        # Sort by weight/relevance
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:20]  # Return top 20
    
    async def query_subgraph(self, center_entity: str, 
                           radius: int = 2) -> Dict[str, Any]:
        """Extract a subgraph around a center entity"""
        subgraph_nodes = set([center_entity])
        
        # BFS to find nodes within radius
        current_level = {center_entity}
        for _ in range(radius):
            next_level = set()
            for node in current_level:
                neighbors = set(self.graph.neighbors(node))
                next_level.update(neighbors)
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Extract subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        return {
            "center_entity": center_entity,
            "radius": radius,
            "nodes": list(subgraph.nodes()),
            "edges": list(subgraph.edges()),
            "node_count": len(subgraph.nodes()),
            "edge_count": len(subgraph.edges())
        }
    
    async def semantic_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate semantic similarity between two entities"""
        if entity1 not in self.nodes or entity2 not in self.nodes:
            return 0.0
        
        node1 = self.nodes[entity1]
        node2 = self.nodes[entity2]
        
        # If both have embeddings, use cosine similarity
        if node1.embedding is not None and node2.embedding is not None:
            similarity = cosine_similarity(
                node1.embedding.reshape(1, -1),
                node2.embedding.reshape(1, -1)
            )[0][0]
            return float(similarity)
        
        # Otherwise, use structural similarity
        common_neighbors = set(self.graph.neighbors(entity1)) & set(self.graph.neighbors(entity2))
        total_neighbors = set(self.graph.neighbors(entity1)) | set(self.graph.neighbors(entity2))
        
        if len(total_neighbors) == 0:
            return 0.0
        
        return len(common_neighbors) / len(total_neighbors)


class CausalInferenceEngine:
    """Handles causal inference and counterfactual reasoning"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_relations: Dict[str, CausalRelation] = {}
        self.observed_data: List[Dict[str, Any]] = []
        
    def add_causal_relation(self, relation: CausalRelation) -> bool:
        """Add a causal relation to the system"""
        try:
            self.causal_relations[relation.id] = relation
            self.causal_graph.add_edge(relation.cause, relation.effect, 
                                     weight=relation.strength, **asdict(relation))
            return True
        except Exception as e:
            logger.error(f"Error adding causal relation: {e}")
            return False
    
    async def infer_causality(self, cause_var: str, effect_var: str,
                            data: List[Dict[str, Any]]) -> CausalRelation:
        """Infer causal relationship between two variables"""
        # Simple causal inference using correlation and temporal precedence
        # In practice, would use more sophisticated methods like IV, RDD, etc.
        
        correlation = self._calculate_correlation(cause_var, effect_var, data)
        temporal_precedence = self._check_temporal_precedence(cause_var, effect_var, data)
        confounders = self._identify_confounders(cause_var, effect_var, data)
        
        # Causal strength estimation (simplified)
        causal_strength = abs(correlation) * (0.8 if temporal_precedence else 0.4)
        
        relation = CausalRelation(
            id=str(uuid4()),
            cause=cause_var,
            effect=effect_var,
            strength=causal_strength,
            mechanism=f"Statistical association with strength {causal_strength:.2f}",
            confounders=confounders,
            evidence=[f"Correlation: {correlation:.3f}", 
                     f"Temporal precedence: {temporal_precedence}"],
            confidence=min(causal_strength + 0.2, 1.0)
        )
        
        return relation
    
    async def counterfactual_reasoning(self, query: CounterfactualQuery) -> CounterfactualQuery:
        """Perform counterfactual reasoning"""
        # Get relevant causal relations
        relevant_relations = self._get_relevant_causal_relations(
            query.original_scenario, query.counterfactual_scenario
        )
        
        # Simulate counterfactual outcome
        predicted_outcome = await self._simulate_counterfactual(
            query.counterfactual_scenario, relevant_relations
        )
        
        # Generate explanation
        explanation = self._generate_counterfactual_explanation(
            query.original_scenario, query.counterfactual_scenario, 
            predicted_outcome, relevant_relations
        )
        
        query.predicted_outcome = predicted_outcome
        query.explanation = explanation
        query.confidence = self._calculate_counterfactual_confidence(relevant_relations)
        
        return query
    
    async def identify_causal_paths(self, cause: str, effect: str) -> List[List[str]]:
        """Identify causal paths between cause and effect"""
        try:
            paths = list(nx.all_simple_paths(self.causal_graph, cause, effect, cutoff=5))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def _calculate_correlation(self, var1: str, var2: str, 
                             data: List[Dict[str, Any]]) -> float:
        """Calculate correlation between two variables"""
        values1 = [d.get(var1, 0) for d in data if var1 in d and var2 in d]
        values2 = [d.get(var2, 0) for d in data if var1 in d and var2 in d]
        
        if len(values1) < 2:
            return 0.0
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    def _check_temporal_precedence(self, cause_var: str, effect_var: str,
                                  data: List[Dict[str, Any]]) -> bool:
        """Check if cause typically precedes effect temporally"""
        # Simplified temporal check
        # In practice, would analyze timestamps and temporal sequences
        return True  # Placeholder
    
    def _identify_confounders(self, cause_var: str, effect_var: str,
                            data: List[Dict[str, Any]]) -> List[str]:
        """Identify potential confounding variables"""
        confounders = []
        
        # Look for variables correlated with both cause and effect
        if data:
            all_vars = set()
            for d in data:
                all_vars.update(d.keys())
            
            for var in all_vars:
                if var != cause_var and var != effect_var:
                    corr_cause = abs(self._calculate_correlation(var, cause_var, data))
                    corr_effect = abs(self._calculate_correlation(var, effect_var, data))
                    
                    # If variable is correlated with both, it might be a confounder
                    if corr_cause > 0.3 and corr_effect > 0.3:
                        confounders.append(var)
        
        return confounders[:5]  # Return top 5 potential confounders
    
    def _get_relevant_causal_relations(self, original: Dict[str, Any],
                                     counterfactual: Dict[str, Any]) -> List[CausalRelation]:
        """Get causal relations relevant to the counterfactual scenario"""
        relevant = []
        
        # Find variables that changed between scenarios
        changed_vars = set()
        for key in original.keys() | counterfactual.keys():
            if original.get(key) != counterfactual.get(key):
                changed_vars.add(key)
        
        # Get relations involving changed variables
        for relation in self.causal_relations.values():
            if relation.cause in changed_vars or relation.effect in changed_vars:
                relevant.append(relation)
        
        return relevant
    
    async def _simulate_counterfactual(self, scenario: Dict[str, Any],
                                     relations: List[CausalRelation]) -> Dict[str, Any]:
        """Simulate the outcome of a counterfactual scenario"""
        outcome = scenario.copy()
        
        # Apply causal relations to propagate effects
        for relation in relations:
            if relation.cause in scenario:
                # Simple linear effect (would be more sophisticated in practice)
                current_effect = outcome.get(relation.effect, 0)
                causal_effect = scenario[relation.cause] * relation.strength
                outcome[relation.effect] = current_effect + causal_effect
        
        return outcome
    
    def _generate_counterfactual_explanation(self, original: Dict[str, Any],
                                           counterfactual: Dict[str, Any],
                                           outcome: Dict[str, Any],
                                           relations: List[CausalRelation]) -> str:
        """Generate explanation for counterfactual reasoning"""
        explanations = []
        
        # Identify key changes
        for key in original.keys() | counterfactual.keys():
            if original.get(key) != counterfactual.get(key):
                explanations.append(f"Changed {key} from {original.get(key)} to {counterfactual.get(key)}")
        
        # Identify causal effects
        for relation in relations:
            if relation.cause in counterfactual:
                explanations.append(f"Due to causal relation, {relation.cause} affects {relation.effect} with strength {relation.strength:.2f}")
        
        return "; ".join(explanations)
    
    def _calculate_counterfactual_confidence(self, relations: List[CausalRelation]) -> float:
        """Calculate confidence in counterfactual prediction"""
        if not relations:
            return 0.5
        
        avg_confidence = sum(r.confidence for r in relations) / len(relations)
        return avg_confidence


class RuleExtractor:
    """Extracts symbolic rules from neural networks"""
    
    def __init__(self):
        self.extracted_rules: List[SymbolicRule] = []
        
    async def extract_rules_from_neural_network(self, model: NeuralSymbolicNetwork,
                                              data_loader,
                                              threshold: float = 0.8) -> List[SymbolicRule]:
        """Extract interpretable rules from trained neural network"""
        rules = []
        model.eval()
        
        activation_patterns = []
        rule_activations_list = []
        
        # Collect activation patterns
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                outputs, attention_weights, rule_activations = model(batch_data)
                activation_patterns.append((batch_data, batch_labels, outputs))
                rule_activations_list.append(rule_activations)
        
        # Analyze rule activations to extract symbolic rules
        all_rule_activations = torch.cat(rule_activations_list, dim=0)
        
        for rule_idx in range(model.num_rules):
            rule_activations = all_rule_activations[:, rule_idx]
            
            # Find patterns where this rule is highly activated
            high_activation_indices = torch.where(rule_activations > threshold)[0]
            
            if len(high_activation_indices) > 5:  # Minimum support
                # Analyze input patterns for these high activations
                premises = self._analyze_input_patterns(
                    activation_patterns, high_activation_indices
                )
                
                conclusions = self._analyze_output_patterns(
                    activation_patterns, high_activation_indices
                )
                
                if premises and conclusions:
                    rule = SymbolicRule(
                        id=f"neural_rule_{rule_idx}",
                        premise=premises,
                        conclusion=conclusions[0],  # Take most common conclusion
                        confidence=float(rule_activations[high_activation_indices].mean()),
                        support=len(high_activation_indices),
                        operator=LogicalOperator.AND,
                        domain="neural_extracted",
                        created_at=datetime.utcnow(),
                        learned_from="neural"
                    )
                    rules.append(rule)
        
        self.extracted_rules.extend(rules)
        return rules
    
    def _analyze_input_patterns(self, activation_patterns, indices) -> List[str]:
        """Analyze input patterns to extract rule premises"""
        premises = []
        
        # Collect input data for high-activation samples
        inputs = []
        for i in indices:
            batch_idx = 0
            sample_idx = i
            
            # Find which batch this index belongs to
            for batch_data, _, _ in activation_patterns:
                if sample_idx < len(batch_data):
                    inputs.append(batch_data[sample_idx])
                    break
                sample_idx -= len(batch_data)
        
        if inputs:
            # Analyze common patterns in inputs
            inputs_tensor = torch.stack(inputs)
            
            # Find features that are commonly active
            mean_activation = inputs_tensor.mean(dim=0)
            std_activation = inputs_tensor.std(dim=0)
            
            # Features with high mean and low std are good candidates for premises
            significant_features = torch.where(
                (mean_activation > 0.5) & (std_activation < 0.2)
            )[0]
            
            for feature_idx in significant_features[:5]:  # Top 5 features
                premises.append(f"feature_{feature_idx}_high")
        
        return premises
    
    def _analyze_output_patterns(self, activation_patterns, indices) -> List[str]:
        """Analyze output patterns to extract rule conclusions"""
        conclusions = []
        
        # Collect output data for high-activation samples
        outputs = []
        for i in indices:
            sample_idx = i
            
            for _, _, batch_outputs in activation_patterns:
                if sample_idx < len(batch_outputs):
                    outputs.append(batch_outputs[sample_idx])
                    break
                sample_idx -= len(batch_outputs)
        
        if outputs:
            outputs_tensor = torch.stack(outputs)
            
            # Find most common output class/pattern
            if outputs_tensor.dim() > 1:
                predicted_classes = outputs_tensor.argmax(dim=1)
                unique_classes, counts = torch.unique(predicted_classes, return_counts=True)
                most_common_class = unique_classes[counts.argmax()]
                conclusions.append(f"class_{most_common_class}")
            else:
                # Regression case
                mean_output = outputs_tensor.mean()
                conclusions.append(f"output_approx_{mean_output:.2f}")
        
        return conclusions


class NeuroSymbolicEngine:
    """Main engine coordinating all neuro-symbolic components"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 output_dim: int = 10, num_rules: int = 50):
        self.symbolic_reasoner = SymbolicReasoner()
        self.knowledge_graph = KnowledgeGraphEngine()
        self.causal_engine = CausalInferenceEngine()
        self.rule_extractor = RuleExtractor()
        
        # Neural components
        self.neural_network = NeuralSymbolicNetwork(
            input_dim, hidden_dim, output_dim, num_rules
        )
        
        # Integration state
        self.trained = False
        self.reasoning_cache: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Initialize the neuro-symbolic engine"""
        logger.info("Initializing Neuro-Symbolic AI Engine")
        
        # Add some default rules
        default_rules = self._create_default_rules()
        for rule in default_rules:
            self.symbolic_reasoner.add_rule(rule)
        
        return True
    
    async def hybrid_reasoning(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid neural-symbolic reasoning"""
        query_id = str(uuid4())
        
        # Check cache first
        cache_key = str(hash(json.dumps(query, sort_keys=True)))
        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        
        result = {
            "query_id": query_id,
            "query": query,
            "neural_output": None,
            "symbolic_reasoning": None,
            "knowledge_graph_context": None,
            "causal_analysis": None,
            "final_answer": None,
            "explanation": None,
            "confidence": 0.0
        }
        
        # Neural processing
        if "input_features" in query:
            neural_output = await self._neural_inference(query["input_features"])
            result["neural_output"] = neural_output
        
        # Symbolic reasoning
        if "goal" in query:
            symbolic_result = await self.symbolic_reasoner.explain_inference(query["goal"])
            result["symbolic_reasoning"] = symbolic_result
        
        # Knowledge graph context
        if "entity" in query:
            kg_context = await self.knowledge_graph.query_subgraph(query["entity"])
            result["knowledge_graph_context"] = kg_context
        
        # Causal analysis
        if "cause" in query and "effect" in query:
            causal_paths = await self.causal_engine.identify_causal_paths(
                query["cause"], query["effect"]
            )
            result["causal_analysis"] = {"paths": causal_paths}
        
        # Integrate results
        final_answer, explanation, confidence = await self._integrate_reasoning_results(result)
        result["final_answer"] = final_answer
        result["explanation"] = explanation
        result["confidence"] = confidence
        
        # Cache result
        self.reasoning_cache[cache_key] = result
        
        return result
    
    async def learn_from_data(self, training_data, validation_data, epochs: int = 10):
        """Train the neural-symbolic system on data"""
        logger.info("Training neural-symbolic system")
        
        # Train neural network
        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.neural_network.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, batch_labels in training_data:
                optimizer.zero_grad()
                
                outputs, attention_weights, rule_activations = self.neural_network(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Add regularization for rule sparsity
                rule_sparsity_loss = 0.01 * torch.mean(torch.abs(rule_activations))
                total_loss = loss + rule_sparsity_loss
                
                total_loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        
        # Extract rules from trained network
        extracted_rules = await self.rule_extractor.extract_rules_from_neural_network(
            self.neural_network, validation_data
        )
        
        # Add extracted rules to symbolic reasoner
        for rule in extracted_rules:
            self.symbolic_reasoner.add_rule(rule)
        
        self.trained = True
        logger.info(f"Training complete. Extracted {len(extracted_rules)} rules.")
    
    async def generate_explanation(self, input_data, output) -> str:
        """Generate human-readable explanation for a prediction"""
        explanations = []
        
        # Neural explanation via attention
        if self.trained:
            with torch.no_grad():
                _, attention_weights, rule_activations = self.neural_network(input_data)
                
                # Find most activated rules
                top_rules = torch.topk(rule_activations.mean(dim=0), k=3)
                for rule_idx in top_rules.indices:
                    if rule_idx < len(self.rule_extractor.extracted_rules):
                        rule = self.rule_extractor.extracted_rules[rule_idx]
                        explanations.append(f"Rule: {' AND '.join(rule.premise)} → {rule.conclusion}")
        
        # Symbolic explanation
        if hasattr(output, 'item'):
            output_class = f"class_{output.argmax().item()}"
            symbolic_explanation = await self.symbolic_reasoner.explain_inference(output_class)
            if symbolic_explanation["provable"]:
                explanations.append(f"Symbolic reasoning: {symbolic_explanation['explanation']}")
        
        return "; ".join(explanations) if explanations else "No clear explanation available"
    
    async def counterfactual_explanation(self, original_input, counterfactual_input) -> str:
        """Generate counterfactual explanation"""
        query = CounterfactualQuery(
            id=str(uuid4()),
            original_scenario={"input": original_input},
            counterfactual_scenario={"input": counterfactual_input},
            predicted_outcome=None,
            confidence=0.0,
            explanation=""
        )
        
        result = await self.causal_engine.counterfactual_reasoning(query)
        return result.explanation
    
    async def _neural_inference(self, input_features):
        """Perform neural inference"""
        if not self.trained:
            return {"error": "Model not trained"}
        
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            outputs, attention_weights, rule_activations = self.neural_network(input_tensor)
            
        return {
            "predictions": outputs.cpu().numpy().tolist(),
            "attention_weights": attention_weights.cpu().numpy().tolist(),
            "rule_activations": rule_activations.cpu().numpy().tolist()
        }
    
    async def _integrate_reasoning_results(self, results: Dict[str, Any]) -> Tuple[Any, str, float]:
        """Integrate neural and symbolic reasoning results"""
        explanations = []
        confidences = []
        
        # Extract information from different reasoning modes
        if results["neural_output"]:
            neural_conf = max(results["neural_output"]["predictions"][0]) if results["neural_output"]["predictions"] else 0.5
            confidences.append(neural_conf)
            explanations.append("Neural network prediction")
        
        if results["symbolic_reasoning"] and results["symbolic_reasoning"]["provable"]:
            confidences.append(0.9)  # High confidence for provable symbolic reasoning
            explanations.append("Symbolic logical inference")
        
        if results["knowledge_graph_context"]:
            kg_conf = min(1.0, results["knowledge_graph_context"]["node_count"] / 10.0)
            confidences.append(kg_conf)
            explanations.append("Knowledge graph context")
        
        # Aggregate confidence
        final_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Generate final answer (simplified integration)
        if results["neural_output"]:
            final_answer = results["neural_output"]["predictions"]
        else:
            final_answer = "Unable to determine"
        
        explanation = f"Integrated reasoning using: {', '.join(explanations)}"
        
        return final_answer, explanation, final_confidence
    
    def _create_default_rules(self) -> List[SymbolicRule]:
        """Create some default symbolic rules"""
        rules = []
        
        # Example rule: If it's a mammal and it lives in water, it might be a whale
        rule1 = SymbolicRule(
            id="default_rule_1",
            premise=["is_mammal", "lives_in_water"],
            conclusion="is_whale",
            confidence=0.8,
            support=100,
            operator=LogicalOperator.AND,
            domain="biology",
            created_at=datetime.utcnow(),
            learned_from="symbolic"
        )
        rules.append(rule1)
        
        # Example rule: If accuracy > 0.9 and precision > 0.9, then model is good
        rule2 = SymbolicRule(
            id="default_rule_2",
            premise=["high_accuracy", "high_precision"],
            conclusion="good_model",
            confidence=0.9,
            support=200,
            operator=LogicalOperator.AND,
            domain="machine_learning",
            created_at=datetime.utcnow(),
            learned_from="symbolic"
        )
        rules.append(rule2)
        
        return rules
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "neural_network": {
                "trained": self.trained,
                "input_dim": self.neural_network.input_dim,
                "output_dim": self.neural_network.output_dim,
                "num_rules": self.neural_network.num_rules
            },
            "symbolic_reasoner": {
                "num_rules": len(self.symbolic_reasoner.rules),
                "num_facts": len(self.symbolic_reasoner.facts)
            },
            "knowledge_graph": {
                "num_nodes": len(self.knowledge_graph.nodes),
                "num_edges": len(self.knowledge_graph.edges)
            },
            "causal_engine": {
                "num_causal_relations": len(self.causal_engine.causal_relations)
            },
            "rule_extractor": {
                "num_extracted_rules": len(self.rule_extractor.extracted_rules)
            },
            "cache_size": len(self.reasoning_cache)
        }