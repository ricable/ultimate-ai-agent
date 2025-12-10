"""
Test suite for Reinforcement Learning Optimizer - Self-improvement through RL.

This module tests the reinforcement learning capabilities for SAFLA optimization:
- Q-learning and policy gradient algorithms
- Strategy optimization and adaptation
- Multi-agent coordination
- Reward function design and evaluation
- Experience replay and memory management

Following TDD principles: Red-Green-Refactor cycle.
"""

import pytest
import numpy as np
import torch
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass

# Import the classes we'll implement
from safla.core.ml_rl_optimizer import (
    RLOptimizer,
    QLearningAgent,
    PolicyGradientAgent,
    MultiAgentCoordinator,
    ExperienceReplay,
    RewardFunction,
    StrategyOptimizer,
    RLConfig,
    AgentState,
    Action,
    Reward,
    Experience,
    OptimizationStrategy,
    PerformanceMetrics
)


@dataclass
class MockEnvironment:
    """Mock environment for RL testing."""
    state_dim: int = 10
    action_dim: int = 4
    max_steps: int = 100
    current_step: int = 0
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        return np.random.rand(self.state_dim)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        self.current_step += 1
        next_state = np.random.rand(self.state_dim)
        reward = np.random.rand() - 0.5  # Random reward between -0.5 and 0.5
        done = self.current_step >= self.max_steps
        info = {"step": self.current_step}
        return next_state, reward, done, info


class TestRLOptimizer:
    """Test suite for Reinforcement Learning Optimizer functionality."""
    
    @pytest.fixture
    def rl_config(self):
        """Create RL configuration for testing."""
        return RLConfig(
            algorithm="dqn",
            state_dim=10,
            action_dim=4,
            learning_rate=1e-3,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32,
            target_update_freq=100,
            max_episodes=1000
        )
    
    @pytest.fixture
    def rl_optimizer(self, rl_config):
        """Create RLOptimizer instance for testing."""
        return RLOptimizer(config=rl_config)
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment for testing."""
        return MockEnvironment()
    
    def test_rl_optimizer_initialization(self, rl_optimizer, rl_config):
        """Test RLOptimizer initialization with correct parameters."""
        assert rl_optimizer.config == rl_config
        assert rl_optimizer.agent is not None
        assert rl_optimizer.experience_replay is not None
        assert rl_optimizer.reward_function is not None
        assert rl_optimizer.total_episodes == 0
        assert rl_optimizer.total_steps == 0
    
    def test_rl_optimizer_invalid_config(self):
        """Test RLOptimizer rejects invalid configuration."""
        with pytest.raises(ValueError, match="State dimension must be positive"):
            invalid_config = RLConfig(
                algorithm="dqn",
                state_dim=0,
                action_dim=4
            )
            RLOptimizer(config=invalid_config)
        
        with pytest.raises(ValueError, match="Action dimension must be positive"):
            invalid_config = RLConfig(
                algorithm="dqn",
                state_dim=10,
                action_dim=0
            )
            RLOptimizer(config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_single_episode_training(self, rl_optimizer, mock_environment):
        """Test training for a single episode."""
        initial_epsilon = rl_optimizer.agent.epsilon
        
        episode_reward, episode_steps = await rl_optimizer.train_episode(mock_environment)
        
        assert isinstance(episode_reward, float)
        assert isinstance(episode_steps, int)
        assert episode_steps > 0
        assert rl_optimizer.total_episodes == 1
        assert rl_optimizer.total_steps == episode_steps
        
        # Epsilon should decay after episode
        assert rl_optimizer.agent.epsilon <= initial_epsilon
    
    @pytest.mark.asyncio
    async def test_multi_episode_training(self, rl_optimizer, mock_environment):
        """Test training for multiple episodes."""
        num_episodes = 5
        
        training_results = await rl_optimizer.train_multiple_episodes(
            environment=mock_environment,
            num_episodes=num_episodes
        )
        
        assert len(training_results["episode_rewards"]) == num_episodes
        assert len(training_results["episode_steps"]) == num_episodes
        assert rl_optimizer.total_episodes == num_episodes
        assert all(isinstance(reward, float) for reward in training_results["episode_rewards"])
        assert all(isinstance(steps, int) for steps in training_results["episode_steps"])
    
    def test_action_selection_exploration_vs_exploitation(self, rl_optimizer):
        """Test action selection balances exploration and exploitation."""
        state = np.random.rand(rl_optimizer.config.state_dim)
        
        # Test with high epsilon (exploration)
        rl_optimizer.agent.epsilon = 0.9
        actions_exploration = [rl_optimizer.agent.select_action(state) for _ in range(100)]
        
        # Test with low epsilon (exploitation)
        rl_optimizer.agent.epsilon = 0.1
        actions_exploitation = [rl_optimizer.agent.select_action(state) for _ in range(100)]
        
        # Exploration should have more variety
        exploration_variety = len(set(actions_exploration))
        exploitation_variety = len(set(actions_exploitation))
        
        assert exploration_variety >= exploitation_variety
        assert all(0 <= action < rl_optimizer.config.action_dim for action in actions_exploration)
        assert all(0 <= action < rl_optimizer.config.action_dim for action in actions_exploitation)
    
    def test_q_value_updates(self, rl_optimizer):
        """Test Q-value updates during learning."""
        state = np.random.rand(rl_optimizer.config.state_dim)
        action = 0
        reward = 1.0
        next_state = np.random.rand(rl_optimizer.config.state_dim)
        done = False
        
        # Get initial Q-value
        initial_q_value = rl_optimizer.agent.get_q_value(state, action)
        
        # Update Q-value
        rl_optimizer.agent.update_q_value(state, action, reward, next_state, done)
        
        # Q-value should change
        updated_q_value = rl_optimizer.agent.get_q_value(state, action)
        assert updated_q_value != initial_q_value
    
    def test_experience_replay_storage(self, rl_optimizer):
        """Test experience replay memory storage."""
        experience_replay = rl_optimizer.experience_replay
        
        # Create sample experience
        experience = Experience(
            state=np.random.rand(rl_optimizer.config.state_dim),
            action=1,
            reward=0.5,
            next_state=np.random.rand(rl_optimizer.config.state_dim),
            done=False,
            timestamp=datetime.now()
        )
        
        # Store experience
        experience_replay.store(experience)
        
        assert experience_replay.size() == 1
        assert not experience_replay.is_empty()
        
        # Store multiple experiences
        for i in range(10):
            exp = Experience(
                state=np.random.rand(rl_optimizer.config.state_dim),
                action=i % rl_optimizer.config.action_dim,
                reward=np.random.rand(),
                next_state=np.random.rand(rl_optimizer.config.state_dim),
                done=False,
                timestamp=datetime.now()
            )
            experience_replay.store(exp)
        
        assert experience_replay.size() == 11
    
    def test_experience_replay_sampling(self, rl_optimizer):
        """Test experience replay batch sampling."""
        experience_replay = rl_optimizer.experience_replay
        
        # Store enough experiences for sampling
        for i in range(100):
            experience = Experience(
                state=np.random.rand(rl_optimizer.config.state_dim),
                action=i % rl_optimizer.config.action_dim,
                reward=np.random.rand(),
                next_state=np.random.rand(rl_optimizer.config.state_dim),
                done=i % 10 == 0,  # Every 10th experience is terminal
                timestamp=datetime.now()
            )
            experience_replay.store(experience)
        
        # Sample batch
        batch_size = rl_optimizer.config.batch_size
        batch = experience_replay.sample(batch_size)
        
        assert len(batch) == batch_size
        assert all(isinstance(exp, Experience) for exp in batch)
        
        # Check that sampling is random (not sequential)
        batch1 = experience_replay.sample(batch_size)
        batch2 = experience_replay.sample(batch_size)
        
        # Batches should likely be different (with high probability)
        different_experiences = sum(1 for exp1, exp2 in zip(batch1, batch2) 
                                  if not np.array_equal(exp1.state, exp2.state))
        assert different_experiences > 0
    
    def test_reward_function_design(self, rl_optimizer):
        """Test custom reward function design and evaluation."""
        reward_function = rl_optimizer.reward_function
        
        # Test different scenarios
        scenarios = [
            {"performance_improvement": 0.1, "efficiency_gain": 0.05},
            {"performance_improvement": -0.05, "efficiency_gain": 0.02},
            {"performance_improvement": 0.2, "efficiency_gain": -0.01}
        ]
        
        rewards = []
        for scenario in scenarios:
            reward = reward_function.calculate_reward(scenario)
            rewards.append(reward)
            assert isinstance(reward, float)
        
        # Reward should correlate with performance improvement
        best_scenario_idx = max(range(len(scenarios)), 
                              key=lambda i: scenarios[i]["performance_improvement"])
        assert rewards[best_scenario_idx] == max(rewards)
    
    def test_strategy_optimization(self, rl_optimizer):
        """Test strategy optimization capabilities."""
        strategy_optimizer = StrategyOptimizer(rl_optimizer)
        
        # Define optimization strategies
        strategies = [
            OptimizationStrategy(
                name="aggressive",
                parameters={"exploration_rate": 0.3, "learning_rate": 1e-2}
            ),
            OptimizationStrategy(
                name="conservative", 
                parameters={"exploration_rate": 0.1, "learning_rate": 1e-4}
            ),
            OptimizationStrategy(
                name="balanced",
                parameters={"exploration_rate": 0.2, "learning_rate": 1e-3}
            )
        ]
        
        # Mock strategy evaluation
        with patch.object(strategy_optimizer, '_evaluate_strategy') as mock_eval:
            mock_eval.side_effect = [0.75, 0.65, 0.85]  # balanced performs best
            
            best_strategy = strategy_optimizer.optimize_strategy(strategies)
            
            assert best_strategy.name == "balanced"
            assert mock_eval.call_count == len(strategies)
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, rl_config):
        """Test multi-agent coordination and communication."""
        # Create multiple agents
        num_agents = 3
        coordinator = MultiAgentCoordinator(
            num_agents=num_agents,
            agent_configs=[rl_config] * num_agents
        )
        
        assert len(coordinator.agents) == num_agents
        assert coordinator.communication_enabled
        
        # Test agent coordination
        states = [np.random.rand(rl_config.state_dim) for _ in range(num_agents)]
        
        # Get coordinated actions
        actions = await coordinator.get_coordinated_actions(states)
        
        assert len(actions) == num_agents
        assert all(0 <= action < rl_config.action_dim for action in actions)
    
    def test_performance_metrics_tracking(self, rl_optimizer):
        """Test performance metrics tracking and analysis."""
        # Simulate training data
        episode_rewards = [10, 15, 12, 18, 22, 25, 20, 28, 30, 35]
        episode_steps = [100, 95, 105, 90, 85, 80, 88, 75, 70, 65]
        
        for reward, steps in zip(episode_rewards, episode_steps):
            rl_optimizer._record_episode_metrics(reward, steps)
        
        metrics = rl_optimizer.get_performance_metrics()
        
        assert "average_reward" in metrics
        assert "reward_trend" in metrics
        assert "average_steps" in metrics
        assert "convergence_rate" in metrics
        assert "exploration_rate" in metrics
        
        assert metrics["average_reward"] == np.mean(episode_rewards)
        assert metrics["average_steps"] == np.mean(episode_steps)
        assert metrics["reward_trend"] > 0  # Should show improvement
    
    def test_model_saving_and_loading(self, rl_optimizer, tmp_path):
        """Test model saving and loading functionality."""
        # Train for a few steps to modify model
        state = np.random.rand(rl_optimizer.config.state_dim)
        action = 0
        reward = 1.0
        next_state = np.random.rand(rl_optimizer.config.state_dim)
        
        rl_optimizer.agent.update_q_value(state, action, reward, next_state, False)
        
        # Save model
        model_path = tmp_path / "rl_model.pth"
        rl_optimizer.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Create new optimizer and load model
        new_optimizer = RLOptimizer(config=rl_optimizer.config)
        new_optimizer.load_model(str(model_path))
        
        # Q-values should match
        original_q = rl_optimizer.agent.get_q_value(state, action)
        loaded_q = new_optimizer.agent.get_q_value(state, action)
        
        assert abs(original_q - loaded_q) < 1e-6
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_rate(self, rl_optimizer, mock_environment):
        """Test adaptive learning rate adjustment."""
        initial_lr = rl_optimizer.agent.learning_rate
        
        # Simulate poor performance (should increase learning rate)
        poor_rewards = [-1.0, -0.8, -1.2, -0.9, -1.1]
        for reward in poor_rewards:
            rl_optimizer._update_adaptive_learning_rate(reward)
        
        assert rl_optimizer.agent.learning_rate >= initial_lr
        
        # Reset and simulate good performance (should decrease learning rate)
        rl_optimizer.agent.learning_rate = initial_lr
        good_rewards = [1.0, 1.2, 0.9, 1.1, 1.3]
        for reward in good_rewards:
            rl_optimizer._update_adaptive_learning_rate(reward)
        
        assert rl_optimizer.agent.learning_rate <= initial_lr
    
    def test_curriculum_learning(self, rl_optimizer):
        """Test curriculum learning with progressive difficulty."""
        curriculum = [
            {"difficulty": "easy", "max_steps": 50, "reward_scale": 1.0},
            {"difficulty": "medium", "max_steps": 100, "reward_scale": 0.8},
            {"difficulty": "hard", "max_steps": 200, "reward_scale": 0.6}
        ]
        
        rl_optimizer.set_curriculum(curriculum)
        
        # Should start with easy difficulty
        current_stage = rl_optimizer.get_current_curriculum_stage()
        assert current_stage["difficulty"] == "easy"
        
        # Simulate successful completion of stages
        rl_optimizer._advance_curriculum_stage()
        current_stage = rl_optimizer.get_current_curriculum_stage()
        assert current_stage["difficulty"] == "medium"
        
        rl_optimizer._advance_curriculum_stage()
        current_stage = rl_optimizer.get_current_curriculum_stage()
        assert current_stage["difficulty"] == "hard"


class TestQLearningAgent:
    """Test suite for Q-Learning agent implementation."""
    
    @pytest.fixture
    def q_agent(self):
        """Create Q-Learning agent for testing."""
        config = RLConfig(
            algorithm="q_learning",
            state_dim=5,
            action_dim=3,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )
        return QLearningAgent(config)
    
    def test_q_table_initialization(self, q_agent):
        """Test Q-table initialization."""
        assert q_agent.q_table is not None
        assert q_agent.q_table.shape == (q_agent.state_space_size, q_agent.config.action_dim)
        
        # Q-table should be initialized to zeros or small random values
        assert np.all(np.abs(q_agent.q_table) <= 1.0)
    
    def test_state_discretization(self, q_agent):
        """Test continuous state discretization."""
        continuous_state = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        
        discrete_state = q_agent.discretize_state(continuous_state)
        
        assert isinstance(discrete_state, int)
        assert 0 <= discrete_state < q_agent.state_space_size
        
        # Same continuous state should map to same discrete state
        discrete_state2 = q_agent.discretize_state(continuous_state)
        assert discrete_state == discrete_state2
    
    def test_epsilon_greedy_action_selection(self, q_agent):
        """Test epsilon-greedy action selection."""
        state = np.random.rand(q_agent.config.state_dim)
        
        # With epsilon = 1.0, should always explore (random actions)
        q_agent.epsilon = 1.0
        actions = [q_agent.select_action(state) for _ in range(100)]
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should have variety
        
        # With epsilon = 0.0, should always exploit (same action)
        q_agent.epsilon = 0.0
        actions = [q_agent.select_action(state) for _ in range(10)]
        assert len(set(actions)) == 1  # Should be consistent
    
    def test_q_value_update_formula(self, q_agent):
        """Test Q-value update follows correct formula."""
        state = np.random.rand(q_agent.config.state_dim)
        action = 1
        reward = 1.0
        next_state = np.random.rand(q_agent.config.state_dim)
        
        # Get initial values
        discrete_state = q_agent.discretize_state(state)
        discrete_next_state = q_agent.discretize_state(next_state)
        
        old_q_value = q_agent.q_table[discrete_state, action]
        max_next_q = np.max(q_agent.q_table[discrete_next_state])
        
        # Update Q-value
        q_agent.update_q_value(state, action, reward, next_state, done=False)
        
        # Calculate expected new Q-value
        expected_q = old_q_value + q_agent.config.learning_rate * (
            reward + q_agent.config.discount_factor * max_next_q - old_q_value
        )
        
        new_q_value = q_agent.q_table[discrete_state, action]
        assert abs(new_q_value - expected_q) < 1e-6


class TestPolicyGradientAgent:
    """Test suite for Policy Gradient agent implementation."""
    
    @pytest.fixture
    def pg_agent(self):
        """Create Policy Gradient agent for testing."""
        config = RLConfig(
            algorithm="policy_gradient",
            state_dim=8,
            action_dim=4,
            learning_rate=1e-3,
            discount_factor=0.99
        )
        return PolicyGradientAgent(config)
    
    def test_policy_network_initialization(self, pg_agent):
        """Test policy network initialization."""
        assert pg_agent.policy_network is not None
        assert pg_agent.optimizer is not None
        
        # Test network output shape
        state = torch.randn(1, pg_agent.config.state_dim)
        action_probs = pg_agent.policy_network(state)
        
        assert action_probs.shape == (1, pg_agent.config.action_dim)
        assert torch.allclose(torch.sum(action_probs, dim=1), torch.ones(1))  # Probabilities sum to 1
    
    def test_action_sampling(self, pg_agent):
        """Test action sampling from policy."""
        state = np.random.rand(pg_agent.config.state_dim)
        
        actions = [pg_agent.select_action(state) for _ in range(100)]
        
        assert all(0 <= action < pg_agent.config.action_dim for action in actions)
        assert len(set(actions)) > 1  # Should have variety due to stochastic policy
    
    def test_policy_gradient_update(self, pg_agent):
        """Test policy gradient update mechanism."""
        # Create episode data
        states = [np.random.rand(pg_agent.config.state_dim) for _ in range(10)]
        actions = [np.random.randint(0, pg_agent.config.action_dim) for _ in range(10)]
        rewards = [np.random.rand() for _ in range(10)]
        
        # Get initial policy parameters
        initial_params = [param.clone() for param in pg_agent.policy_network.parameters()]
        
        # Update policy
        pg_agent.update_policy(states, actions, rewards)
        
        # Parameters should change
        updated_params = list(pg_agent.policy_network.parameters())
        
        for initial, updated in zip(initial_params, updated_params):
            assert not torch.equal(initial, updated)
    
    def test_advantage_calculation(self, pg_agent):
        """Test advantage calculation for policy gradient."""
        rewards = [1.0, 0.5, 2.0, -0.5, 1.5]
        
        advantages = pg_agent.calculate_advantages(rewards)
        
        assert len(advantages) == len(rewards)
        assert isinstance(advantages, list)
        assert all(isinstance(adv, float) for adv in advantages)


class TestExperienceReplay:
    """Test suite for Experience Replay functionality."""
    
    @pytest.fixture
    def experience_replay(self):
        """Create ExperienceReplay instance for testing."""
        return ExperienceReplay(capacity=1000)
    
    def test_experience_replay_initialization(self, experience_replay):
        """Test ExperienceReplay initialization."""
        assert experience_replay.capacity == 1000
        assert experience_replay.size() == 0
        assert experience_replay.is_empty()
    
    def test_circular_buffer_behavior(self, experience_replay):
        """Test circular buffer behavior when capacity is exceeded."""
        # Create small buffer for testing
        small_buffer = ExperienceReplay(capacity=3)
        
        # Add experiences beyond capacity
        for i in range(5):
            experience = Experience(
                state=np.array([i]),
                action=i,
                reward=float(i),
                next_state=np.array([i+1]),
                done=False,
                timestamp=datetime.now()
            )
            small_buffer.store(experience)
        
        # Should not exceed capacity
        assert small_buffer.size() == 3
        
        # Should contain most recent experiences
        batch = small_buffer.sample(3)
        rewards = [exp.reward for exp in batch]
        assert set(rewards) == {2.0, 3.0, 4.0}  # Most recent 3
    
    def test_prioritized_experience_replay(self):
        """Test prioritized experience replay."""
        per_buffer = ExperienceReplay(capacity=1000, prioritized=True)
        
        # Add experiences with different TD errors (priorities)
        for i in range(10):
            experience = Experience(
                state=np.random.rand(5),
                action=i % 4,
                reward=np.random.rand(),
                next_state=np.random.rand(5),
                done=False,
                timestamp=datetime.now()
            )
            priority = abs(np.random.rand() - 0.5)  # Random priority
            per_buffer.store(experience, priority=priority)
        
        # Sample should be biased toward high-priority experiences
        batch = per_buffer.sample(5)
        assert len(batch) == 5
        
        # Test priority updates
        new_priorities = [0.9, 0.1, 0.8, 0.2, 0.7]
        per_buffer.update_priorities(list(range(5)), new_priorities)
    
    def test_experience_replay_statistics(self, experience_replay):
        """Test experience replay statistics and metrics."""
        # Add diverse experiences
        for i in range(50):
            experience = Experience(
                state=np.random.rand(5),
                action=i % 4,
                reward=np.random.rand() - 0.5,
                next_state=np.random.rand(5),
                done=i % 10 == 0,
                timestamp=datetime.now() - timedelta(seconds=i)
            )
            experience_replay.store(experience)
        
        stats = experience_replay.get_statistics()
        
        assert "total_experiences" in stats
        assert "average_reward" in stats
        assert "terminal_state_ratio" in stats
        assert "action_distribution" in stats
        
        assert stats["total_experiences"] == 50
        assert 0.0 <= stats["terminal_state_ratio"] <= 1.0


class TestRewardFunction:
    """Test suite for Reward Function design and evaluation."""
    
    @pytest.fixture
    def reward_function(self):
        """Create RewardFunction instance for testing."""
        return RewardFunction()
    
    def test_basic_reward_calculation(self, reward_function):
        """Test basic reward calculation."""
        # Test positive performance improvement
        context = {
            "performance_improvement": 0.1,
            "efficiency_gain": 0.05,
            "resource_usage": 0.8
        }
        
        reward = reward_function.calculate_reward(context)
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for improvements
        
        # Test negative performance
        context["performance_improvement"] = -0.1
        negative_reward = reward_function.calculate_reward(context)
        assert negative_reward < reward  # Should be lower
    
    def test_multi_objective_reward(self, reward_function):
        """Test multi-objective reward function."""
        objectives = {
            "accuracy": 0.85,
            "speed": 0.9,
            "memory_efficiency": 0.7,
            "robustness": 0.8
        }
        
        weights = {
            "accuracy": 0.4,
            "speed": 0.3,
            "memory_efficiency": 0.2,
            "robustness": 0.1
        }
        
        reward = reward_function.calculate_multi_objective_reward(objectives, weights)
        
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0
        
        # Verify weighted calculation
        expected_reward = sum(objectives[obj] * weights[obj] for obj in objectives)
        assert abs(reward - expected_reward) < 1e-6
    
    def test_shaped_reward_function(self, reward_function):
        """Test reward shaping for better learning."""
        # Test progress-based shaping
        progress_contexts = [
            {"progress": 0.1, "goal_distance": 0.9},
            {"progress": 0.5, "goal_distance": 0.5},
            {"progress": 0.9, "goal_distance": 0.1}
        ]
        
        shaped_rewards = [
            reward_function.calculate_shaped_reward(context) 
            for context in progress_contexts
        ]
        
        # Rewards should increase with progress
        assert shaped_rewards[0] < shaped_rewards[1] < shaped_rewards[2]
    
    def test_adaptive_reward_function(self, reward_function):
        """Test adaptive reward function that learns from experience."""
        # Simulate learning from successful episodes
        successful_episodes = [
            {"actions": [1, 2, 1], "outcome": "success", "final_reward": 1.0},
            {"actions": [2, 1, 2], "outcome": "success", "final_reward": 0.8},
            {"actions": [1, 1, 1], "outcome": "failure", "final_reward": -0.5}
        ]
        
        # Update reward function based on episodes
        reward_function.update_from_episodes(successful_episodes)
        
        # Test that reward function has adapted
        test_context = {"actions": [1, 2, 1]}  # Similar to successful pattern
        adapted_reward = reward_function.calculate_reward(test_context)
        
        assert isinstance(adapted_reward, float)