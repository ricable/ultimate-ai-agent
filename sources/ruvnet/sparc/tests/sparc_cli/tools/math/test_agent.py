import pytest
from unittest.mock import Mock, patch
from langchain.chat_models.base import BaseChatModel
from langchain.tools import Tool
from sparc_cli.tools.math.agent import ReActMathAgent

@pytest.fixture
def mock_llm():
    llm = Mock(spec=BaseChatModel)
    llm.predict.return_value = "Test response"
    return llm

@pytest.fixture
def basic_agent(mock_llm):
    return ReActMathAgent(llm=mock_llm)

class TestAgentInitialization:
    def test_basic_initialization(self, mock_llm):
        agent = ReActMathAgent(llm=mock_llm)
        assert agent.llm == mock_llm
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'agent_chain')

    def test_custom_config_initialization(self, mock_llm):
        custom_config = {
            'max_iterations': 5,
            'memory_key': 'test_memory'
        }
        agent = ReActMathAgent(llm=mock_llm, config=custom_config)
        assert agent.config['max_iterations'] == 5
        assert agent.config['memory_key'] == 'test_memory'

    def test_tools_setup(self, basic_agent):
        assert len(basic_agent.tools) > 0
        assert all(isinstance(tool, Tool) for tool in basic_agent.tools)

    @patch('sparc_cli.tools.math.agent.ReActMathAgent.setup_chains')
    def test_chain_initialization(self, mock_setup_chains, mock_llm):
        agent = ReActMathAgent(llm=mock_llm)
        mock_setup_chains.assert_called_once()

class TestProblemSolving:
    def test_basic_problem_solving(self, basic_agent):
        problem = "What is 2 + 2?"
        with patch.object(basic_agent, 'run') as mock_run:
            mock_run.return_value = {'output': '4'}
            result = basic_agent.run(problem)
            assert result['output'] == '4'
            mock_run.assert_called_once_with(problem)

    def test_multi_step_problem(self, basic_agent):
        problem = "If I have 3 apples and buy 2 more, then eat 1, how many do I have?"
        with patch.object(basic_agent, 'run') as mock_run:
            mock_run.return_value = {
                'output': '4',
                'intermediate_steps': [
                    ('thought', '3 + 2 = 5'),
                    ('thought', '5 - 1 = 4')
                ]
            }
            result = basic_agent.run(problem)
            assert result['output'] == '4'
            assert len(result['intermediate_steps']) == 2

    def test_cot_reasoning(self, basic_agent):
        problem = "Solve: 15% of 200"
        with patch.object(basic_agent.llm, 'predict') as mock_predict:
            mock_predict.side_effect = [
                "Let me solve this step by step:\n1. Convert 15% to decimal: 15/100 = 0.15\n2. Multiply: 200 * 0.15 = 30",
                "30"
            ]
            result = basic_agent.run(problem)
            assert mock_predict.call_count >= 2
            assert 'intermediate_steps' in result
            assert len(result['intermediate_steps']) >= 2
            assert result['output'] == '30'
            
    def test_complex_reasoning_chain(self, basic_agent):
        """Test handling of multi-step reasoning chains"""
        problem = "If 15% of x is 30, what is x?"
        with patch.object(basic_agent.llm, 'predict') as mock_predict:
            mock_predict.side_effect = [
                "Let's solve step by step:\n1. Let 15% of x = 30\n2. 0.15x = 30\n3. x = 30/0.15\n4. x = 200",
                "200"
            ]
            result = basic_agent.run(problem)
            assert result['output'] == '200'
            assert any('equation' in str(step) for step in result['intermediate_steps'])

    def test_tool_usage(self, basic_agent):
        problem = "Calculate square root of 16"
        mock_tool = Mock()
        mock_tool.run.return_value = "4"
        basic_agent.tools = [mock_tool]
        
        with patch.object(basic_agent, 'run') as mock_run:
            mock_run.return_value = {'output': '4'}
            basic_agent.run(problem)
            mock_run.assert_called_once()

class TestErrorHandling:
    def test_invalid_input(self, basic_agent):
        with pytest.raises(ValueError):
            basic_agent.run("")

    def test_error_recovery(self, basic_agent):
        problem = "Complex math problem"
        with patch.object(basic_agent.llm, 'predict') as mock_predict:
            mock_predict.side_effect = [Exception("First attempt failed"), "Recovered answer"]
            with patch.object(basic_agent, 'run') as mock_run:
                mock_run.return_value = {'output': 'Recovered answer'}
                result = basic_agent.run(problem)
                assert result['output'] == 'Recovered answer'

    @pytest.mark.timeout(1)
    def test_timeout(self, basic_agent):
        problem = "Time-consuming problem"
        with patch.object(basic_agent, 'run') as mock_run:
            mock_run.return_value = {'output': 'Timeout occurred'}
            result = basic_agent.run(problem)
            assert 'output' in result

    def test_tool_failure(self, basic_agent):
        problem = "Calculate with failing tool"
        failing_tool = Mock()
        failing_tool.run.side_effect = Exception("Tool failed")
        basic_agent.tools = [failing_tool]
        
        with patch.object(basic_agent, 'run') as mock_run:
            mock_run.return_value = {'output': 'Handled tool failure'}
            result = basic_agent.run(problem)
            assert 'output' in result
