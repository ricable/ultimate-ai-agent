import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import numpy as np
from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool

from sparc_cli.tools.math.evaluator import (
    MathBenchmarkEvaluator,
    NuminaMathEvaluator, 
    MathOdysseyEvaluator,
    CalculatorTool,
    SymbolicSolverTool
)
from sparc_cli.tools.math.models import BenchmarkRequest, BenchmarkResponse

# Mock responses for different problem types
MOCK_RESPONSES = {
    "numerical": "After calculating step by step:\n1. First add 2 + 2 = 4\n2. Then multiply by 2\nFinal answer: 8",
    "symbolic": "Let's solve this:\n1. Combine like terms: 2x + 3x = 5x\n2. Add constant: 5x + 2\nResult: 5x + 2",
    "matrix": "Matrix multiplication steps:\n1. Multiply corresponding elements\n2. Sum the products\nResult: [[7,10],[15,22]]"
}

# Test Fixtures
@pytest.fixture
def mock_llm():
    llm = Mock(spec=BaseLLM)
    def side_effect(prompt):
        if "numerical" in prompt.lower():
            return MOCK_RESPONSES["numerical"]
        elif "symbolic" in prompt.lower():
            return MOCK_RESPONSES["symbolic"]
        elif "matrix" in prompt.lower():
            return MOCK_RESPONSES["matrix"]
        return "4"
    llm.predict.side_effect = side_effect
    return llm

@pytest.fixture
def mock_calculator():
    calculator = Mock(spec=CalculatorTool)
    calculator.run.return_value = 8.0
    return calculator

@pytest.fixture
def mock_symbolic_solver():
    solver = Mock(spec=SymbolicSolverTool)
    solver.run.return_value = "5x + 2"
    return solver

@pytest.fixture
def basic_evaluator(mock_llm):
    class TestEvaluator(MathBenchmarkEvaluator):
        def setup_tools(self) -> None:
            self.tools = []
            
        def evaluate_problem(self, request: BenchmarkRequest) -> BenchmarkResponse:
            return self.process_results(
                request.problem_id,
                8.0,
                {'valid': True, 'message': 'Test passed'}
            )
    
    return TestEvaluator(llm=mock_llm)

@pytest.fixture
def numina_evaluator(mock_llm, mock_calculator):
    evaluator = NuminaMathEvaluator(llm=mock_llm)
    evaluator.calculator = mock_calculator
    return evaluator

@pytest.fixture
def odyssey_evaluator(mock_llm, mock_calculator, mock_symbolic_solver):
    evaluator = MathOdysseyEvaluator(llm=mock_llm)
    evaluator.calculator = mock_calculator
    evaluator.symbolic_solver = mock_symbolic_solver
    return evaluator

# Basic Functionality Tests
class TestBasicFunctionality:
    def test_evaluator_initialization(self, mock_llm):
        """Test evaluator initialization with different settings"""
        evaluator = NuminaMathEvaluator(
            llm=mock_llm,
            temperature=0.7,
            max_tokens=100
        )
        assert evaluator.llm == mock_llm
        assert isinstance(evaluator.calculator, CalculatorTool)

    @pytest.mark.parametrize("problem_text,expected", [
        ("4 * 2", 8.0),
        ("16 / 2", 8.0),
        ("2 + 6", 8.0)
    ])
    def test_simple_numerical(self, numina_evaluator, problem_text, expected):
        """Test basic numerical calculations"""
        request = BenchmarkRequest(
            problem_id="num-1",
            problem_text=problem_text,
            expected_type="numerical",
            metadata={'expected_answer': expected}
        )
        response = numina_evaluator.evaluate_problem(request)
        assert response.answer == expected
        assert response.validation_result['valid']

    def test_symbolic_expression(self, odyssey_evaluator):
        """Test symbolic expression evaluation"""
        request = BenchmarkRequest(
            problem_id="sym-1",
            problem_text="2x + 3x + 2",
            expected_type="symbolic",
            metadata={'expected_answer': "5x + 2"}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert response.answer == "5x + 2"
        assert response.validation_result['valid']

    def test_matrix_operations(self, odyssey_evaluator):
        """Test matrix operations"""
        request = BenchmarkRequest(
            problem_id="mat-1",
            problem_text="[[1,2],[3,4]] * [[2,3],[1,2]]",
            expected_type="matrix",
            metadata={'expected_answer': [[7,10],[15,22]]}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert response.validation_result['valid']

# Advanced Test Cases
class TestAdvancedFunctionality:
    def test_chain_of_thought(self, odyssey_evaluator):
        """Test chain-of-thought reasoning process"""
        request = BenchmarkRequest(
            problem_id="cot-1",
            problem_text="Solve step by step: 2x + 3 = 11",
            expected_type="symbolic",
            metadata={'expected_answer': "x = 4"}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert 'steps' in response.metadata
        assert len(response.metadata['steps']) > 1
        assert all('reasoning' in step for step in response.metadata['steps'])
        assert response.validation_result['valid']
        
    def test_complex_symbolic_evaluation(self, odyssey_evaluator):
        """Test evaluation of complex symbolic expressions"""
        request = BenchmarkRequest(
            problem_id="sym-2",
            problem_text="Simplify: (x + 1)^2 - (x - 1)^2",
            expected_type="symbolic",
            metadata={'expected_answer': "4x"}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert response.validation_result['valid']

    def test_multi_step_problem(self, odyssey_evaluator):
        """Test handling of complex multi-step problems"""
        problem = """
        1. First solve: 2x + 3 = 11
        2. Then substitute x in: 3x^2 + 2
        """
        request = BenchmarkRequest(
            problem_id="multi-1",
            problem_text=problem,
            expected_type="numerical",
            metadata={'expected_answer': 50}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert response.validation_result['valid']
        assert 'intermediate_results' in response.metadata

    @pytest.mark.parametrize("result_type,input_val,expected", [
        ("exact", 8.0, 8.0),
        ("approximate", 7.99999, 8.0),
        ("percentage", 0.08, "8%"),
        ("interval", [7.9, 8.1], 8.0)
    ])
    def test_result_types(self, basic_evaluator, result_type, input_val, expected):
        """Test different types of results and their validation"""
        result = basic_evaluator.validate_solution(
            input_val, 
            expected,
            result_type
        )
        assert result['valid']

# Edge Cases and Error Handling
class TestEdgeCases:
    def test_invalid_input_handling(self, numina_evaluator):
        """Test handling of invalid inputs"""
        request = BenchmarkRequest(
            problem_id="err-1",
            problem_text="1/0",
            expected_type="numerical",
            metadata={}
        )
        response = numina_evaluator.evaluate_problem(request)
        assert not response.validation_result['valid']
        assert "error" in response.validation_result['message'].lower()

    def test_unsupported_operations(self, odyssey_evaluator):
        """Test handling of unsupported operations"""
        request = BenchmarkRequest(
            problem_id="err-2",
            problem_text="∭ sin(x) dx dy dz",  # Triple integral
            expected_type="symbolic",
            metadata={}
        )
        response = odyssey_evaluator.evaluate_problem(request)
        assert not response.validation_result['valid']
        assert "unsupported" in response.validation_result['message'].lower()

    def test_malformed_problem(self, basic_evaluator):
        """Test handling of malformed problem descriptions"""
        request = BenchmarkRequest(
            problem_id="err-3",
            problem_text="",  # Empty problem
            expected_type="numerical",
            metadata={}
        )
        response = basic_evaluator.evaluate_problem(request)
        assert not response.validation_result['valid']
        assert "invalid problem" in response.validation_result['message'].lower()

    def test_validation_edge_cases(self, basic_evaluator):
        """Test edge cases in response validation"""
        cases = [
            (float('inf'), 'numerical'),
            (float('nan'), 'numerical'),
            (None, 'symbolic'),
            ('', 'matrix')
        ]
        for value, type_ in cases:
            result = basic_evaluator.validate_solution(value, value, type_)
            assert not result['valid']
            assert 'invalid' in result['message'].lower()

# Tool-specific Tests
class TestTools:
    def test_calculator_tool_limits(self):
        """Test calculator tool limits and error handling"""
        calculator = CalculatorTool()
        with pytest.raises(ValueError):
            calculator._run("9" * 100)  # Too large number
        with pytest.raises(ValueError):
            calculator._run("1/0")  # Division by zero

    def test_symbolic_solver_complexity(self):
        """Test symbolic solver with varying complexity"""
        solver = SymbolicSolverTool()
        simple = solver._run("x + x")
        complex = solver._run("sin(x^2) + cos(x^2)")
        assert simple != complex
        assert all(isinstance(result, str) for result in [simple, complex])
