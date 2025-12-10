import pytest
from typing import Dict, Any
from dataclasses import asdict
from sparc_cli.tools.math.models import (
    BenchmarkRequest,
    BenchmarkResponse,
    ValidationResult
)

class TestBenchmarkRequest:
    def test_create_basic_request(self):
        request = BenchmarkRequest(
            problem_id="test-1",
            problem_text="2 + 2",
            expected_type="numerical",
            metadata={"expected_answer": 4}
        )
        assert request.problem_id == "test-1"
        assert request.problem_text == "2 + 2"
        assert request.expected_type == "numerical"
        assert request.metadata["expected_answer"] == 4

    def test_request_validation(self):
        with pytest.raises(ValueError):
            BenchmarkRequest(
                problem_id="",
                problem_text="",
                expected_type="invalid"
            )

    def test_request_serialization(self):
        request = BenchmarkRequest(
            problem_text="3 * 4",
            expected_result=12,
            metadata={"difficulty": "easy"}
        )
        serialized = asdict(request)
        assert serialized["problem_text"] == "3 * 4"
        assert serialized["expected_result"] == 12
        assert serialized["metadata"] == {"difficulty": "easy"}

    @pytest.mark.parametrize("validation_type", [
        "numerical",
        "symbolic",
        "matrix"
    ])
    def test_validation_types(self, validation_type):
        request = BenchmarkRequest(
            problem_text="test",
            expected_result=None,
            validation_type=validation_type
        )
        assert request.validation_type == validation_type

class TestBenchmarkResponse:
    def test_create_basic_response(self):
        response = BenchmarkResponse(
            result=4,
            is_correct=True,
            confidence=0.95
        )
        assert response.result == 4
        assert response.is_correct is True
        assert response.confidence == 0.95
        assert response.explanation == ""
        assert response.metadata is None

    def test_response_with_explanation(self):
        response = BenchmarkResponse(
            result="x + 2",
            is_correct=True,
            confidence=1.0,
            explanation="Simplified the expression"
        )
        assert response.explanation == "Simplified the expression"

    def test_response_serialization(self):
        response = BenchmarkResponse(
            result=[[1, 2], [3, 4]],
            is_correct=True,
            confidence=0.8,
            metadata={"computation_time": 0.5}
        )
        serialized = asdict(response)
        assert serialized["result"] == [[1, 2], [3, 4]]
        assert serialized["is_correct"] is True
        assert serialized["confidence"] == 0.8
        assert serialized["metadata"] == {"computation_time": 0.5}

    @pytest.mark.parametrize("confidence", [
        -0.1,
        1.1,
        2.0
    ])
    def test_invalid_confidence(self, confidence):
        with pytest.raises(ValueError):
            BenchmarkResponse(
                result=None,
                is_correct=False,
                confidence=confidence
            )
