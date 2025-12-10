from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkRequest:
    """A request to evaluate a math benchmark problem.
    
    Attributes:
        problem_id: Unique identifier for the problem
        problem_text: The actual problem text/statement
        expected_type: Type of answer expected (numerical/symbolic/matrix)
        metadata: Additional metadata about the problem
    """
    problem_id: str
    problem_text: str
    expected_type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary representation.
        
        Returns:
            Dictionary containing all fields of the request
        """
        return {
            'problem_id': self.problem_id,
            'problem_text': self.problem_text,
            'expected_type': self.expected_type,
            'metadata': self.metadata
        }

@dataclass 
class BenchmarkResponse:
    """Response containing the answer and validation results for a benchmark problem.
    
    Attributes:
        problem_id: ID of the problem this response is for
        answer: The computed answer (can be numerical, symbolic expression, or matrix)
        validation_result: Dictionary containing validation details
        metadata: Additional metadata about the response
    """
    problem_id: str
    answer: Any
    validation_result: Dict[str, Any]
    metadata: Dict[str, Any]

    @property
    def is_valid(self) -> bool:
        """Check if the response passed validation.
        
        Returns:
            True if validation passed, False otherwise
        """
        return self.validation_result.get('valid', False)
