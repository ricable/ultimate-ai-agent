"""
8B Model Benchmarks
===================

Specialized benchmarking tools for 8B parameter language models.
"""

try:
    from .comprehensive_model_report import (
        Comprehensive8BReportGenerator,
        ModelBenchmark,
        run_comprehensive_8b_model_benchmark
    )
    MODEL_8B_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    MODEL_8B_COMPREHENSIVE_AVAILABLE = False

try:
    from .practical_finetuning_test import (
        test_hf_8b_class_finetuning,
        run_practical_8b_finetuning_test
    )
    MODEL_8B_FINETUNING_AVAILABLE = True
except ImportError:
    MODEL_8B_FINETUNING_AVAILABLE = False

__all__ = []

if MODEL_8B_COMPREHENSIVE_AVAILABLE:
    __all__.extend([
        'Comprehensive8BReportGenerator',
        'ModelBenchmark',
        'run_comprehensive_8b_model_benchmark'
    ])

if MODEL_8B_FINETUNING_AVAILABLE:
    __all__.extend([
        'test_hf_8b_class_finetuning',
        'run_practical_8b_finetuning_test'
    ])