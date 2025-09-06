"""
core/__init__.py
GACoT Core Evaluation Components
"""

from .evaluator import FinancialReasoningEval, EvaluationResult
from .test_data import TestDataManager
from .cache_manager import CacheManager

# Import from new viz package structure
from ..viz import (
    create_visualizations,
    aggregate_model_results,
    visualize_cached_results
)

# Import metrics from new structure
from ..metrics import (
    EvalMetrics,
    MetricsCalculator,
    DependencyGraphEvaluator,
    CascadeIdentificationEvaluator,
    ValueCorrectnessEvaluator,
    EfficiencyEvaluator
)

__all__ = [
    # Core evaluation
    "FinancialReasoningEval",
    "EvaluationResult",
    
    # Data management
    "TestDataManager",
    "CacheManager",
    
    # Visualization (from new viz package)
    "create_visualizations",
    "aggregate_model_results",
    "visualize_cached_results",
    
    # Metrics (re-exported)
    "EvalMetrics",
    "MetricsCalculator",
    "DependencyGraphEvaluator",
    "CascadeIdentificationEvaluator",
    "ValueCorrectnessEvaluator",
    "EfficiencyEvaluator",
]

__version__ = "2.0.0"