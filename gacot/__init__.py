"""
GACoT: Graph-Augmented Chain of Thought
Financial reasoning with dependency tracking
"""

from .llm_client import LLMClient
from .dependency_tracker import DependencyTracker, Variable
from .circular_solver import CircularSolver

__version__ = "1.0.0"

__all__ = [
    "LLMClient",
    "Variable",
]