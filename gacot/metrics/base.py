"""
metrics/base.py
Base classes and data structures for evaluation metrics
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class EvalMetrics:
    """Core evaluation metrics container."""
    dependency_graph_extraction: float  # F1 score for graph extraction
    cascade_identification_accuracy: float  # Accuracy of cascade prediction
    value_correctness: float  # Mathematical correctness
    efficiency_score: float  # Token efficiency and focus
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "dge": self.dependency_graph_extraction,
            "cia": self.cascade_identification_accuracy,
            "vc": self.value_correctness,
            "efficiency": self.efficiency_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'EvalMetrics':
        """Create from dictionary."""
        return cls(
            dependency_graph_extraction=data.get("dge", 0.0),
            cascade_identification_accuracy=data.get("cia", 0.0),
            value_correctness=data.get("vc", 0.0),
            efficiency_score=data.get("efficiency", 0.0)
        )
    
    def __repr__(self):
        return (
            f"EvalMetrics(\n"
            f"  DGE (Graph Extraction): {self.dependency_graph_extraction:.1%}\n"
            f"  CIA (Cascade Accuracy): {self.cascade_identification_accuracy:.1%}\n"
            f"  VC (Value Correctness): {self.value_correctness:.1%}\n"
            f"  Efficiency: {self.efficiency_score:.1%}\n"
            f")"
        )


class MetricEvaluator(ABC):
    """Abstract base class for individual metric evaluators."""
    
    @abstractmethod
    def evaluate(
        self,
        response: str,
        problem: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Evaluate a single metric.
        
        Args:
            response: Model response text
            problem: Problem data including ground truth
            **kwargs: Additional parameters
            
        Returns:
            Metric score between 0 and 1
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        pass
    
    def _normalize_var(self, var: str) -> str:
        """Normalize variable name for comparison."""
        if not var:
            return ""
        return var.lower().replace('_', '').replace('-', '').replace(' ', '').strip()
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from text."""
        import re
        
        # Remove numbers and symbols
        text = re.sub(r'[\$£€¥]\d+[MBK]?|\d+\.?\d*[MBK]?|[%]', '', text)
        text = re.sub(r'[+\-*/()%,]', ' ', text)
        
        variables = []
        for match in re.finditer(r'\b[A-Za-z][A-Za-z_]*\b', text):
            var = match.group()
            # Skip common words
            if len(var) > 1 and var.lower() not in [
                'is', 'of', 'the', 'and', 'or', 'to', 'from', 
                'by', 'at', 'in', 'for', 'with', 'as'
            ]:
                variables.append(var)
        
        return variables