"""
metrics/calculator.py
Metrics calculator that orchestrates all modular evaluators
"""

import numpy as np
from typing import Dict, List, Optional
from .base import EvalMetrics
from .dependency_graph import DependencyGraphEvaluator
from .cascade_identification import CascadeIdentificationEvaluator
from .value_extractor import ValueCorrectnessEvaluator
from .efficiency import EfficiencyEvaluator


class MetricsCalculator:
    """Orchestrates evaluation using modular metric evaluators."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize with all metric evaluators.
        
        Args:
            verbose: Enable verbose output for debugging
        """
        self.verbose = verbose
        
        # Initialize all evaluators
        self.dge_evaluator = DependencyGraphEvaluator()
        self.cia_evaluator = CascadeIdentificationEvaluator()
        self.vc_evaluator = ValueCorrectnessEvaluator()
        self.eff_evaluator = EfficiencyEvaluator()
    
    def evaluate_response(
        self,
        response: str,
        problem: Dict,
        use_runtime: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single response against ground truth.
        
        Args:
            response: Model response text
            problem: Problem with ground truth
            use_runtime: Whether runtime tracking was used (for context)
            
        Returns:
            Dictionary with metric scores
        """
        # Run each evaluator
        dge = self.dge_evaluator.evaluate(
            response, 
            problem, 
            verbose=self.verbose
        )
        
        cia = self.cia_evaluator.evaluate(
            response,
            problem,
            verbose=self.verbose
        )
        
        vc = self.vc_evaluator.evaluate(
            response, 
            problem,
            verbose=self.verbose
        )
        
        eff = self.eff_evaluator.evaluate(
            response, 
            problem,
            verbose=self.verbose,
            use_runtime=use_runtime
        )
        
        # Debug output if verbose
        if self.verbose and any(score < 0.5 for score in [dge, cia, vc, eff]):
            print(f"  Metrics: DGE={dge:.2f}, CIA={cia:.2f}, VC={vc:.2f}, EFF={eff:.2f}")
        
        return {
            "dge": dge,
            "cia": cia,
            "vc": vc,
            "efficiency": eff
        }
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> EvalMetrics:
        """
        Aggregate metrics across multiple evaluations.
        
        Args:
            all_metrics: List of metric dictionaries
            
        Returns:
            Aggregated EvalMetrics object
        """
        if not all_metrics:
            return EvalMetrics(0.0, 0.0, 0.0, 0.0)
        
        # Calculate means for each metric
        dge_scores = [m.get("dge", 0.0) for m in all_metrics]
        cia_scores = [m.get("cia", 0.0) for m in all_metrics]
        vc_scores = [m.get("vc", 0.0) for m in all_metrics]
        eff_scores = [m.get("efficiency", 0.0) for m in all_metrics]
        
        aggregated = EvalMetrics(
            dependency_graph_extraction=np.mean(dge_scores) if dge_scores else 0.0,
            cascade_identification_accuracy=np.mean(cia_scores) if cia_scores else 0.0,
            value_correctness=np.mean(vc_scores) if vc_scores else 0.0,
            efficiency_score=np.mean(eff_scores) if eff_scores else 0.0
        )
        
        # Debug output if verbose
        if self.verbose:
            print(f"\n  Aggregated Metrics:")
            print(f"    DGE: {aggregated.dependency_graph_extraction:.1%} (n={len(dge_scores)})")
            print(f"    CIA: {aggregated.cascade_identification_accuracy:.1%} (n={len(cia_scores)})")
            print(f"    VC:  {aggregated.value_correctness:.1%} (n={len(vc_scores)})")
            print(f"    EFF: {aggregated.efficiency_score:.1%} (n={len(eff_scores)})")
            
            # Show distribution if there's variance
            if len(vc_scores) > 1:
                vc_std = np.std(vc_scores)
                if vc_std > 0.2:
                    print(f"    VC variance: std={vc_std:.2f}, min={min(vc_scores):.2f}, max={max(vc_scores):.2f}")
        
        return aggregated