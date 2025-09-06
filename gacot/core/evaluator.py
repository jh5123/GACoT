"""
Main Evaluation Engine
Coordinates the 2x2 evaluation framework
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from ..llm_client import LLMClient
from ..metrics import EvalMetrics, MetricsCalculator  # Use modular metrics
from .cache_manager import CacheManager
from .test_data import TestDataManager


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, results: Dict[str, EvalMetrics]):
        self.results = results
        self.baseline = results.get("baseline")
        self.enhanced = results.get("enhanced")
        
        if self.baseline and self.enhanced:
            self.capability_gaps = {
                "graph_extraction": self.enhanced.dependency_graph_extraction - self.baseline.dependency_graph_extraction,
                "cascade_accuracy": self.enhanced.cascade_identification_accuracy - self.baseline.cascade_identification_accuracy,
                "value_correctness": self.enhanced.value_correctness - self.baseline.value_correctness,
                "efficiency": self.enhanced.efficiency_score - self.baseline.efficiency_score
            }
            self.overall_gap = sum(self.capability_gaps.values()) / len(self.capability_gaps)
        else:
            self.capability_gaps = {}
            self.overall_gap = 0.0
    
    @property
    def capability_gap(self):
        """Alias for overall_gap for backwards compatibility."""
        return self.overall_gap


class FinancialReasoningEval:
    """GACoT evaluation framework with 2x2 matrix analysis."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
        use_live_api: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            verbose: Enable verbose output
            use_live_api: Force API usage instead of cache
        """
        self.model = model
        self.verbose = verbose
        
        self.cache_manager = CacheManager(model)
        self.metrics_calc = MetricsCalculator(verbose=verbose)  # Pass verbose to calculator
        self.test_data_mgr = TestDataManager()
        
        if use_live_api and os.getenv("OPENAI_API_KEY"):
            self.llm = LLMClient(provider="openai", model=model)
            print(f"Using OpenAI API with {model}")
        else:
            self.llm = None
            if self.cache_manager.has_cache():
                print("Using cached responses")
            else:
                print("Using mock responses (no cache found)")
        
        self.results = {}
    
    def run_2x2_evaluation(
        self,
        dataset_path: str = "data/testset",
        num_problems: Optional[int] = None
    ) -> EvaluationResult:
        """
        Run 2x2 evaluation matrix.
        
        Args:
            dataset_path: Path to test dataset
            num_problems: Limit number of problems (None for all)
            
        Returns:
            Evaluation results
        """
        print("\n" + "=" * 60)
        print(" GACoT 2×2 Evaluation Matrix")
        print("=" * 60)
        
        test_problems = self.test_data_mgr.load_problems(dataset_path)
        if num_problems:
            test_problems = test_problems[:num_problems]
        print(f"Evaluating {len(test_problems)} problems")
        
        # Validate test data structure
        self._validate_test_data(test_problems)
        
        conditions = {
            "baseline": (False, False),
            "pure_learning": (True, False),
            "pure_scaffolding": (False, True),
            "enhanced": (True, True)
        }
        
        for condition_name, (use_deps, use_runtime) in conditions.items():
            print(f"\n{condition_name.upper()}")
            print("-" * 40)
            
            metrics = self._evaluate_condition(
                test_problems,
                use_deps_in_prompt=use_deps,
                use_runtime_tracking=use_runtime,
                condition_name=condition_name
            )
            
            self.results[condition_name] = metrics
            
            print(f"  DGE: {metrics.dependency_graph_extraction:.1%}")
            print(f"  CIA: {metrics.cascade_identification_accuracy:.1%}")
            print(f"  VC:  {metrics.value_correctness:.1%}")
            print(f"  EFF: {metrics.efficiency_score:.1%}")
        
        result = self._analyze_results()
        
        return result
    
    def _validate_test_data(self, problems: List[Dict]) -> None:
        """Validate test data structure and warn about missing fields."""
        if not problems:
            return
        
        sample = problems[0]
        warnings = []
        
        # Check for expected structure
        if "dependencies" not in sample:
            warnings.append("No 'dependencies' field - DGE and CIA metrics will be limited")
        elif "graph" not in sample.get("dependencies", {}):
            warnings.append("No dependency graph - DGE metric will use pattern matching only")
        
        if "expected_values" not in sample and "solution" not in sample:
            warnings.append("No 'expected_values' or 'solution' - VC metric will return 0.5 (uncertain)")
        
        if warnings and self.verbose:
            print("\n⚠️  Test Data Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print()
    
    def _evaluate_condition(
        self,
        problems: List[Dict],
        use_deps_in_prompt: bool,
        use_runtime_tracking: bool,
        condition_name: str
    ) -> EvalMetrics:
        """Evaluate one condition of the 2x2 matrix."""
        all_metrics = []
        api_calls = 0
        cache_hits = 0
        
        iterator = tqdm(problems, desc=f"  {condition_name}", leave=False) if TQDM_AVAILABLE else problems
        
        for problem in iterator:
            prompt = self._prepare_prompt(problem, use_deps_in_prompt, use_runtime_tracking)
            
            response, from_cache = self._get_response(prompt)
            
            if from_cache:
                cache_hits += 1
            else:
                api_calls += 1
            
            # Use the modular metrics calculator
            metrics = self.metrics_calc.evaluate_response(
                response, 
                problem, 
                use_runtime=use_runtime_tracking
            )
            all_metrics.append(metrics)
        
        print(f"  API: {api_calls}, Cache: {cache_hits}")
        
        # Aggregate using the calculator
        return self.metrics_calc.aggregate_metrics(all_metrics)
    
    def _prepare_prompt(
        self,
        problem: Dict,
        use_deps: bool,
        use_runtime: bool
    ) -> str:
        """Prepare prompt based on condition."""
        base_prompt = problem["question"]
        
        if "initial_values" in problem:
            base_prompt += "\n\nGiven values:"
            for var, val in problem["initial_values"].items():
                base_prompt += f"\n- {var} = ${val:,.0f}" if val >= 1000 else f"\n- {var} = {val}"
        
        if use_deps and "dependencies" in problem and "graph" in problem["dependencies"]:
            deps_text = "\n\nDependency structure:"
            for var, deps in problem["dependencies"]["graph"].items():
                if deps:
                    deps_text += f"\n- {var} depends on: {', '.join(deps)}"
            base_prompt = deps_text + "\n" + base_prompt
        
        if use_runtime:
            base_prompt += "\n\nIMPORTANT: Track and explicitly state all calculation dependencies. When any value changes, identify all variables that need recalculation."
        
        return base_prompt
    
    def _get_response(self, prompt: str) -> Tuple[str, bool]:
        """Get response from LLM or cache."""
        cached = self.cache_manager.get(prompt)
        if cached:
            return cached, True
        
        if self.llm:
            response = self.llm.call(prompt, temperature=0.0, max_tokens=1500)
            self.cache_manager.save(prompt, response)
            return response, False
        
        from ..llm_client import LLMClient
        mock_llm = LLMClient(provider="mock")
        response = mock_llm.call(prompt)
        return response, False
    
    def _analyze_results(self) -> EvaluationResult:
        """Analyze and display results."""
        print("\n" + "=" * 60)
        print(" RESULTS SUMMARY")
        print("=" * 60)
        
        print("\n2×2 Matrix Results:")
        print("                    | No Runtime | Runtime")
        print("--------------------|------------|----------")
        
        metrics_order = ["dependency_graph_extraction", "cascade_identification_accuracy", "value_correctness", "efficiency_score"]
        metric_names = ["DGE", "CIA", "VC", "EFF"]
        
        for metric, name in zip(metrics_order, metric_names):
            no_deps_no_runtime = getattr(self.results.get("baseline"), metric, 0)
            no_deps_runtime = getattr(self.results.get("pure_scaffolding"), metric, 0)
            deps_no_runtime = getattr(self.results.get("pure_learning"), metric, 0)
            deps_runtime = getattr(self.results.get("enhanced"), metric, 0)
            
            print(f"{name}:")
            print(f"  No Dependencies   |   {no_deps_no_runtime:.1%}    |  {no_deps_runtime:.1%}")
            print(f"  Dependencies      |   {deps_no_runtime:.1%}    |  {deps_runtime:.1%}")
        
        result = EvaluationResult(self.results)
        
        if result.baseline and result.enhanced:
            print("\n" + "=" * 60)
            print(" CAPABILITY GAPS (Enhanced - Baseline)")
            print("=" * 60)
            
            for metric, gap in result.capability_gaps.items():
                metric_display = metric.replace("_", " ").title()
                print(f"  {metric_display}: {gap:+.1%}")
            
            print(f"\n  Overall Gap: {result.overall_gap:+.1%}")
            
            print("\n" + "=" * 60)
            print(" KEY FINDING")
            print("=" * 60)
            
            if result.overall_gap > 0.3:
                print("  LLMs CANNOT intrinsically track financial dependencies.")
                print("  They require explicit scaffolding for calculation consistency.")
            elif result.overall_gap > 0.15:
                print("  LLMs show LIMITED dependency tracking capability.")
                print("  Scaffolding provides significant improvements.")
            else:
                print("  LLMs demonstrate some intrinsic dependency awareness.")
                print("  Scaffolding provides marginal improvements.")
        
        self._save_results(result)
        
        return result
    
    def _save_results(self, result: EvaluationResult) -> None:
        """Save results to file."""
        results_data = {
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "results": {
                condition: {
                    "dge": metrics.dependency_graph_extraction,
                    "cia": metrics.cascade_identification_accuracy,
                    "vc": metrics.value_correctness,
                    "efficiency": metrics.efficiency_score
                }
                for condition, metrics in self.results.items()
            },
            "capability_gaps": result.capability_gaps,
            "overall_gap": result.overall_gap
        }
        
        # Create results directory structure
        results_dir = Path("results") / self.model.replace("/", "_")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {results_file}")