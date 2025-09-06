"""
===== gacot/viz/__init__.py =====
Visualization package for GACoT framework.
Maintains backwards compatibility with original interface.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json

from .charts import (
    create_2x2_heatmap,
    create_capability_gaps_chart,
    create_performance_comparison,
    create_metric_breakdown,
    create_model_comparison
)
from .showcase import create_one_page_showcase
from .results_printer import print_aggregated_results
from .components import DataProcessor


def create_visualizations(
    results: Dict,
    model: str,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create all visualizations for evaluation results.
    Maintains compatibility with original interface.
    
    Args:
        results: Evaluation results dictionary
        model: Model name
        output_dir: Output directory for charts
    """
    if output_dir is None:
        output_dir = Path("results") / model.replace("/", "_")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure results are in dict format
    results_dict = DataProcessor.normalize_results(results)
    
    if not results_dict:
        print("  No valid results to visualize")
        return
    
    print(f"\nCreating visualizations for {model}...")
    
    # Create all chart types
    create_2x2_heatmap(results_dict, model, output_dir)
    create_capability_gaps_chart(results_dict, model, output_dir)
    create_performance_comparison(results_dict, model, output_dir)
    create_metric_breakdown(results_dict, model, output_dir)


def aggregate_model_results(
    models: List[str],
    results_dir: str = "results"
) -> Dict[str, Dict]:
    """
    Aggregate and visualize results from multiple models.
    Maintains compatibility with original interface.
    
    Args:
        models: List of model names
        cache_dir: Cache directory path
        
    Returns:
        Aggregated results dictionary
    """
    all_results = {}
    
    # Load results for each model
    for model in models:
        model_dir = Path("results") / model.replace("/", "_")
        result_files = list(model_dir.glob("results_*.json"))
        
        if result_files:
            latest = sorted(result_files)[-1]
            with open(latest, 'r') as f:
                data = json.load(f)
                all_results[model] = data["results"]
    
    if not all_results:
        print(f"No results found for models: {models}")
        return {}
    
    # Print aggregated results
    print_aggregated_results(all_results)
    
    # Create visualizations
    output_dir = Path(results_dir)
    
    # Original comparison chart
    create_model_comparison(all_results, output_dir)
    
    # Create one-page showcase if multiple models
    if len(all_results) > 1:
        print("\nCreating one-page showcase visualization...")
        create_one_page_showcase(all_results, output_dir)
    
    return all_results


def visualize_cached_results(
    model: str,
    cache_dir: str = "cache"
) -> None:
    """
    Visualize existing cached results.
    Maintains compatibility with original interface.
    
    Args:
        model: Model name
        cache_dir: Cache directory
    """
    model_dir = Path(cache_dir) / model.replace("/", "_")
    result_files = list(model_dir.glob("results_*.json"))
    
    if not result_files:
        print(f"  No results found for {model}")
        return
    
    latest = sorted(result_files)[-1]
    print(f"Loading results from {latest}")
    
    with open(latest, 'r') as f:
        data = json.load(f)
    
    # Create visualizations with the loaded data
    create_visualizations(data.get("results", {}), model)


# Re-export for backwards compatibility
__all__ = [
    'create_visualizations',
    'aggregate_model_results',
    'visualize_cached_results',
    'create_one_page_showcase',
]

