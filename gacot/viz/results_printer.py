
"""
===== gacot/viz/results_printer.py =====
Results printing and formatting utilities.
"""

from typing import Dict, List
import numpy as np

from .config import VizConfig
from .components import DataProcessor


def print_aggregated_results(all_results: Dict[str, Dict]) -> None:
    """
    Print detailed aggregated results for analysis and README.
    
    Args:
        all_results: Dictionary of model results
    """
    # Print detailed results
    print("\n" + "="*80)
    print("AGGREGATED MODEL RESULTS")
    print("="*80)
    
    _print_2x2_matrices(all_results)
    _print_regression_data(all_results)
    _print_component_analysis(all_results)
    _print_summary_statistics(all_results)
    _print_readme_ready(all_results)


def _print_2x2_matrices(all_results: Dict[str, Dict]) -> None:
    """Print 2x2 matrix results for each model."""
    print("\n## 2×2 Evaluation Matrix Results\n")
    
    conditions = VizConfig.get_ordered_conditions()
    metrics = ['dge', 'cia', 'vc', 'efficiency']
    
    for model in all_results:
        print(f"\n### {model}\n")
        
        # Normalize results
        results = DataProcessor.normalize_results(all_results[model])
        
        # Print table header
        print("| Condition | DGE | CIA | VC | Efficiency |")
        print("|-----------|-----|-----|-----|------------|")
        
        model_data = {}
        for condition in conditions:
            if condition in results:
                result = results[condition]
                
                # Store for later calculations
                model_data[condition] = result
                
                # Print row
                label = VizConfig.CONDITION_DESCRIPTIONS[condition]
                row = f"| {label:<40} |"
                for metric in metrics:
                    value = result.get(metric, 0)
                    row += f" {value:.1%} |"
                print(row)
        
        # Calculate and print improvements
        if 'baseline' in model_data and 'enhanced' in model_data:
            print(f"\n**Improvements (Enhanced vs Baseline):**")
            gaps = DataProcessor.calculate_gaps(
                model_data['baseline'],
                model_data['enhanced']
            )
            for metric in metrics:
                if metric in gaps:
                    baseline_val = model_data['baseline'][metric]
                    enhanced_val = model_data['enhanced'][metric]
                    improvement = gaps[metric]
                    print(f"- {VizConfig.METRIC_FULL_NAMES[metric]}: "
                          f"{improvement:+.1%} ({baseline_val:.1%} → {enhanced_val:.1%})")


def _print_regression_data(all_results: Dict[str, Dict]) -> None:
    """Print regression-ready data."""
    print("\n" + "="*80)
    print("REGRESSION ANALYSIS DATA")
    print("="*80)
    
    print("\n## Value Correctness (VC) Analysis\n")
    print("For linear regression: VC ~ β₀ + β₁(has_deps) + β₂(has_runtime) + β₃(has_deps × has_runtime)\n")
    
    # Prepare regression data
    print("\n### CSV Format (for regression):\n")
    print("model,condition,has_deps,has_runtime,dge,cia,vc,efficiency")
    
    for model in all_results:
        results = DataProcessor.normalize_results(all_results[model])
        for condition, values in results.items():
            has_deps = 1 if condition in ['pure_learning', 'enhanced'] else 0
            has_runtime = 1 if condition in ['pure_scaffolding', 'enhanced'] else 0
            
            print(f"{model},{condition},{has_deps},{has_runtime},"
                  f"{values.get('dge', 0):.3f},{values.get('cia', 0):.3f},"
                  f"{values.get('vc', 0):.3f},{values.get('efficiency', 0):.3f}")


def _print_component_analysis(all_results: Dict[str, Dict]) -> None:
    """Print component contribution analysis."""
    print("\n## Component Contribution Analysis\n")
    
    for model in all_results:
        results = DataProcessor.normalize_results(all_results[model])
        
        if all(c in results for c in VizConfig.get_ordered_conditions()):
            print(f"\n### {model}")
            
            runtime, deps, synergy = DataProcessor.calculate_component_contributions(results)
            baseline_vc = results['baseline'].get('vc', 0)
            enhanced_vc = results['enhanced'].get('vc', 0)
            
            print(f"- Baseline VC: {baseline_vc:.1%}")
            print(f"- Runtime Contribution: {runtime:+.1%}")
            print(f"- Dependencies Contribution: {deps:+.1%}")
            print(f"- Synergy Effect: {synergy:+.1%}")
            print(f"- Total (Enhanced): {enhanced_vc:.1%}")
            
            # Verify decomposition
            reconstructed = baseline_vc + runtime + deps + synergy
            print(f"- Verification: {reconstructed:.1%} (should equal Enhanced)")


def _print_summary_statistics(all_results: Dict[str, Dict]) -> None:
    """Print summary statistics across models."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if len(all_results) <= 1:
        return
    
    print("\n## Average Performance Across Models\n")
    
    conditions = VizConfig.get_ordered_conditions()
    metrics = ['dge', 'cia', 'vc', 'efficiency']
    
    # Collect all values
    avg_results = {
        condition: {metric: [] for metric in metrics}
        for condition in conditions
    }
    
    for model, model_results in all_results.items():
        results = DataProcessor.normalize_results(model_results)
        for condition in conditions:
            if condition in results:
                for metric in metrics:
                    avg_results[condition][metric].append(
                        results[condition].get(metric, 0)
                    )
    
    # Print averages table
    print("| Condition | DGE | CIA | VC | Efficiency |")
    print("|-----------|-----|-----|-----|------------|")
    
    for condition in conditions:
        label = VizConfig.CONDITION_DESCRIPTIONS[condition]
        row = f"| {label:<40} |"
        
        for metric in metrics:
            values = avg_results[condition][metric]
            if values:
                avg = np.mean(values)
                std = np.std(values)
                row += f" {avg:.1%}±{std:.1%} |"
            else:
                row += " N/A |"
        print(row)
    
    # Print average improvements
    print("\n## Average Improvements (Enhanced vs Baseline)\n")
    
    improvements = {metric: [] for metric in metrics}
    
    for model, model_results in all_results.items():
        results = DataProcessor.normalize_results(model_results)
        if 'baseline' in results and 'enhanced' in results:
            for metric in metrics:
                baseline_val = results['baseline'].get(metric, 0)
                enhanced_val = results['enhanced'].get(metric, 0)
                improvements[metric].append(enhanced_val - baseline_val)
    
    for metric in metrics:
        values = improvements[metric]
        if values:
            avg = np.mean(values)
            std = np.std(values)
            print(f"- {VizConfig.METRIC_FULL_NAMES[metric]}: {avg:+.1%} (±{std:.1%})")


def _print_readme_ready(all_results: Dict[str, Dict]) -> None:
    """Print README-ready markdown results."""
    print("\n" + "="*80)
    print("README-READY RESULTS")
    print("="*80)
    
    print("\n```markdown")
    print("### Results\n")
    
    if len(all_results) == 1:
        # Single model results
        model = list(all_results.keys())[0]
        results = DataProcessor.normalize_results(all_results[model])
        
        if 'baseline' in results and 'enhanced' in results:
            print(f"**{model}**\n")
            
            print("**Baseline** (no dependencies, no runtime):")
            baseline = results['baseline']
            print(f"- Dependency Extraction: **{baseline.get('dge', 0):.1%}** F1")
            print(f"- Cascade Prediction: **{baseline.get('cia', 0):.1%}** Accuracy")
            print(f"- Value Correctness: **{baseline.get('vc', 0):.1%}** Accuracy")
            print(f"- Efficiency: **{baseline.get('efficiency', 0):.1%}** Score\n")
            
            print("**With GACoT** (dependencies + runtime):")
            enhanced = results['enhanced']
            for metric, name, unit in [
                ('dge', 'Dependency Extraction', 'F1'),
                ('cia', 'Cascade Prediction', 'Accuracy'),
                ('vc', 'Value Correctness', 'Accuracy'),
                ('efficiency', 'Efficiency', 'Score')
            ]:
                enhanced_val = enhanced.get(metric, 0)
                improvement = enhanced_val - baseline.get(metric, 0)
                print(f"- {name}: **{enhanced_val:.1%}** {unit} "
                      f"({improvement:+.1%})")
    else:
        # Multiple models - show averages
        print("**Average across models:**")
        
        # Calculate averages
        metrics = ['dge', 'cia', 'vc', 'efficiency']
        baseline_avgs = {metric: [] for metric in metrics}
        enhanced_avgs = {metric: [] for metric in metrics}
        
        for model, model_results in all_results.items():
            results = DataProcessor.normalize_results(model_results)
            if 'baseline' in results and 'enhanced' in results:
                for metric in metrics:
                    baseline_avgs[metric].append(results['baseline'].get(metric, 0))
                    enhanced_avgs[metric].append(results['enhanced'].get(metric, 0))
        
        if baseline_avgs['dge']:  # Check if we have data
            print("\n**Baseline** (no dependencies, no runtime):")
            print(f"- Dependency Extraction: **{np.mean(baseline_avgs['dge']):.1%}** F1")
            print(f"- Cascade Prediction: **{np.mean(baseline_avgs['cia']):.1%}** Accuracy")
            print(f"- Value Correctness: **{np.mean(baseline_avgs['vc']):.1%}** Accuracy")
            print(f"- Efficiency: **{np.mean(baseline_avgs['efficiency']):.1%}** Score\n")
            
            print("**With GACoT** (dependencies + runtime):")
            for metric, label, unit in [
                ('dge', 'Dependency Extraction', 'F1'),
                ('cia', 'Cascade Prediction', 'Accuracy'),
                ('vc', 'Value Correctness', 'Accuracy'),
                ('efficiency', 'Efficiency', 'Score')
            ]:
                avg_baseline = np.mean(baseline_avgs[metric])
                avg_enhanced = np.mean(enhanced_avgs[metric])
                improvement = avg_enhanced - avg_baseline
                print(f"- {label}: **{avg_enhanced:.1%}** {unit} ({improvement:+.1%})")
    
    print("```")