"""
Individual chart creation functions.
Each function creates a specific type of visualization.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from .config import VizConfig
from .components import ChartBuilder, DataProcessor


def create_2x2_heatmap(
    results: Dict[str, Dict[str, float]],
    model: str,
    output_dir: Path
) -> None:
    """Create 2x2 evaluation matrix heatmap."""
    # Normalize results
    results = DataProcessor.normalize_results(results)
    
    # Build chart
    builder = ChartBuilder('2x2_matrix')
    builder.create_figure(2, 2)
    builder.add_title(f'GACoT 2×2 Evaluation: {model}')
    
    # Create heatmaps for each metric
    metrics = VizConfig.get_ordered_metrics()
    
    for idx, metric in enumerate(metrics):
        ax = builder.axes[idx // 2][idx % 2]
        
        # Build matrix for this metric
        matrix = DataProcessor.build_2x2_matrix(results, metric)
        
        # Create heatmap
        builder.add_heatmap(
            ax,
            matrix,
            VizConfig.METRIC_DESCRIPTIONS[metric],
            xticklabels=['No Runtime', 'Runtime'],
            yticklabels=['No Deps', 'Deps']
        )
    
    # Save
    output_path = output_dir / f'{VizConfig.format_model_name(model)}_2x2_matrix.png'
    builder.save(output_path)
    print(f"  Saved 2x2 matrix to {output_path}")


def create_capability_gaps_chart(
    results: Dict[str, Dict[str, float]],
    model: str,
    output_dir: Path
) -> None:
    """Create capability gaps visualization."""
    results = DataProcessor.normalize_results(results)
    
    baseline = results.get('baseline', {})
    enhanced = results.get('enhanced', {})
    
    if not baseline or not enhanced:
        print("   Missing baseline or enhanced results for gap analysis")
        return
    
    # Calculate gaps
    gaps = DataProcessor.calculate_gaps(baseline, enhanced)
    
    # Build chart
    builder = ChartBuilder('capability_gaps')
    builder.create_figure()
    ax = builder.axes[0][0]
    
    # Prepare data
    labels = [VizConfig.METRIC_DESCRIPTIONS[m].replace(' ', '\n') 
              for m in gaps.keys()]
    values = list(gaps.values())
    colors = [VizConfig.PERFORMANCE_COLORS['positive'] if v > 0 
              else VizConfig.PERFORMANCE_COLORS['negative'] 
              for v in values]
    
    # Create bars
    builder.add_bar_chart(
        ax, 
        range(len(labels)), 
        values,
        color=colors
    )
    
    # Formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    builder.set_labels(
        ax,
        ylabel='Improvement (Enhanced - Baseline)',
        title=f'Capability Gaps: {model}'
    )
    builder.add_grid(ax)
    builder.remove_spines(ax)
    
    # Add overall gap annotation
    overall_gap = sum(values) / len(values) if values else 0
    ax.text(0.98, 0.98, f'Overall Gap: {overall_gap:+.1%}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    # Save
    output_path = output_dir / f'{VizConfig.format_model_name(model)}_capability_gaps.png'
    builder.save(output_path)
    print(f"  Saved capability gaps to {output_path}")


def create_performance_comparison(
    results: Dict[str, Dict[str, float]],
    model: str,
    output_dir: Path
) -> None:
    """Create performance comparison across conditions."""
    results = DataProcessor.normalize_results(results)
    
    builder = ChartBuilder('performance_comparison')
    builder.create_figure()
    ax = builder.axes[0][0]
    
    # Prepare data
    conditions = VizConfig.get_ordered_conditions()
    condition_names = [VizConfig.CONDITION_LABELS[c] for c in conditions]
    
    # Build grouped data
    metrics_data = {}
    for metric in VizConfig.get_ordered_metrics():
        metric_values = []
        for condition in conditions:
            value = results.get(condition, {}).get(metric, 0.0)
            metric_values.append(value)
        metrics_data[VizConfig.METRIC_LABELS[metric]] = metric_values
    
    # Create grouped bars
    builder.add_grouped_bars(
        ax,
        condition_names,
        metrics_data
    )
    
    # Formatting
    builder.set_labels(
        ax,
        xlabel='Condition',
        ylabel='Score',
        title=f'Performance Comparison: {model}'
    )
    builder.add_grid(ax)
    builder.remove_spines(ax)
    
    ax.set_ylim(0, 1.15)
    
    # Save
    output_path = output_dir / f'{VizConfig.format_model_name(model)}_performance_comparison.png'
    builder.save(output_path)
    print(f"  Saved performance comparison to {output_path}")


def create_metric_breakdown(
    results: Dict[str, Dict[str, float]],
    model: str,
    output_dir: Path
) -> None:
    """Create detailed metric breakdown chart."""
    results = DataProcessor.normalize_results(results)
    
    if 'enhanced' not in results or 'baseline' not in results:
        return
    
    builder = ChartBuilder('metric_breakdown')
    builder.create_figure(2, 2)
    builder.add_title(f'Metric Analysis: {model}')
    
    conditions = VizConfig.get_ordered_conditions()
    condition_labels = [VizConfig.CONDITION_LABELS[c] for c in conditions]
    
    # Create subplot for each metric
    metrics = VizConfig.get_ordered_metrics()
    
    for idx, metric in enumerate(metrics):
        ax = builder.axes[idx // 2][idx % 2]
        
        # Get values for this metric
        values = []
        for condition in conditions:
            value = results.get(condition, {}).get(metric, 0.0)
            values.append(value)
        
        # Create bars with condition colors
        for i, (label, value, condition) in enumerate(zip(condition_labels, values, conditions)):
            color = VizConfig.get_condition_color(condition)
            ax.bar(i, value, color=color, alpha=0.8, 
                  edgecolor='black', linewidth=1)
            
            # Add value label
            ax.text(i, value, f'{value:.1%}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Formatting
        ax.set_title(VizConfig.METRIC_FULL_NAMES[metric], fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(condition_labels)))
        ax.set_xticklabels(condition_labels)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Save
    output_path = output_dir / f'{VizConfig.format_model_name(model)}_metric_breakdown.png'
    builder.save(output_path)
    print(f"  Saved metric breakdown to {output_path}")


def create_model_comparison(
    all_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """Create multi-model comparison chart."""
    builder = ChartBuilder('model_comparison')
    builder.create_figure(2, 2)
    builder.add_title('Multi-Model Comparison')
    
    metrics = VizConfig.get_ordered_metrics()
    
    for idx, metric in enumerate(metrics):
        ax = builder.axes[idx // 2][idx % 2]
        
        model_names = []
        baseline_scores = []
        enhanced_scores = []
        
        for model, results in all_results.items():
            results = DataProcessor.normalize_results(results)
            
            if 'baseline' in results and 'enhanced' in results:
                model_names.append(VizConfig.format_model_name(model))
                baseline_scores.append(results['baseline'].get(metric, 0))
                enhanced_scores.append(results['enhanced'].get(metric, 0))
        
        if not model_names:
            continue
        
        # Create grouped bars
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_scores, width, 
                      label='Baseline', 
                      color=VizConfig.PERFORMANCE_COLORS['negative'], 
                      alpha=0.8)
        bars2 = ax.bar(x + width/2, enhanced_scores, width, 
                      label='Enhanced', 
                      color=VizConfig.PERFORMANCE_COLORS['positive'], 
                      alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(VizConfig.METRIC_FULL_NAMES[metric], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.0%}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )
    
    # Save
    save_path = output_dir / 'model_comparison.png'
    builder.save(save_path)
    print(f"  Saved model comparison to {save_path}")