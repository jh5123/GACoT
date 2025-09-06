"""
One-page showcase visualization.
Creates comprehensive single-page visualization of GACoT findings.
"""

from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns
except ImportError:
    print("⚠️  Matplotlib required for showcase visualization")

from .config import VizConfig
from .components import ChartBuilder, DataProcessor


def create_one_page_showcase(
    all_results: Dict[str, Dict],
    output_dir: Path
) -> Path:
    """
    Create a single-page visualization showcasing the key GACoT findings.
    
    Focus: How components contribute to Value Correctness (VC)
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=VizConfig.CHARTS['one_page_showcase'].figsize)
    fig.suptitle(
        'GACoT: Decomposing Dependency Reasoning Capabilities', 
        fontsize=VizConfig.CHARTS['one_page_showcase'].title_fontsize,
        fontweight='bold',
        y=0.98
    )
    
    # Create grid: 2 main rows with subgrids
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.2, 0.8],
        width_ratios=[1.3, 1.1, 0.6],
        hspace=0.5,
        wspace=0.25
    )
    
    # Normalize all results
    normalized_results = {}
    for model, results in all_results.items():
        normalized_results[model] = DataProcessor.normalize_results(results)
    
    # ========== TOP LEFT: 2x2 VC Heatmap ==========
    _add_vc_heatmap(fig, gs[0, 0], normalized_results)
    
    # ========== TOP MIDDLE: Component Contributions ==========
    _add_component_contributions(fig, gs[0, 1], normalized_results)
    
    # ========== TOP RIGHT: Regression Analysis ==========
    _add_regression_analysis(fig, gs[0, 2], normalized_results)
    
    # ========== BOTTOM LEFT-MIDDLE: Full Model Comparison ==========
    _add_model_comparison(fig, gs[1, :2], normalized_results)
    
    # ========== BOTTOM RIGHT: Key Findings ==========
    _add_key_findings(fig, gs[1, 2], normalized_results)
    
    # Save with high quality
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    
    output_path = output_dir / 'gacot_one_page_showcase.png'
    plt.savefig(
        output_path,
        dpi=VizConfig.CHARTS['one_page_showcase'].dpi,
        bbox_inches='tight',
        pad_inches=0.5,
        facecolor='white'
    )
    plt.close()
    
    print(f"  Saved one-page showcase to {output_path}")
    return output_path


def _add_vc_heatmap(fig, gridspec, all_results: Dict):
    """Add 2x2 VC heatmap averaged across models."""
    ax = fig.add_subplot(gridspec)
    
    # Calculate average VC matrix across all models
    vc_matrices = []
    for model, results in all_results.items():
        matrix = DataProcessor.build_2x2_matrix(results, 'vc')
        vc_matrices.append(matrix)
    
    avg_matrix = np.mean(vc_matrices, axis=0)
    
    # Create heatmap with custom vmin/vmax
    heatmap_config = VizConfig.HEATMAP_DEFAULTS.copy()
    heatmap_config.update({
        'vmin': 0.6,  # Override default
        'vmax': 0.9,  # Override default
    })
    
    sns.heatmap(
        avg_matrix,
        ax=ax,
        xticklabels=['No Runtime', 'Runtime'],
        yticklabels=['No Deps', 'Deps'],
        **heatmap_config
    )
    
    ax.set_title(
        f'Value Correctness: 2×2 Matrix\n(Average across {len(all_results)} models)',
        fontsize=13,
        fontweight='bold'
    )
    
    # Add improvement annotations
    runtime_effect = avg_matrix[0, 1] - avg_matrix[0, 0]
    deps_effect = avg_matrix[1, 0] - avg_matrix[0, 0]
    
    ax.text(0.5, -0.18, f'Runtime Effect: +{runtime_effect:.1%}',
           transform=ax.transAxes, ha='center', fontsize=11,
           color='darkblue', fontweight='bold')
    ax.text(-0.18, 0.5, f'Deps Effect:\n+{deps_effect:.1%}',
           transform=ax.transAxes, ha='center', va='center', fontsize=11,
           color='darkblue', fontweight='bold')


def _add_component_contributions(fig, gridspec, all_results: Dict):
    """Add component contribution analysis."""
    ax = fig.add_subplot(gridspec)
    
    # Calculate contributions for each model
    runtime_contribs = []
    deps_contribs = []
    synergy_effects = []
    model_names = []
    
    for model, results in all_results.items():
        runtime, deps, synergy = DataProcessor.calculate_component_contributions(results)
        runtime_contribs.append(runtime)
        deps_contribs.append(deps)
        synergy_effects.append(synergy)
        model_names.append(VizConfig.format_model_name(model))
    
    # Sort by total improvement
    total_improvements = [r + d + s for r, d, s in 
                         zip(runtime_contribs, deps_contribs, synergy_effects)]
    sorted_indices = np.argsort(total_improvements)
    
    runtime_sorted = [runtime_contribs[i] for i in sorted_indices]
    deps_sorted = [deps_contribs[i] for i in sorted_indices]
    synergy_sorted = [synergy_effects[i] for i in sorted_indices]
    models_sorted = [model_names[i] for i in sorted_indices]
    
    # Create stacked bars
    x = np.arange(len(runtime_sorted))
    width = 0.7
    
    # Calculate averages for legend
    avg_runtime = np.mean(runtime_contribs)
    avg_deps = np.mean(deps_contribs)
    avg_synergy = np.mean(synergy_effects)
    
    bars1 = ax.bar(x, runtime_sorted, width,
                   label=f'Runtime ({avg_runtime:+.1%} avg)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(x, deps_sorted, width, bottom=runtime_sorted,
                   label=f'Dependencies ({avg_deps:+.1%} avg)',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    bottom_for_synergy = [r + d for r, d in zip(runtime_sorted, deps_sorted)]
    bars3 = ax.bar(x, synergy_sorted, width, bottom=bottom_for_synergy,
                   label=f'Synergy ({avg_synergy:+.1%} avg)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_ylabel('Contribution to VC', fontsize=12, fontweight='bold')
    ax.set_title('Component Decomposition\n(How each factor contributes to VC)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.02, 0.35)
    
    # Add average total improvement line
    avg_total = np.mean(total_improvements)
    ax.axhline(y=avg_total, color='darkred', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(len(x)-8.6, avg_total + 0.01, f' Avg Total: {avg_total:.1%}',
           va='center', ha='left', fontsize=10, color='darkred', fontweight='bold')


def _add_regression_analysis(fig, gridspec, all_results: Dict):
    """Add regression analysis visualization."""
    ax = fig.add_subplot(gridspec)
    
    # Prepare regression data
    X, y, conditions_list = DataProcessor.prepare_regression_data(all_results)
    
    # Fit regression model
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ coeffs
    
    # Calculate R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Color points by condition
    color_map = {c: VizConfig.get_condition_color(c) for c in VizConfig.get_ordered_conditions()}
    colors = [color_map[c] for c in conditions_list]
    
    # Plot actual vs predicted
    ax.scatter(y, y_pred, alpha=0.7, s=40, c=colors, edgecolors='black', linewidth=0.5)
    ax.plot([0.6, 0.9], [0.6, 0.9], 'r--', alpha=0.5, linewidth=2)
    
    # Formatting
    ax.set_xlabel('Actual VC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted VC', fontsize=11, fontweight='bold')
    ax.set_title('Regression Model Fit', fontsize=13, fontweight='bold')
    ax.set_xlim(0.62, 0.92)
    ax.set_ylim(0.62, 0.92)
    ax.grid(True, alpha=0.3)
    
    # Add equation
    eq_text = VizConfig.REGRESSION_EQUATION_TEMPLATE.format(
        intercept=coeffs[0],
        deps_coef=coeffs[1],
        runtime_coef=coeffs[2],
        synergy_sign='-' if coeffs[3] < 0 else '+',
        synergy_abs=abs(coeffs[3]),
        r2=r2
    )
    
    ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Add legend
    legend_elements = [
        Patch(facecolor=VizConfig.get_condition_color('baseline'), label='Baseline'),
        Patch(facecolor=VizConfig.get_condition_color('pure_scaffolding'), label='Runtime'),
        Patch(facecolor=VizConfig.get_condition_color('pure_learning'), label='Deps'),
        Patch(facecolor=VizConfig.get_condition_color('enhanced'), label='Both')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)


def _add_model_comparison(fig, gridspec, all_results: Dict):
    """Add full model performance comparison."""
    ax = fig.add_subplot(gridspec)
    
    # Calculate improvements for each model
    metrics = VizConfig.get_ordered_metrics()
    improvements = {metric: [] for metric in metrics}
    model_names = []
    
    for model, results in all_results.items():
        if 'baseline' in results and 'enhanced' in results:
            model_names.append(VizConfig.format_model_name(model))
            for metric in metrics:
                baseline_val = results['baseline'].get(metric, 0)
                enhanced_val = results['enhanced'].get(metric, 0)
                improvements[metric].append(enhanced_val - baseline_val)
    
    # Create grouped bars
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        color = VizConfig.get_metric_color(metric)
        bars = ax.bar(x + offset, improvements[metric], width,
                     label=VizConfig.METRIC_DESCRIPTIONS[metric],
                     color=color, alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on significant bars
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.0%}', ha='center',
                       va='bottom' if height > 0 else 'top',
                       fontsize=8, fontweight='bold')
    
    # Calculate and display averages
    avg_improvements = {metric: np.mean(improvements[metric]) for metric in metrics}
    avg_text = ' | '.join([f'{VizConfig.METRIC_LABELS[m]}: {avg_improvements[m]:+.1%}'
                          for m in metrics])
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (Enhanced - Baseline)', fontsize=12, fontweight='bold')
    ax.set_title(f'Full Metric Improvements by Model\nAverages: {avg_text}',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='upper left', ncol=4, framealpha=0.9, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.2, 0.9)
    
    # Add colored background zones
    ax.axhspan(-0.25, 0, alpha=0.1, color='red')
    ax.axhspan(0, 0.9, alpha=0.1, color='green')


def _add_key_findings(fig, gridspec, all_results: Dict):
    """Add key findings text box."""
    ax = fig.add_subplot(gridspec)
    ax.axis('off')
    
    # Calculate average contributions across all models
    all_runtime = []
    all_deps = []
    all_synergy = []
    
    for model, results in all_results.items():
        runtime, deps, synergy = DataProcessor.calculate_component_contributions(results)
        all_runtime.append(runtime)
        all_deps.append(deps)
        all_synergy.append(synergy)
    
    # Format key findings text
    summary_text = VizConfig.KEY_FINDINGS_TEMPLATE.format(
        runtime_contrib=np.mean(all_runtime),
        deps_contrib=np.mean(all_deps),
        synergy=np.mean(all_synergy)
    )
    
    ax.text(-0.2, 1., summary_text,
           transform=ax.transAxes,
           fontsize=12, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                    edgecolor='black', linewidth=2, alpha=0.95),
           linespacing=1.3, wrap=True)