"""
Visualization configuration and constants.
Styling, colors, and chart settings.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ChartConfig:
    """Configuration for a specific chart type."""
    figsize: Tuple[int, int]
    dpi: int = 150
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    grid_alpha: float = 0.3
    bbox_inches: str = 'tight'
    facecolor: str = 'white'


class VizConfig:
    """Visualization configuration."""
    
    # ==================== COLOR SCHEMES ====================
    
    # Condition colors (for bars, points, etc.)
    CONDITION_COLORS = {
        'baseline': '#95a5a6',         # Gray
        'pure_learning': '#2ecc71',     # Green
        'pure_scaffolding': '#3498db',  # Blue
        'enhanced': '#f39c12',          # Orange/Gold
    }
    
    # Metric colors (for grouped bars)
    METRIC_COLORS = {
        'dge': '#3498db',        # Blue
        'cia': '#9b59b6',        # Purple
        'vc': '#2ecc71',         # Green
        'efficiency': '#f39c12',  # Orange
    }
    
    # Performance colors
    PERFORMANCE_COLORS = {
        'positive': '#2ecc71',   # Green for improvements
        'negative': '#e74c3c',   # Red for regressions
        'neutral': '#95a5a6',    # Gray for no change
    }
    
    # Heatmap colormap
    HEATMAP_COLORMAP = 'RdYlGn'  # Red-Yellow-Green
    
    # ==================== CHART CONFIGURATIONS ====================
    
    CHARTS = {
        '2x2_matrix': ChartConfig(
            figsize=(14, 12),
            title_fontsize=18,
            label_fontsize=14
        ),
        'capability_gaps': ChartConfig(
            figsize=(10, 6),
            title_fontsize=16,
            label_fontsize=12
        ),
        'performance_comparison': ChartConfig(
            figsize=(12, 7),
            title_fontsize=16,
            label_fontsize=12
        ),
        'metric_breakdown': ChartConfig(
            figsize=(14, 10),
            title_fontsize=16,
            label_fontsize=12
        ),
        'model_comparison': ChartConfig(
            figsize=(14, 10),
            title_fontsize=18,
            label_fontsize=12
        ),
        'one_page_showcase': ChartConfig(
            figsize=(16, 10),
            dpi=200,
            title_fontsize=22,
            label_fontsize=13
        ),
        'executive_summary': ChartConfig(
            figsize=(12, 5),
            title_fontsize=16,
            label_fontsize=12
        ),
    }
    
    # ==================== LABELS AND FORMATTING ====================
    
    # Condition display names
    CONDITION_LABELS = {
        'baseline': 'Baseline',
        'pure_learning': 'Learning',
        'pure_scaffolding': 'Scaffolding',
        'enhanced': 'Enhanced',
    }
    
    # Full condition descriptions
    CONDITION_DESCRIPTIONS = {
        'baseline': 'Baseline (No Deps, No Runtime)',
        'pure_scaffolding': 'Pure Scaffolding (No Deps, Runtime)',
        'pure_learning': 'Pure Learning (Deps, No Runtime)',
        'enhanced': 'Enhanced (Deps + Runtime)',
    }
    
    # Metric display names
    METRIC_LABELS = {
        'dge': 'DGE',
        'cia': 'CIA',
        'vc': 'VC',
        'efficiency': 'EFF',
    }
    
    # Full metric names
    METRIC_FULL_NAMES = {
        'dge': 'Dependency Graph Extraction',
        'cia': 'Cascade Identification Accuracy',
        'vc': 'Value Correctness',
        'efficiency': 'Efficiency',
    }
    
    # Metric descriptions
    METRIC_DESCRIPTIONS = {
        'dge': 'Graph Extraction (DGE)',
        'cia': 'Cascade Accuracy (CIA)',
        'vc': 'Value Correctness (VC)',
        'efficiency': 'Computational Efficiency',
    }
    
    # ==================== CHART DEFAULTS ====================
    
    # Heatmap defaults
    HEATMAP_DEFAULTS = {
        'annot': True,
        'fmt': '.1%',
        'cmap': HEATMAP_COLORMAP,
        'vmin': 0,
        'vmax': 1,
        'linewidths': 2,
        'linecolor': 'white',
        'cbar_kws': {'label': 'Score', 'format': '%.0%%'},
    }
    
    # Bar chart defaults
    BAR_DEFAULTS = {
        'alpha': 0.8,
        'edgecolor': 'black',
        'linewidth': 1.5,
    }
    
    # Grid defaults
    GRID_DEFAULTS = {
        'alpha': 0.3,
        'axis': 'y',
    }
    
    # ==================== TEXT TEMPLATES ====================
    
    KEY_FINDINGS_TEMPLATE = """Key Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Runtime scaffolding: {runtime_contrib:+.1%} avg VC contribution
✓ Dependencies training: {deps_contrib:+.1%} avg VC contribution
✗ Negative synergy: {synergy:+.1%} (diminishing returns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implication: Components partially overlap in their approach to solving dependency reasoning"""
    
    REGRESSION_EQUATION_TEMPLATE = """VC = {intercept:.3f}
  + {deps_coef:.3f}·Deps
  + {runtime_coef:.3f}·Runtime
  {synergy_sign} {synergy_abs:.3f}·(D×R)

R² = {r2:.3f}"""
    
    # ==================== HELPER METHODS ====================
    
    @classmethod
    def get_chart_config(cls, chart_type: str) -> ChartConfig:
        """Get configuration for a specific chart type."""
        return cls.CHARTS.get(chart_type, ChartConfig(figsize=(12, 8)))
    
    @classmethod
    def get_condition_color(cls, condition: str) -> str:
        """Get color for a condition."""
        return cls.CONDITION_COLORS.get(condition, '#95a5a6')
    
    @classmethod
    def get_metric_color(cls, metric: str) -> str:
        """Get color for a metric."""
        return cls.METRIC_COLORS.get(metric, '#3498db')
    
    @classmethod
    def format_metric_value(cls, value: float, as_percentage: bool = True) -> str:
        """Format a metric value for display."""
        if as_percentage:
            return f"{value:.1%}"
        else:
            return f"{value:.3f}"
    
    @classmethod
    def format_model_name(cls, model: str) -> str:
        """Format model name for display."""
        return model.replace('gpt-', '').replace('/', '_')
    
    @classmethod
    def get_ordered_conditions(cls) -> List[str]:
        """Get conditions in standard order."""
        return ['baseline', 'pure_learning', 'pure_scaffolding', 'enhanced']
    
    @classmethod
    def get_ordered_metrics(cls) -> List[str]:
        """Get metrics in standard order (reordered for new layout)."""
        return ['vc', 'efficiency', 'dge', 'cia']
    
    @classmethod
    def get_2x2_layout(cls) -> List[List[str]]:
        """Get 2x2 matrix layout of conditions."""
        return [
            ['baseline', 'pure_scaffolding'],
            ['pure_learning', 'enhanced']
        ]