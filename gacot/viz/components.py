"""
Reusable visualization components and builders.
Provides building blocks for creating consistent charts.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("⚠️  Visualization libraries not available.")
    print("   Install with: pip install matplotlib seaborn pandas")

from .config import VizConfig, ChartConfig


class ChartBuilder:
    """Builder pattern for creating consistent charts."""
    
    def __init__(self, chart_type: str = 'default'):
        """
        Initialize chart builder with configuration.
        
        Args:
            chart_type: Type of chart (for getting config)
        """
        self.config = VizConfig.get_chart_config(chart_type)
        self.fig = None
        self.axes = None
        self.chart_type = chart_type
    
    def create_figure(self, rows: int = 1, cols: int = 1) -> 'ChartBuilder':
        """Create figure with subplots."""
        self.fig, self.axes = plt.subplots(
            rows, cols, 
            figsize=self.config.figsize
        )
        
        # Make axes always iterable
        if rows == 1 and cols == 1:
            self.axes = [[self.axes]]
        elif rows == 1:
            self.axes = [self.axes]
        elif cols == 1:
            self.axes = [[ax] for ax in self.axes]
            
        return self
    
    def add_title(self, title: str, subtitle: Optional[str] = None) -> 'ChartBuilder':
        """Add main title and optional subtitle."""
        if self.fig:
            self.fig.suptitle(
                title, 
                fontsize=self.config.title_fontsize,
                fontweight='bold',
                y=0.98 if subtitle else None
            )
            if subtitle:
                self.fig.text(
                    0.5, 0.94, subtitle,
                    ha='center', 
                    fontsize=self.config.label_fontsize
                )
        return self
    
    def add_heatmap(
        self, 
        ax: Any,
        data: List[List[float]],
        title: str,
        xticklabels: List[str],
        yticklabels: List[str],
        **kwargs
    ) -> 'ChartBuilder':
        """Add a heatmap with consistent styling."""
        # Merge defaults with custom kwargs
        heatmap_config = VizConfig.HEATMAP_DEFAULTS.copy()
        heatmap_config.update(kwargs)
        
        sns.heatmap(
            data,
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            **heatmap_config
        )
        
        ax.set_title(title, fontsize=self.config.label_fontsize, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        return self
    
    def add_bar_chart(
        self,
        ax: Any,
        x_data: Any,
        y_data: Any,
        label: Optional[str] = None,
        color: Optional[str] = None,
        orientation: str = 'vertical',
        add_values: bool = True,
        **kwargs
    ) -> 'ChartBuilder':
        """Add bar chart with consistent styling."""
        # Merge defaults
        bar_config = VizConfig.BAR_DEFAULTS.copy()
        bar_config.update(kwargs)
        
        if orientation == 'vertical':
            bars = ax.bar(x_data, y_data, label=label, color=color, **bar_config)
        else:
            bars = ax.barh(x_data, y_data, label=label, color=color, **bar_config)
        
        # Add value labels if requested
        if add_values:
            self._add_bar_values(ax, bars, orientation)
        
        return self
    
    def add_grouped_bars(
        self,
        ax: Any,
        categories: List[str],
        groups: Dict[str, List[float]],
        width: float = 0.2,
        add_values: bool = True
    ) -> 'ChartBuilder':
        """Add grouped bar chart."""
        x = np.arange(len(categories))
        n_groups = len(groups)
        
        for i, (group_name, values) in enumerate(groups.items()):
            offset = (i - n_groups/2 + 0.5) * width
            color = VizConfig.get_metric_color(group_name.lower())
            
            bars = ax.bar(
                x + offset, values, width,
                label=group_name,
                color=color,
                **VizConfig.BAR_DEFAULTS
            )
            
            if add_values:
                self._add_bar_values(ax, bars, 'vertical')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(loc='upper left', framealpha=0.9)
        
        return self
    
    def add_grid(self, ax: Any) -> 'ChartBuilder':
        """Add grid with consistent styling."""
        ax.grid(True, **VizConfig.GRID_DEFAULTS)
        return self
    
    def remove_spines(self, ax: Any, top: bool = True, right: bool = True) -> 'ChartBuilder':
        """Remove chart spines for cleaner look."""
        if top:
            ax.spines['top'].set_visible(False)
        if right:
            ax.spines['right'].set_visible(False)
        return self
    
    def set_labels(
        self,
        ax: Any,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None
    ) -> 'ChartBuilder':
        """Set axis labels and title."""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=self.config.label_fontsize, fontweight='bold')
        return self
    
    def save(self, output_path: Path, **kwargs) -> 'ChartBuilder':
        """Save figure with consistent settings."""
        save_config = {
            'dpi': self.config.dpi,
            'bbox_inches': self.config.bbox_inches,
            'facecolor': self.config.facecolor
        }
        save_config.update(kwargs)
        
        plt.tight_layout()
        plt.savefig(output_path, **save_config)
        plt.close()
        
        return self
    
    def _add_bar_values(self, ax: Any, bars: Any, orientation: str) -> None:
        """Add value labels to bars."""
        for bar in bars:
            if orientation == 'vertical':
                height = bar.get_height()
                if abs(height) > 0.01:  # Only show if significant
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.1%}' if abs(height) < 1 else f'{height:.0f}',
                        ha='center',
                        va='bottom' if height > 0 else 'top',
                        fontsize=self.config.tick_fontsize,
                        fontweight='bold'
                    )
            else:
                width = bar.get_width()
                if abs(width) > 0.01:
                    ax.text(
                        width,
                        bar.get_y() + bar.get_height()/2.,
                        f'{width:.1%}' if abs(width) < 1 else f'{width:.0f}',
                        ha='left' if width > 0 else 'right',
                        va='center',
                        fontsize=self.config.tick_fontsize,
                        fontweight='bold'
                    )


class DataProcessor:
    """Process and prepare data for visualization."""
    
    @staticmethod
    def normalize_results(results: Dict) -> Dict[str, Dict[str, float]]:
        """
        Normalize results to consistent dict format.
        
        Handles both EvalMetrics objects and dict formats.
        """
        normalized = {}
        
        for condition, data in results.items():
            if hasattr(data, 'to_dict'):
                # It's an EvalMetrics object
                normalized[condition] = data.to_dict()
            elif hasattr(data, '__dict__'):
                # It's an object with attributes
                normalized[condition] = {
                    'dge': getattr(data, 'dependency_graph_extraction', 0),
                    'cia': getattr(data, 'cascade_identification_accuracy', 0),
                    'vc': getattr(data, 'value_correctness', 0),
                    'efficiency': getattr(data, 'efficiency_score', 0)
                }
            elif isinstance(data, dict):
                # Already a dict
                normalized[condition] = data
            else:
                print(f"  Unknown data format for condition '{condition}'")
        
        return normalized
    
    @staticmethod
    def build_2x2_matrix(
        results: Dict[str, Dict[str, float]],
        metric: str
    ) -> List[List[float]]:
        """Build 2x2 matrix for a specific metric."""
        matrix = []
        for row_conditions in VizConfig.get_2x2_layout():
            row_values = []
            for condition in row_conditions:
                if condition in results and metric in results[condition]:
                    value = results[condition][metric]
                else:
                    value = 0.0
                row_values.append(value)
            matrix.append(row_values)
        return matrix
    
    @staticmethod
    def calculate_gaps(
        baseline: Dict[str, float],
        enhanced: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate capability gaps between baseline and enhanced."""
        gaps = {}
        for metric in VizConfig.get_ordered_metrics():
            baseline_val = baseline.get(metric, 0)
            enhanced_val = enhanced.get(metric, 0)
            gaps[metric] = enhanced_val - baseline_val
        return gaps
    
    @staticmethod
    def calculate_component_contributions(
        results: Dict[str, Dict[str, float]]
    ) -> Tuple[float, float, float]:
        """
        Calculate component contributions to value correctness.
        
        Returns:
            Tuple of (runtime_contribution, deps_contribution, synergy)
        """
        baseline_vc = results.get('baseline', {}).get('vc', 0)
        scaffolding_vc = results.get('pure_scaffolding', {}).get('vc', 0)
        learning_vc = results.get('pure_learning', {}).get('vc', 0)
        enhanced_vc = results.get('enhanced', {}).get('vc', 0)
        
        runtime_contribution = scaffolding_vc - baseline_vc
        deps_contribution = learning_vc - baseline_vc
        synergy = enhanced_vc - (baseline_vc + runtime_contribution + deps_contribution)
        
        return runtime_contribution, deps_contribution, synergy
    
    @staticmethod
    def prepare_regression_data(
        all_results: Dict[str, Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for regression analysis.
        
        Returns:
            Tuple of (X_matrix, y_values, condition_labels)
        """
        vc_values = []
        has_deps = []
        has_runtime = []
        conditions_list = []
        
        for model in all_results:
            for condition, values in all_results[model].items():
                vc_values.append(values.get('vc', 0))
                has_deps.append(1 if condition in ['pure_learning', 'enhanced'] else 0)
                has_runtime.append(1 if condition in ['pure_scaffolding', 'enhanced'] else 0)
                conditions_list.append(condition)
        
        # Build design matrix
        X = np.column_stack([
            has_deps,
            has_runtime,
            np.array(has_deps) * np.array(has_runtime)  # Interaction term
        ])
        X = np.column_stack([np.ones(len(vc_values)), X])  # Add intercept
        y = np.array(vc_values)
        
        return X, y, conditions_list
    
    @staticmethod
    def format_value(value: float, value_type: str = 'percentage') -> str:
        """Format a value for display."""
        if value_type == 'percentage':
            return f"{value:.1%}"
        elif value_type == 'currency':
            if value >= 1e9:
                return f"${value/1e9:.1f}B"
            elif value >= 1e6:
                return f"${value/1e6:.1f}M"
            elif value >= 1000:
                return f"${value:,.0f}"
            else:
                return f"${value:.2f}"
        else:
            return f"{value:.3f}"