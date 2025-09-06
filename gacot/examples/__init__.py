"""
Module: __init__
Purpose: Component of GACoT framework
Author: GACoT Framework
Date: 2025
"""


# ============================================================================
# gacot/examples/__init__.py
# ============================================================================
"""
GACoT Examples and Demonstrations

This module contains demonstrations of GACoT's advanced capabilities,
particularly focusing on complex scenarios that showcase the framework's
ability to handle challenging financial modeling problems.

Available Demos:
- Circular Reference Solving: LBO, tax shields, working capital
- Dependency Tracking: Complex multi-variable models
- Cascade Identification: Change propagation analysis
"""

from .circular_demo import CircularReferenceDemo

__all__ = [
    "CircularReferenceDemo",
]

# Optional: Provide convenient access to demos
def run_circular_demo():
    """Quick function to run circular reference demonstrations."""
    demo = CircularReferenceDemo()
    demo.run_all_demos()


def run_lbo_demo():
    """Run just the LBO debt sizing demonstration."""
    demo = CircularReferenceDemo()
    return demo.demo_lbo_debt_sizing()