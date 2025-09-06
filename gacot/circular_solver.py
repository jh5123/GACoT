"""
Circular Reference Solver
Iterative solving for circular dependencies in financial models
"""

import re
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .dependency_tracker import DependencyTracker
from .llm_client import LLMClient


@dataclass
class SolverConfig:
    """Configuration for circular reference solver."""
    max_iterations: int = 10
    tolerance: float = 0.01  # 1% convergence threshold
    initial_guess: float = 0.0
    verbose: bool = False


class CircularSolver:
    """Solve circular references iteratively."""
    
    def __init__(
        self,
        tracker: DependencyTracker,
        llm: LLMClient,
        config: Optional[SolverConfig] = None
    ):
        """
        Initialize circular solver.
        
        Args:
            tracker: Dependency tracker instance
            llm: LLM client for calculations
            config: Solver configuration
        """
        self.tracker = tracker
        self.llm = llm
        self.config = config or SolverConfig()
    
    def solve(
        self,
        initial_prompt: str
    ) -> Tuple[bool, Dict[str, float], int]:
        """
        Solve circular dependencies iteratively.
        
        Args:
            initial_prompt: Initial calculation prompt
            
        Returns:
            Tuple of (converged, final_values, iterations_used)
        """
        cycles = self.tracker.detect_cycles()
        
        if not cycles:
            # No cycles, solve directly
            response = self.llm.call(initial_prompt)
            values = self._extract_values(response)
            return True, values, 1
        
        if self.config.verbose:
            print(" Circular references detected:")
            for cycle in cycles:
                print(f"   {' → '.join(cycle)}")
            print("  Using iterative solver...")
        
        # Iterative solving
        return self._solve_iteratively(initial_prompt, cycles)
    
    def _solve_iteratively(
        self,
        initial_prompt: str,
        cycles: list
    ) -> Tuple[bool, Dict[str, float], int]:
        """
        Perform iterative solving.
        
        Args:
            initial_prompt: Initial prompt
            cycles: List of circular dependency cycles
            
        Returns:
            Tuple of (converged, final_values, iterations_used)
        """
        # Prepare initial prompt with guidance
        prompt = self._prepare_iterative_prompt(initial_prompt, cycles)
        
        prev_values = {}
        converged = False
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Add previous values to prompt if available
            if prev_values:
                prompt = self._add_iteration_context(
                    initial_prompt, prev_values, iteration
                )
            
            # Get LLM calculation
            response = self.llm.call(prompt)
            current_values = self._extract_values(response)
            
            if self.config.verbose:
                print(f"  Iteration {iteration}: {self._format_values(current_values)}")
            
            # Check convergence
            if self._check_convergence(prev_values, current_values):
                converged = True
                if self.config.verbose:
                    print(f"Converged after {iteration} iterations")
                return True, current_values, iteration
            
            prev_values = current_values
        
        if self.config.verbose:
            print(f"WARNING: Did not converge within {self.config.max_iterations} iterations")
        
        return False, prev_values, self.config.max_iterations
    
    def _prepare_iterative_prompt(
        self,
        prompt: str,
        cycles: list
    ) -> str:
        """Prepare prompt for iterative solving."""
        context = [
            "\n[CIRCULAR REFERENCE SOLVING]",
            "Circular dependencies detected:",
        ]
        
        for cycle in cycles[:3]:  # Show first 3 cycles
            context.append(f"  {' → '.join(cycle)}")
        
        context.extend([
            "",
            "Instructions:",
            f"1. Start with initial guess of {self.config.initial_guess} for unknown values",
            "2. Calculate all values using current estimates",
            "3. Show all intermediate calculations",
            "",
        ])
        
        return prompt + "\n".join(context)
    
    def _add_iteration_context(
        self,
        prompt: str,
        values: Dict[str, float],
        iteration: int
    ) -> str:
        """Add previous iteration values to prompt."""
        context = [
            f"\n[ITERATION {iteration}]",
            "Use these values from previous iteration:",
        ]
        
        for var, val in sorted(values.items()):
            context.append(f"  {var} = ${val:,.2f}")
        
        context.extend([
            "",
            "Recalculate all dependent values using these inputs.",
            ""
        ])
        
        return prompt + "\n".join(context)
    
    def _extract_values(self, text: str) -> Dict[str, float]:
        """
        Extract numerical values from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Dictionary of variable names to values
        """
        values = {}
        
        # Pattern: Variable = $XXX or Variable = XXX
        patterns = [
            r'(\w+)\s*=\s*\$?([\d,]+(?:\.\d+)?)\s*([MB])?',
            r'(\w+):\s*\$?([\d,]+(?:\.\d+)?)\s*([MB])?',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                var = match.group(1)
                value_str = match.group(2).replace(',', '')
                multiplier = match.group(3)
                
                try:
                    value = float(value_str)
                    
                    # Handle M/B suffixes
                    if multiplier == 'M':
                        value *= 1e6
                    elif multiplier == 'B':
                        value *= 1e9
                    
                    values[var] = value
                except ValueError:
                    continue
        
        return values
    
    def _check_convergence(
        self,
        prev: Dict[str, float],
        current: Dict[str, float]
    ) -> bool:
        """
        Check if values have converged.
        
        Args:
            prev: Previous iteration values
            current: Current iteration values
            
        Returns:
            True if converged
        """
        if not prev:
            return False
        
        # Check all common variables
        common_vars = set(prev.keys()) & set(current.keys())
        
        if not common_vars:
            return False
        
        for var in common_vars:
            prev_val = prev[var]
            curr_val = current[var]
            
            # Skip if both are zero
            if prev_val == 0 and curr_val == 0:
                continue
            
            # Check absolute difference for small values
            if abs(prev_val) < 1:
                if abs(curr_val - prev_val) > self.config.tolerance:
                    return False
            else:
                # Check relative difference for larger values
                relative_change = abs(curr_val - prev_val) / abs(prev_val)
                if relative_change > self.config.tolerance:
                    return False
        
        return True
    
    def _format_values(self, values: Dict[str, float]) -> str:
        """Format values for display."""
        if not values:
            return "No values"
        
        items = []
        for var, val in sorted(values.items())[:5]:  # Show first 5
            if val >= 1e9:
                items.append(f"{var}=${val/1e9:.1f}B")
            elif val >= 1e6:
                items.append(f"{var}=${val/1e6:.1f}M")
            elif val >= 1000:
                items.append(f"{var}=${val:,.0f}")
            else:
                items.append(f"{var}=${val:.2f}")
        
        if len(values) > 5:
            items.append(f"... +{len(values)-5} more")
        
        return ", ".join(items)