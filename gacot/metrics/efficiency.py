"""
metrics/efficiency.py
Efficiency metric evaluator
"""

import re
from typing import Dict, Set
from .base import MetricEvaluator


class EfficiencyEvaluator(MetricEvaluator):
    """Evaluates computational efficiency and focus."""
    
    def name(self) -> str:
        return "Efficiency"
    
    def evaluate(
        self,
        response: str,
        problem: Dict,
        **kwargs
    ) -> float:
        """
        Evaluate efficiency: focused, minimal calculations without redundancy.
        
        Args:
            response: Model response
            problem: Problem data
            
        Returns:
            Efficiency score between 0 and 1
        """
        lines = response.split('\n')
        
        # Analyze response structure
        calc_lines = 0
        explanation_lines = 0
        redundant_calcs = 0
        seen_calcs = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if self._is_calculation_line(line):
                calc_sig = self._get_calculation_signature(line)
                if calc_sig:
                    if calc_sig in seen_calcs:
                        redundant_calcs += 1
                    seen_calcs.add(calc_sig)
                calc_lines += 1
            else:
                explanation_lines += 1
        
        total_lines = calc_lines + explanation_lines
        if total_lines == 0:
            return 0.0
        
        # Component scores
        focus_ratio = self._calculate_focus_ratio(calc_lines, total_lines)
        redundancy_score = self._calculate_redundancy_score(redundant_calcs)
        conciseness_score = self._calculate_conciseness_score(
            calc_lines,
            self._get_expected_calculations(problem)
        )
        
        # Weighted combination
        efficiency = (
            0.4 * focus_ratio +
            0.3 * conciseness_score +
            0.3 * redundancy_score
        )
        
        # Debug output
        if efficiency < 0.5 and kwargs.get('verbose', False):
            print(f"    Efficiency Debug:")
            print(f"      Focus: {focus_ratio:.2f}, Conciseness: {conciseness_score:.2f}, Non-redundancy: {redundancy_score:.2f}")
            print(f"      Lines: {calc_lines} calc, {explanation_lines} explanation, {redundant_calcs} redundant")
        
        return min(max(efficiency, 0.0), 1.0)
    
    def _is_calculation_line(self, line: str) -> bool:
        """Check if a line contains a calculation."""
        calc_indicators = [
            '=', ':',  # Assignment
            'equals', 'becomes', 'is now',  # Result statements
            'calculate', 'compute',  # Calculation verbs
            re.compile(r'\$[\d,]+'),  # Dollar amounts
            re.compile(r'\d+\s*[%]'),  # Percentages
        ]
        
        for indicator in calc_indicators:
            if isinstance(indicator, str):
                if indicator in line:
                    return True
            else:  # regex
                if indicator.search(line):
                    return True
        
        return False
    
    def _get_calculation_signature(self, line: str) -> str:
        """Extract a signature for a calculation to detect redundancy."""
        # Try to extract the variable being calculated
        patterns = [
            r'^(\w+)\s*[=:]',  # Variable = or Variable:
            r'(?:calculate|computing)\s+(\w+)',  # calculate Variable
            r'(\w+)\s+(?:is|becomes|equals)',  # Variable is/becomes/equals
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return self._normalize_var(match.group(1))
        
        return ""
    
    def _calculate_focus_ratio(self, calc_lines: int, total_lines: int) -> float:
        """Calculate ratio of calculation lines to total."""
        if total_lines == 0:
            return 0.0
        
        ratio = calc_lines / total_lines
        
        # Ideal range is 40-70% calculations
        if 0.4 <= ratio <= 0.7:
            return 1.0
        elif ratio < 0.4:
            # Too much explanation
            return ratio / 0.4
        else:
            # Too dense, no explanation
            return 1.0 - (ratio - 0.7) / 0.3
    
    def _calculate_redundancy_score(self, redundant_count: int) -> float:
        """Calculate score based on redundancy."""
        # Each redundant calculation reduces score
        penalty_per_redundancy = 0.15
        score = 1.0 - (redundant_count * penalty_per_redundancy)
        return max(score, 0.0)
    
    def _calculate_conciseness_score(
        self,
        actual_calcs: int,
        expected_calcs: int
    ) -> float:
        """Calculate score based on conciseness."""
        if expected_calcs == 0:
            # No expected value, use heuristic
            if actual_calcs <= 10:
                return 1.0
            elif actual_calcs <= 20:
                return 0.8
            else:
                return max(0.5 - (actual_calcs - 20) * 0.02, 0.0)
        
        # Compare to expected
        if actual_calcs <= expected_calcs:
            return 1.0
        elif actual_calcs <= expected_calcs * 1.5:
            # Up to 50% more is acceptable
            return 1.0 - (actual_calcs - expected_calcs) / (expected_calcs * 0.5) * 0.3
        else:
            # Too verbose
            return max(0.7 - (actual_calcs - expected_calcs * 1.5) / expected_calcs * 0.2, 0.0)
    
    def _get_expected_calculations(self, problem: Dict) -> int:
        """Get the expected number of calculations for this problem."""
        # Check if explicitly provided
        if "expected_calculations" in problem:
            return problem["expected_calculations"]
        
        # Estimate from expected values
        expected_values = problem.get("expected_values", {})
        if expected_values:
            return len(expected_values)
        
        # Estimate from solution
        if "solution" in problem:
            solution = problem["solution"]
            # Count calculation indicators in solution
            calc_count = len(re.findall(r'[=:]', solution))
            return max(calc_count, 3)  # Minimum 3
        
        # Default estimate
        return 5