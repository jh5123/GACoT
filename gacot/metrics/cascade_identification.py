"""
metrics/cascade_identification.py
Cascade Identification Accuracy (CIA) metric evaluator
"""

import re
from typing import Dict, List, Optional
from .base import MetricEvaluator


class CascadeIdentificationEvaluator(MetricEvaluator):
    """Evaluates cascade identification accuracy."""
    
    def name(self) -> str:
        return "Cascade Identification Accuracy"
    
    def evaluate(
        self,
        response: str,
        problem: Dict,
        **kwargs
    ) -> float:
        """
        Binary test: When X changes, does model identify affected variables?
        
        Args:
            response: Model response
            problem: Problem with cascade ground truth
            
        Returns:
            Accuracy score between 0 and 1
        """
        gold_cascades = problem.get("dependencies", {}).get("cascades", [])
        
        if not gold_cascades:
            # No cascades to test
            return 1.0
        
        correct = 0
        total = 0
        errors = []
        
        for cascade in gold_cascades:
            change_var = cascade.get("change", "")
            affects = cascade.get("affects", [])
            
            if not change_var or not affects:
                continue
            
            # Find where the change is mentioned
            change_context = self._find_change_mention(response, change_var)
            
            if change_context is not None:
                # Check if each affected variable is mentioned as needing update
                for affected in affects:
                    total += 1
                    if self._verify_cascade_mentioned(
                        response, 
                        change_var, 
                        affected, 
                        change_context
                    ):
                        correct += 1
                    else:
                        errors.append(f"{change_var}->{affected}")
            else:
                # Change not mentioned at all
                total += len(affects)
                errors.extend([f"{change_var}->{a}" for a in affects])
        
        if total == 0:
            return 1.0
        
        accuracy = correct / total
        
        # Debug output for low scores
        if accuracy < 0.5 and kwargs.get('verbose', False):
            print(f"    CIA Debug: {correct}/{total} cascades identified")
            if errors:
                print(f"      Missed cascades: {', '.join(errors[:3])}")
        
        return accuracy
    
    def _find_change_mention(self, response: str, variable: str) -> Optional[int]:
        """
        Find where a variable change is mentioned in the response.
        
        Returns:
            Position in text where change is mentioned, or None
        """
        patterns = [
            f'{variable}.*(?:changes|increases|decreases|becomes|updates|modified)',
            f'(?:change|update|modify|adjust|alter).*{variable}',
            f'new.*{variable}',
            f'{variable}.*(?:from|to).*\\d+',
            f'if {variable}',
            f'when {variable}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.start()
        
        return None
    
    def _verify_cascade_mentioned(
        self, 
        response: str, 
        change_var: str, 
        affected_var: str,
        change_pos: int
    ) -> bool:
        """
        Verify if cascade effect is properly identified.
        
        Args:
            response: Full response text
            change_var: Variable that changed
            affected_var: Variable that should be affected
            change_pos: Position where change is mentioned
            
        Returns:
            True if cascade is properly identified
        """
        # Look in text after the change mention
        text_after = response[change_pos:]
        
        # Normalize variable names for comparison
        affected_norm = self._normalize_var(affected_var)
        
        # Patterns indicating the affected variable needs update
        patterns = [
            f'{affected_var}.*(?:recalculated|updated|changes|affected|revised)',
            f'(?:recalculate|update|change|revise|adjust).*{affected_var}',
            f'{affected_var}.*(?:becomes|is now|new value|will be)',
            f'(?:must|need to|should).*(?:recalculate|update).*{affected_var}',
            f'{affected_var}.*(?:needs|requires).*(?:recalculation|update)',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_after[:500], re.IGNORECASE):  # Look in next 500 chars
                return True
        
        # Also check if variable is mentioned in a clear recalculation context
        recalc_patterns = [
            r'recalculat\w+:?\s*([^\.]+)',
            r'(?:must|need to) (?:update|recalculate):?\s*([^\.]+)',
            r'affected (?:variables|values):?\s*([^\.]+)',
        ]
        
        for pattern in recalc_patterns:
            match = re.search(pattern, text_after[:500], re.IGNORECASE)
            if match and affected_norm in self._normalize_var(match.group(1)):
                return True
        
        return False