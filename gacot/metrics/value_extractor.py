"""
Universal Value Correctness Evaluator
Handles all types of financial calculations with smart number extraction and matching
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math


@dataclass
class ExtractedValue:
    """Container for an extracted value with context."""
    value: float
    raw_text: str
    context: str
    is_percentage: bool = False
    is_currency: bool = False
    is_shares: bool = False
    confidence: float = 1.0


class UniversalValueExtractor:
    """Extract and match numerical values from any financial calculation."""
    
    def __init__(self):
        """Initialize the extractor."""
        # Keywords that indicate final answers (increase confidence)
        self.answer_indicators = [
            'answer', 'result', 'total', 'final', 'approximately', 'about',
            'conclusion', 'therefore', 'thus', 'so', '='
        ]
        
        # Keywords for different value types
        self.percentage_indicators = [
            'irr', 'return', 'rate', 'yield', 'growth', 'percentage', '%',
            'margin', 'roi', 'percent', 'volatility', 'correlation'
        ]
        
        self.share_indicators = [
            'share', 'stock', 'equity unit', 'securities', 'issued'
        ]
        
        self.currency_indicators = [
            '$', 'dollar', 'million', 'billion', 'equity', 'debt', 
            'value', 'ebitda', 'fcf', 'npv', 'price', 'cost', 'premium'
        ]
    
    def extract_all_values(self, text: str) -> List[ExtractedValue]:
        """
        Extract all meaningful numerical values from text.
        
        Returns:
            List of ExtractedValue objects sorted by confidence
        """
        values = []
        
        # Clean text for processing
        clean_text = text.replace(',', '')
        
        # Pattern to match various number formats
        # This captures numbers with optional decimal points, currency symbols, percentages, etc.
        patterns = [
            # Currency with multipliers (e.g., $1.234M, 123.45 million)
            (r'\$?\s*([\d.]+)\s*(?:M|million|B|billion)', 'currency_scaled'),
            # Percentages (e.g., 12.34%, 0.1234 as percentage)
            (r'([\d.]+)\s*%', 'percentage'),
            # Share counts (e.g., 12.56 shares, 41.4 shares)
            (r'([\d.]+)\s+shares?', 'shares'),
            # Currency amounts (e.g., $1,234.56, 1234.56)
            (r'\$\s*([\d.]+)', 'currency'),
            # Scientific notation or very precise decimals
            (r'([\d]+\.[\d]{3,})', 'precise'),
            # General numbers (including those after = or :)
            (r'(?:[=:≈]\s*|^|\s)([\d]+(?:\.[\d]+)?)', 'general'),
        ]
        
        used_positions = set()  # Track positions to avoid duplicates
        
        for pattern, value_type in patterns:
            for match in re.finditer(pattern, clean_text, re.IGNORECASE):
                # Skip if we've already captured a number at this position
                if match.start() in used_positions:
                    continue
                    
                try:
                    value_str = match.group(1)
                    value = float(value_str)
                    
                    # Apply scaling for millions/billions
                    if value_type == 'currency_scaled':
                        if 'B' in match.group(0) or 'billion' in match.group(0).lower():
                            value *= 1e9
                        elif 'M' in match.group(0) or 'million' in match.group(0).lower():
                            value *= 1e6
                    
                    # Convert percentage to decimal if needed
                    is_percentage = value_type == 'percentage'
                    if is_percentage:
                        # Store both percentage and decimal forms
                        values.append(ExtractedValue(
                            value=value,  # As percentage (e.g., 12.34)
                            raw_text=match.group(0),
                            context=self._get_context(text, match.start(), match.end()),
                            is_percentage=True,
                            confidence=self._calculate_confidence(text, match)
                        ))
                        # Also store as decimal (e.g., 0.1234)
                        values.append(ExtractedValue(
                            value=value / 100,
                            raw_text=match.group(0),
                            context=self._get_context(text, match.start(), match.end()),
                            is_percentage=True,
                            confidence=self._calculate_confidence(text, match) * 0.9
                        ))
                    else:
                        # Determine value type
                        is_currency = value_type in ['currency', 'currency_scaled']
                        is_shares = value_type == 'shares'
                        
                        values.append(ExtractedValue(
                            value=value,
                            raw_text=match.group(0),
                            context=self._get_context(text, match.start(), match.end()),
                            is_currency=is_currency,
                            is_shares=is_shares,
                            confidence=self._calculate_confidence(text, match)
                        ))
                    
                    # Mark positions as used
                    for i in range(match.start(), match.end()):
                        used_positions.add(i)
                        
                except (ValueError, AttributeError):
                    continue
        
        # Special handling for boxed/emphasized answers (highest confidence)
        boxed_patterns = [
            r'\\boxed\{([\d.,]+)\}',
            r'\*\*([\d.,]+)\*\*',
        ]
        
        for pattern in boxed_patterns:
            for match in re.finditer(pattern, text):
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    # Determine type from context
                    context = self._get_context(text, match.start(), match.end())
                    is_percentage = any(ind in context.lower() for ind in self.percentage_indicators)
                    is_currency = any(ind in context.lower() for ind in self.currency_indicators)
                    is_shares = any(ind in context.lower() for ind in self.share_indicators)
                    
                    values.append(ExtractedValue(
                        value=value,
                        raw_text=match.group(0),
                        context=context,
                        is_percentage=is_percentage,
                        is_currency=is_currency,
                        is_shares=is_shares,
                        confidence=2.0  # Highest confidence for boxed answers
                    ))
                except (ValueError, AttributeError):
                    continue
        
        # Sort by confidence (highest first)
        values.sort(key=lambda x: x.confidence, reverse=True)
        
        return values
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _calculate_confidence(self, text: str, match) -> float:
        """
        Calculate confidence score for an extracted value.
        Higher scores for values near answer indicators.
        """
        confidence = 1.0
        
        # Get surrounding context
        context = self._get_context(text, match.start(), match.end(), 150)
        context_lower = context.lower()
        
        # Boost confidence if near answer indicators
        for indicator in self.answer_indicators:
            if indicator in context_lower:
                confidence += 0.5
                break
        
        # Boost if it's the last number in the text (often the final answer)
        if match.end() > len(text) * 0.8:
            confidence += 0.3
        
        # Boost if preceded by equals or colon
        if match.start() > 0:
            prefix = text[max(0, match.start()-5):match.start()]
            if any(c in prefix for c in ['=', ':', '≈']):
                confidence += 0.4
        
        return confidence
    
    def match_values(
        self,
        extracted: List[ExtractedValue],
        expected: Dict[str, float],
        tolerance_factor: float = 1.0
    ) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        """
        Match extracted values to expected values.
        
        Args:
            extracted: List of extracted values
            expected: Dictionary of expected values
            tolerance_factor: Multiplier for tolerance (1.0 = normal, 2.0 = lenient)
            
        Returns:
            Tuple of (score, matches) where matches maps expected_key to (expected_val, extracted_val)
        """
        if not expected or not extracted:
            return 0.0, {}
        
        matches = {}
        matched_indices = set()
        
        # Try to match each expected value
        for exp_key, exp_val in expected.items():
            best_match = None
            best_error = float('inf')
            best_index = -1
            
            for i, ext_val in enumerate(extracted):
                if i in matched_indices:
                    continue
                
                # Calculate error
                error = self._calculate_error(exp_val, ext_val.value)
                
                # Apply tolerance based on value type and magnitude
                tolerance = self._get_tolerance(exp_val, ext_val)
                adjusted_tolerance = tolerance * tolerance_factor
                
                if error < adjusted_tolerance and error < best_error:
                    best_error = error
                    best_match = ext_val.value
                    best_index = i
            
            if best_match is not None:
                matches[exp_key] = (exp_val, best_match)
                matched_indices.add(best_index)
        
        # Calculate score
        score = len(matches) / len(expected) if expected else 0.0
        
        return score, matches
    
    def _calculate_error(self, expected: float, extracted: float) -> float:
        """Calculate relative error between two values."""
        if expected == 0 and extracted == 0:
            return 0.0
        if expected == 0:
            return abs(extracted)
        return abs(extracted - expected) / abs(expected)
    
    def _get_tolerance(self, expected: float, extracted_val: ExtractedValue) -> float:
        """
        Get appropriate tolerance based on value type and magnitude.
        
        Different financial metrics have different acceptable error ranges.
        """
        abs_expected = abs(expected)
        
        # Percentage values (IRR, rates, etc.)
        if extracted_val.is_percentage or abs_expected < 1:
            if abs_expected < 0.01:  # Very small percentages
                return 0.1  # 10% relative error
            elif abs_expected < 0.1:  # Small percentages (< 10%)
                return 0.05  # 5% relative error
            else:
                return 0.02  # 2% relative error
        
        # Share counts
        elif extracted_val.is_shares:
            if abs_expected < 10:
                return 0.05  # 5% for small share counts
            else:
                return 0.01  # 1% for larger share counts
        
        # Currency/value amounts
        else:
            if abs_expected < 100:
                return 0.02  # 2% for small amounts
            elif abs_expected < 1000:
                return 0.015  # 1.5% for medium amounts
            elif abs_expected < 1e6:
                return 0.01  # 1% for large amounts
            else:
                return 0.005  # 0.5% for very large amounts


class ValueCorrectnessEvaluator:
    """
    Main evaluator class for value correctness.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.extractor = UniversalValueExtractor()
    
    def evaluate(
        self,
        response: str,
        problem: Dict,
        verbose: bool = False
    ) -> float:
        """
        Evaluate value correctness for any financial problem.
        
        Args:
            response: Model response
            problem: Problem dictionary with expected_values or solution
            verbose: Whether to print debug info
            
        Returns:
            Score between 0 and 1
        """
        # Get expected values
        expected_values = self._get_expected_values(problem)
        
        if not expected_values:
            return 0.5  # Uncertain if no expected values
        
        # Extract all values from response
        extracted = self.extractor.extract_all_values(response)
        
        if not extracted:
            if verbose:
                print("    VC: No values extracted from response")
            return 0.0
        
        # Match values
        score, matches = self.extractor.match_values(extracted, expected_values)
        
        # Debug output
        if verbose and score < 1.0:
            print(f"    VC: Matched {len(matches)}/{len(expected_values)} values")
            for key, (exp, ext) in matches.items():
                error = abs(ext - exp) / abs(exp) if exp != 0 else abs(ext)
                print(f"      {key}: expected {exp:.4f}, got {ext:.4f} (error: {error*100:.2f}%)")
            
            unmatched = set(expected_values.keys()) - set(matches.keys())
            if unmatched:
                print(f"      Unmatched: {', '.join(unmatched)}")
                if verbose and extracted[:3]:  # Show top extracted values for debugging
                    print(f"      Top extracted values: {[e.value for e in extracted[:3]]}")
        
        return score
    
    def _get_expected_values(self, problem: Dict) -> Dict[str, float]:
        """
        Extract expected values from problem definition.
        Handles both explicit expected_values and extraction from solution text.
        Enhanced for step-by-step solutions.
        """
        # First try explicit expected_values
        if "expected_values" in problem and problem["expected_values"]:
            return problem["expected_values"]
        
        # Try to extract from solution
        if "solution" in problem and problem["solution"]:
            extracted = self._extract_from_solution(problem["solution"])
            if extracted:
                return extracted
        
        # Try other common field names
        for field in ["answer", "gold", "gold_answer", "correct_answer"]:
            if field in problem:
                value = problem[field]
                if isinstance(value, (int, float)):
                    return {"result": float(value)}
                elif isinstance(value, dict):
                    return value
                elif isinstance(value, str):
                    # Try to extract from string
                    extracted = self._extract_from_solution(value)
                    if extracted:
                        return extracted
        
        return {} 
    
    def _extract_from_solution(self, solution: str) -> Dict[str, float]:
        """
        Extract expected values from step-by-step solution text.
        Optimized for "Step N:" format with calculations.
        """
        values = {}
        
        if not solution:
            return values
        
        # Strategy 1: Find all "= number" patterns (these are calculation results)
        import re
        
        # Pattern to match various number formats after equals sign
        equals_pattern = r'=\s*\$?([\d,]+(?:\.[\d]+)?)'
        
        all_values = []
        for match in re.finditer(equals_pattern, solution):
            try:
                # Clean and convert the number
                num_str = match.group(1).replace(',', '')
                value = float(num_str)
                all_values.append(value)
            except ValueError:
                continue
        
        # Strategy 2: Extract values with their variable names
        named_patterns = [
            # Variable = value (like "Compound Interest = 291.63")
            r'(\w+(?:\s+\w+)*)\s*=\s*\$?([\d,]+(?:\.[\d]+)?)',
            # Compute the X: ... = value
            r'Compute\s+the\s+(\w+(?:\s+\w+)*):[^=]+=\s*\$?([\d,]+(?:\.[\d]+)?)',
        ]
        
        named_values = {}
        for pattern in named_patterns:
            for match in re.finditer(pattern, solution, re.IGNORECASE):
                try:
                    var_name = match.group(1).strip().lower().replace(' ', '_')
                    value_str = match.group(2).replace(',', '')
                    value = float(value_str)
                    named_values[var_name] = value
                except (ValueError, IndexError):
                    continue
        
        # The final answer is typically the last calculated value
        if all_values:
            # The last value in the solution is usually the final answer
            values["result"] = all_values[-1]
            
            # Also include other significant values
            if len(all_values) > 1:
                # Include all unique values for comprehensive matching
                unique_values = []
                seen = set()
                for v in all_values:
                    # Round to avoid floating point duplicates
                    rounded = round(v, 2)
                    if rounded not in seen:
                        unique_values.append(v)
                        seen.add(rounded)
                
                # Add other values with generic keys
                for i, val in enumerate(unique_values[:-1]):  # Exclude the last one (already "result")
                    values[f"intermediate_{i+1}"] = val
        
        # Add named values (these override generic keys)
        values.update(named_values)
        
        # For this specific problem type, ensure we capture the key final values
        # Look specifically for "Compound Interest" which is the answer
        for key in ['compound_interest', 'compound_amount', 'final_value', 'total', 'answer']:
            if key in named_values:
                values["result"] = named_values[key]  # Override with the most specific answer
                break
        
        return values