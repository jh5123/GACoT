"""
metrics/dependency_graph.py
Dependency Graph Extraction (DGE) metric evaluator with tiered scoring
"""

import re
from typing import Dict, List, Set, Tuple
from .base import MetricEvaluator
from ..core.dependency_extractor import DependencyExtractor


class DependencyGraphEvaluator(MetricEvaluator):
    """Evaluates dependency graph extraction accuracy with tiered scoring."""
    
    def __init__(self):
        """Initialize with dependency extractor."""
        self.extractor = DependencyExtractor()
        self._setup_dependency_indicators()
    
    def name(self) -> str:
        return "Dependency Graph Extraction"
    
    def _setup_dependency_indicators(self):
        """Setup tiered dependency indicators for scoring."""
        self.dependency_indicators = {
            # Strong indicators (full points) - explicit dependency language
            'strong': [
                r'depends?\s+on\s*:?\s*\w+',           # "depends on X, Y"
                r'dependency|dependencies',             # explicit use of word
                r'recalculate.+when.+changes',         # cascade awareness
                r'if\s+\w+\s+changes?.+recalculate',   # conditional recalc
                r'affected\s+by\s+changes?\s+in',      # impact awareness
                r'must\s+be\s+updated\s+when',         # update requirement
                r'triggers?\s+recalculation',          # trigger language
            ],
            # Medium indicators (0.6 points) - indirect dependency language
            'medium': [
                r'calculated\s+from',                  # "X calculated from Y"
                r'based\s+on',                         # "X based on Y"
                r'uses?\s+values?\s+from',            # indirect dependency
                r'requires?',                          # "X requires Y"
                r'derived\s+from',                     # derivation
                r'function\s+of',                      # functional relationship
                r'determined\s+by',                    # determination
            ],
            # Weak indicators (0.3 points) - implied dependencies
            'weak': [
                r'=\s*[^=]+[+\-*/][^=]+',            # formula implies dependency
                r'step\s+\d+.+step\s+\d+',           # sequential steps
                r'using\s+the\s+\w+',                 # using variable
                r'with\s+\w+\s*=',                    # with assignment
            ]
        }
    
    def evaluate(
        self,
        response: str,
        problem: Dict,
        **kwargs
    ) -> float:
        """
        Calculate dependency extraction score using tiered system.
        
        Components:
        - Level 1 (40%): Explicit dependency statements
        - Level 2 (30%): Dependency notation/formalism
        - Level 3 (30%): Accuracy against gold standard
        
        Args:
            response: Model response
            problem: Problem with ground truth dependencies
            
        Returns:
            Score between 0 and 1
        """
        gold_graph = problem.get("dependencies", {}).get("graph", {})
        
        # Level 1: Check for explicit dependency statements (40% of score)
        explicit_score = self._score_explicit_dependencies(response)
        
        # Level 2: Check for dependency notation/formalism (30% of score)
        notation_score = self._score_dependency_notation(response)
        
        # Level 3: Check accuracy against gold standard (30% of score)
        accuracy_score = self._score_dependency_accuracy(response, gold_graph)
        
        final_score = (0.4 * explicit_score + 
                      0.3 * notation_score + 
                      0.3 * accuracy_score)
        
        # Debug output for low scores
        if final_score < 0.5 and kwargs.get('verbose', False):
            print(f"    DGE Debug: Explicit={explicit_score:.2f}, Notation={notation_score:.2f}, Accuracy={accuracy_score:.2f}")
        
        return final_score
    
    def _score_explicit_dependencies(self, response: str) -> float:
        """Score presence of explicit dependency language."""
        max_score = 0.0
        matches = {'strong': [], 'medium': [], 'weak': []}
        
        for level, patterns in self.dependency_indicators.items():
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    matches[level].append(pattern)
                    if level == 'strong':
                        max_score = max(max_score, 1.0)
                    elif level == 'medium':
                        max_score = max(max_score, 0.6)
                    elif level == 'weak':
                        max_score = max(max_score, 0.3)
        
        # Bonus for multiple different indicators
        if len(matches['strong']) >= 2:
            max_score = min(max_score * 1.1, 1.0)
        elif len(matches['medium']) >= 3:
            max_score = min(max_score * 1.2, 1.0)
        
        return max_score
    
    def _score_dependency_notation(self, response: str) -> float:
        """Score formal notation and structure."""
        score = 0.0
        
        # Check for functional notation: f(x,y) or NPV(rate, cashflows)
        if re.search(r'\w+\s*=\s*f?\([^)]+\)', response):
            score += 0.3
        
        # Check for arrow notation: X → Y → Z or X -> Y
        if re.search(r'\w+\s*[→➔⟶\->]+\s*\w+', response):
            score += 0.25
        
        # Check for dependency tables/lists
        if re.search(r'(depends?\s+on|dependencies).*?[\n:].*?[-•*●◦]', response, re.DOTALL):
            score += 0.25
        
        # Check for structured variable definitions
        if re.search(r'(variables?|definitions?|where):.*?\n.*?[=:]', response, re.IGNORECASE):
            score += 0.2
        
        # Check for explicit graph/tree language
        if re.search(r'(dependency\s+)?(graph|tree|chain|structure)', response, re.IGNORECASE):
            score += 0.15
        
        # Check for mathematical notation with subscripts/indices
        if re.search(r'\w+_\d+|\w+\[\d+\]', response):
            score += 0.1
        
        # Check for "cascade" or "propagation" language
        if re.search(r'cascade|propagat|ripple\s+effect', response, re.IGNORECASE):
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_dependency_accuracy(self, response: str, gold_graph: Dict) -> float:
        """Score accuracy against gold standard with fuzzy matching."""
        # Use the extractor to get dependencies
        extracted = self.extractor.extract(response)
        
        if not gold_graph:
            # No gold standard - give credit if no dependencies extracted
            return 1.0 if not extracted else 0.5
        
        # Build normalized edge sets
        gold_edges = self._graph_to_edges(gold_graph)
        pred_edges = self._graph_to_edges(extracted)
        
        if not gold_edges:
            return 1.0 if not pred_edges else 0.0
        
        # Calculate exact matches
        true_positives = len(pred_edges & gold_edges)
        
        # Give partial credit for close matches
        partial_matches = 0
        unmatched_pred = pred_edges - gold_edges
        unmatched_gold = gold_edges - pred_edges
        
        for pred_edge in unmatched_pred:
            for gold_edge in unmatched_gold:
                if self._edges_similar(pred_edge, gold_edge):
                    partial_matches += 0.5
                    unmatched_gold.discard(gold_edge)  # Don't match same gold edge twice
                    break
        
        # Calculate adjusted metrics
        adjusted_tp = true_positives + partial_matches
        
        # Calculate precision and recall
        precision = adjusted_tp / len(pred_edges) if pred_edges else 0
        recall = adjusted_tp / len(gold_edges) if gold_edges else 0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def _graph_to_edges(self, graph: Dict[str, List[str]]) -> Set[Tuple[str, str]]:
        """Convert dependency graph to normalized edge set."""
        edges = set()
        for target, sources in graph.items():
            target_norm = self._normalize_var(target)
            for source in sources:
                source_norm = self._normalize_var(source)
                if source_norm and target_norm:
                    edges.add((source_norm, target_norm))
        return edges
    
    def _edges_similar(self, edge1: Tuple[str, str], edge2: Tuple[str, str]) -> bool:
        """Check if two edges are similar (fuzzy matching)."""
        from difflib import SequenceMatcher
        
        source1, target1 = edge1
        source2, target2 = edge2
        
        # Check if targets are similar
        target_sim = SequenceMatcher(None, target1.lower(), target2.lower()).ratio()
        if target_sim < 0.7:
            return False
        
        # Check if sources are similar
        source_sim = SequenceMatcher(None, source1.lower(), source2.lower()).ratio()
        
        # Allow some flexibility - if one matches well, accept
        return source_sim > 0.7 or (target_sim > 0.9 and source_sim > 0.5)