"""
Dependency Tracking System
Core logic for managing calculation dependencies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict


@dataclass
class Variable:
    """Represents a financial variable with dependencies."""
    name: str
    value: Optional[float] = None
    formula: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    is_dirty: bool = False
    
    def mark_dirty(self) -> None:
        """Mark variable as needing recalculation."""
        self.is_dirty = True
    
    def mark_clean(self) -> None:
        """Mark variable as up-to-date."""
        self.is_dirty = False


class DependencyTracker:
    """Track and manage calculation dependencies."""
    
    def __init__(self):
        """Initialize empty dependency tracker."""
        self.variables: Dict[str, Variable] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_variable(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        value: Optional[float] = None,
        formula: Optional[str] = None
    ) -> Variable:
        """
        Add or update a variable.
        
        Args:
            name: Variable name
            dependencies: List of variables this depends on
            value: Current value
            formula: Calculation formula
            
        Returns:
            The Variable object
        """
        if name in self.variables:
            var = self.variables[name]
            if value is not None:
                var.value = value
            if formula is not None:
                var.formula = formula
            if dependencies is not None:
                var.dependencies = dependencies
        else:
            var = Variable(
                name=name,
                dependencies=dependencies or [],
                value=value,
                formula=formula
            )
            self.variables[name] = var
        
        # Update dependency graphs
        if dependencies:
            self._update_graphs(name, dependencies)
        
        return var
    
    def add_dependencies(self, deps: Dict[str, List[str]]) -> None:
        """
        Add multiple dependencies at once.
        
        Args:
            deps: Dictionary mapping variables to their dependencies
        """
        for target, sources in deps.items():
            self.add_variable(target, dependencies=sources)
    
    def _update_graphs(self, target: str, sources: List[str]) -> None:
        """Update forward and reverse dependency graphs."""
        # Clear old dependencies
        old_sources = self.dependency_graph.get(target, set())
        for source in old_sources:
            self.reverse_graph[source].discard(target)
        
        # Add new dependencies
        self.dependency_graph[target] = set(sources)
        for source in sources:
            self.reverse_graph[source].add(target)
    
    def mark_dirty(self, changed_var: str) -> Set[str]:
        """
        Mark all dependent variables as needing recalculation.
        
        Args:
            changed_var: Variable that changed
            
        Returns:
            Set of all affected variables
        """
        dirty = set()
        queue = [changed_var]
        
        while queue:
            current = queue.pop(0)
            if current not in dirty:
                dirty.add(current)
                
                # Mark variable as dirty
                if current in self.variables:
                    self.variables[current].mark_dirty()
                
                # Add all dependents to queue
                for dependent in self.reverse_graph.get(current, []):
                    if dependent not in dirty:
                        queue.append(dependent)
        
        return dirty
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies using DFS.
        
        Returns:
            List of cycles (each cycle is a list of variables)
        """
        cycles = []
        visited = set()
        rec_stack = []
        
        def dfs(node: str) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = rec_stack.index(node)
                cycles.append(rec_stack[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                dfs(neighbor)
            
            rec_stack.pop()
        
        # Check all nodes
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def get_calculation_order(self) -> List[str]:
        """
        Get topological order for calculations.
        
        Returns:
            List of variables in calculation order.
            Empty list if cycles exist.
        """
        if self.detect_cycles():
            return []
        
        # Kahn's algorithm for topological sort
        in_degree = defaultdict(int)
        
        for node in self.dependency_graph:
            for dep in self.dependency_graph[node]:
                in_degree[dep] += 1
        
        # Start with nodes that have no dependencies
        queue = [
            node for node in self.dependency_graph
            if in_degree[node] == 0
        ]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            # Remove edge from graph
            for neighbor in self.reverse_graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order
    
    def get_minimal_recalc_set(self, changed_vars: List[str]) -> Set[str]:
        """
        Get minimal set of variables to recalculate.
        
        Args:
            changed_vars: List of changed variables
            
        Returns:
            Minimal set of variables needing recalculation
        """
        recalc_set = set()
        
        for var in changed_vars:
            affected = self.mark_dirty(var)
            recalc_set.update(affected)
        
        # Remove the changed variables themselves (they're inputs)
        recalc_set.difference_update(changed_vars)
        
        return recalc_set
    
    def inject_context(self, prompt: str) -> str:
        """
        Add dependency context to a prompt.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Prompt with dependency context
        """
        dirty_vars = [v.name for v in self.variables.values() if v.is_dirty]
        
        if not dirty_vars and not self.dependency_graph:
            return prompt
        
        context_parts = ["[DEPENDENCY CONTEXT]"]
        
        # Add dirty variables
        if dirty_vars:
            context_parts.append(
                f"Variables needing recalculation: {', '.join(dirty_vars)}"
            )
        
        # Add dependency chains
        for var in self.variables.values():
            if var.dependencies:
                context_parts.append(
                    f"  {var.name} depends on: {', '.join(var.dependencies)}"
                )
        
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            context_parts.append("\n  CIRCULAR REFERENCES DETECTED:")
            for cycle in cycles:
                context_parts.append(f"  {' → '.join(cycle)}")
            context_parts.append(
                "Use iterative solving: start with initial guess, "
                "iterate until convergence"
            )
        else:
            # Add calculation order
            order = self.get_calculation_order()
            if order:
                context_parts.append(
                    f"\nOptimal calculation order: {' → '.join(order)}"
                )
        
        context_parts.append("[END CONTEXT]\n")
        
        return "\n".join(context_parts) + "\n" + prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export tracker state as dictionary.
        
        Returns:
            Dictionary representation of tracker state
        """
        return {
            "variables": {
                name: {
                    "value": var.value,
                    "formula": var.formula,
                    "dependencies": var.dependencies,
                    "is_dirty": var.is_dirty
                }
                for name, var in self.variables.items()
            },
            "dependency_graph": {
                k: list(v) for k, v in self.dependency_graph.items()
            },
            "cycles": self.detect_cycles()
        }