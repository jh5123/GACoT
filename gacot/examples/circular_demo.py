"""
gacot/examples/circular_demo.py

Demonstration of GACoT's circular reference solving capabilities.
This showcases how the framework handles complex financial models with circular dependencies,
a common challenge in LBO modeling and iterative financial calculations.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from gacot import DependencyTracker, CircularSolver, LLMClient
from gacot.circular_solver import SolverConfig


class CircularReferenceDemo:
    """
    Demonstrates GACoT's ability to handle circular references in financial models.
    
    This is particularly relevant for:
    - LBO (Leveraged Buyout) models with debt sizing
    - Working capital calculations
    - Tax shield computations
    - Iterative DCF models
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the demo with necessary components.
        
        Args:
            verbose: Enable detailed output for demonstration
        """
        self.verbose = verbose
        self.tracker = DependencyTracker()
        self.llm = LLMClient(provider="mock")  # Use mock for demo
        
        # Configure solver for demonstration
        config = SolverConfig(
            max_iterations=10,
            tolerance=0.01,  # 1% convergence threshold
            verbose=verbose
        )
        self.solver = CircularSolver(self.tracker, self.llm, config)
    
    def demo_lbo_debt_sizing(self) -> Dict[str, float]:
        """
        Demonstrate LBO debt sizing with circular reference.
        
        Common scenario in private equity where:
        - Debt amount depends on total sources
        - Debt fees depend on debt amount  
        - Total sources include debt and fees (circular!)
        
        Returns:
            Dictionary of solved values
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(" LBO DEBT SIZING WITH CIRCULAR REFERENCE")
            print("=" * 60)
            print("\nScenario: Private equity buyout with debt financing")
            print("- Equity contribution: $400M")
            print("- Target debt: 60% of total sources")
            print("- Debt arrangement fees: 2% of debt amount")
            print("- Total sources = Equity + Debt + Fees (circular!)")
        
        # Set up the circular dependency structure
        self.tracker.add_dependencies({
            "Total_Sources": ["Equity", "Debt", "Fees"],
            "Debt": ["Total_Sources"],  # Debt = 60% of Total_Sources
            "Fees": ["Debt"],            # Fees = 2% of Debt
        })
        
        # Create the problem prompt
        prompt = """
        Calculate the LBO financing structure where:
        - Equity investment = $400 million
        - Debt = 60% of Total_Sources
        - Arrangement Fees = 2% of Debt amount
        - Total_Sources = Equity + Debt + Fees
        
        This creates a circular reference that needs iterative solving.
        """
        
        # Add dependency context to prompt
        enhanced_prompt = self.tracker.inject_context(prompt)
        
        if self.verbose:
            print("\n Dependency Structure:")
            cycles = self.tracker.detect_cycles()
            for cycle in cycles:
                print(f"   Circular: {' → '.join(cycle)}")
        
        # Solve using the circular solver
        converged, values, iterations = self.solver.solve(enhanced_prompt)
        
        if self.verbose:
            self._print_results(converged, values, iterations)
        
        return values
    
    def demo_tax_shield_iteration(self) -> Dict[str, float]:
        """
        Demonstrate iterative tax shield calculation.
        
        Scenario where:
        - Interest expense creates tax shield
        - Tax shield affects cash flow
        - Cash flow affects debt capacity
        - Debt capacity affects interest expense (circular!)
        
        Returns:
            Dictionary of solved values
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(" TAX SHIELD ITERATION DEMO")
            print("=" * 60)
            print("\nScenario: Debt capacity with tax shield benefits")
        
        # Set up dependencies
        self.tracker.add_dependencies({
            "Interest_Expense": ["Debt_Amount"],
            "Tax_Shield": ["Interest_Expense", "Tax_Rate"],
            "After_Tax_Cash_Flow": ["EBITDA", "Tax_Shield", "Interest_Expense"],
            "Debt_Capacity": ["After_Tax_Cash_Flow"],
            "Debt_Amount": ["Debt_Capacity"],  # Circular!
        })
        
        prompt = """
        Calculate debt capacity with tax shield:
        - EBITDA = $100 million
        - Tax Rate = 30%
        - Interest Rate = 5%
        - Debt Capacity = 5x After-Tax Cash Flow
        - Interest Expense = Interest Rate × Debt Amount
        - Tax Shield = Interest Expense × Tax Rate
        """
        
        enhanced_prompt = self.tracker.inject_context(prompt)
        converged, values, iterations = self.solver.solve(enhanced_prompt)
        
        if self.verbose:
            self._print_results(converged, values, iterations)
        
        return values
    
    def demo_working_capital_cycle(self) -> Dict[str, float]:
        """
        Demonstrate working capital circular calculation.
        
        Working capital often creates circularity in financial models:
        - Sales drive receivables
        - Receivables affect cash
        - Cash affects purchasing power
        - Purchasing affects inventory/COGS
        - COGS affects margin and sales capacity
        
        Returns:
            Dictionary of solved values
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print(" WORKING CAPITAL CYCLE DEMO")
            print("=" * 60)
            print("\nScenario: Self-reinforcing working capital dynamics")
        
        self.tracker.add_dependencies({
            "Receivables": ["Sales"],
            "Cash_Available": ["Beginning_Cash", "Receivables", "Payables"],
            "Purchasing_Power": ["Cash_Available"],
            "Inventory": ["Purchasing_Power"],
            "COGS": ["Inventory"],
            "Sales": ["COGS"],  # Circular through margin!
            "Payables": ["COGS"],
        })
        
        prompt = """
        Solve working capital cycle:
        - Beginning Cash = $10 million
        - Receivables = 15% of Sales (45 days)
        - Payables = 10% of COGS (30 days)
        - Inventory turnover = 8x
        - Gross Margin = 40%
        - Sales capacity depends on inventory available
        """
        
        enhanced_prompt = self.tracker.inject_context(prompt)
        converged, values, iterations = self.solver.solve(enhanced_prompt)
        
        if self.verbose:
            self._print_results(converged, values, iterations)
        
        return values
    
    def _print_results(
        self,
        converged: bool,
        values: Dict[str, float],
        iterations: int
    ) -> None:
        """
        Print formatted results of circular solving.
        
        Args:
            converged: Whether solution converged
            values: Final calculated values
            iterations: Number of iterations needed
        """
        print("\n" + "-" * 40)
        print(" SOLUTION")
        print("-" * 40)
        
        if converged:
            print(f" Converged successfully after {iterations} iterations")
        else:
            print(f"  Did not converge within {iterations} iterations")
            print("   (Showing best estimate)")
        
        print("\nFinal Values:")
        for var, val in sorted(values.items()):
            # Format based on magnitude
            if val >= 1e9:
                print(f"  {var:20s} = ${val/1e9:,.2f}B")
            elif val >= 1e6:
                print(f"  {var:20s} = ${val/1e6:,.2f}M")
            elif val >= 1000:
                print(f"  {var:20s} = ${val:,.0f}")
            elif val >= 1:
                print(f"  {var:20s} = ${val:,.2f}")
            else:
                print(f"  {var:20s} = {val:.2%}")
    
    def run_all_demos(self) -> None:
        """Execute all demonstration scenarios."""
        demos = [
            ("LBO Debt Sizing", self.demo_lbo_debt_sizing),
            ("Tax Shield Iteration", self.demo_tax_shield_iteration),
            ("Working Capital Cycle", self.demo_working_capital_cycle),
        ]
        
        print("\n" + "=" * 60)
        print(" GACoT CIRCULAR REFERENCE SOLVING DEMONSTRATIONS")
        print("=" * 60)
        print("\nDemonstrating advanced financial modeling capabilities")
        print("for handling circular dependencies in:")
        print("• Leveraged Buyout (LBO) models")
        print("• Tax shield optimizations")
        print("• Working capital dynamics")
        
        for name, demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"\n Error in {name}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(" DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThese examples showcase GACoT's ability to handle")
        print("complex circular references that are common in")
        print("sophisticated financial models.")


def main():
    """Run the circular reference demonstrations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GACoT Circular Reference Solving Demonstrations"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--demo",
        choices=["lbo", "tax", "working_capital", "all"],
        default="all",
        help="Choose specific demo to run"
    )
    
    args = parser.parse_args()
    
    demo = CircularReferenceDemo(verbose=not args.quiet)
    
    if args.demo == "all":
        demo.run_all_demos()
    elif args.demo == "lbo":
        demo.demo_lbo_debt_sizing()
    elif args.demo == "tax":
        demo.demo_tax_shield_iteration()
    elif args.demo == "working_capital":
        demo.demo_working_capital_cycle()


if __name__ == "__main__":
    main()

 