"""
GACoT Evaluation Entry Point
"""

import argparse
import sys
from pathlib import Path
from typing import List

from gacot.core import (
    FinancialReasoningEval,
    aggregate_model_results,
    visualize_cached_results,
    create_visualizations
)


def setup_parser() -> argparse.ArgumentParser:
    """Configure command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GACoT: Graph-Augmented Chain of Thought Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with API
  %(prog)s --model gpt-4o --force-api
  
  # Use cached results
  %(prog)s --model gpt-4o-mini
  
  # Compare multiple models
  %(prog)s --model gpt-4o gpt-4o-mini claude-3-opus
  
  # Aggregate existing results
  %(prog)s --aggregate gpt-4o gpt-4o-mini
  
  # Visualize cached results
  %(prog)s --visualize gpt-4o
        """
    )
    
    parser.add_argument(
        "--model",
        nargs="+", 
        default=["gpt-4o-mini"],
        help="Model(s) to evaluate"
    )
    
    parser.add_argument(
        "--dataset",
        default="data/testset",
        help="Dataset path (default: data/testset)"
    )
    
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Force API calls even if cache exists"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    parser.add_argument(
        "--aggregate",
        nargs="+",
        help="Aggregate and compare results for multiple models"
    )
    
    parser.add_argument(
        "--visualize",
        help="Visualize existing cached results for a model"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of test problems"
    )
    
    return parser


def evaluate_single_model(
    model: str,
    dataset: str,
    force_api: bool,
    verbose: bool,
    limit: int = None
) -> tuple:
    """Evaluate a single model."""
    evaluator = FinancialReasoningEval(
        model=model,
        verbose=verbose,
        use_live_api=force_api
    )
    
    result = evaluator.run_2x2_evaluation(
        dataset_path=dataset,
        num_problems=limit
    )
    
    return model, result.overall_gap, result.capability_gaps, evaluator.results


def run_evaluation(args: argparse.Namespace) -> int:
    """Execute evaluation based on arguments."""
    print("\n" + "=" * 70)
    print(" GACoT: Financial Dependency Tracking Evaluation")
    print("=" * 70)
    
    models = args.model if isinstance(args.model, list) else [args.model]
    
    print(f"\nEvaluating {len(models)} model(s)")
    print(f"Dataset: {args.dataset}")
    if args.limit:
        print(f"Limited to {args.limit} problems")
    print("-" * 70)
    
    all_results = {}
    
    if len(models) > 1:
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for model in models:
                future = executor.submit(
                    evaluate_single_model,
                    model, args.dataset, args.force_api, args.verbose, args.limit
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                model, overall_gap, capability_gaps, results = future.result()
                all_results[model] = (overall_gap, capability_gaps, results)
                print(f"✓ {model}: Overall gap = {overall_gap:+.1%}")
    else:
        model = models[0]
        _, overall_gap, capability_gaps, results = evaluate_single_model(
            model, args.dataset, args.force_api, args.verbose, args.limit
        )
        all_results[model] = (overall_gap, capability_gaps, results)
    
    if not args.no_viz:
        print("\nCreating visualizations...")
        for model, (_, _, results) in all_results.items():
            create_visualizations(results, model)
    
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE")
    print("=" * 70)
    
    if len(all_results) > 1:
        print("\nModel Rankings (by overall capability gap):")
        for model, (gap, _, _) in sorted(all_results.items(), key=lambda x: x[1][0], reverse=True):
            print(f"  {model}: {gap:+.1%}")
    else:
        model = list(all_results.keys())[0]
        gap = all_results[model][0]
        caps = all_results[model][1]
        
        print(f"\nModel: {model}")
        print(f"Overall Capability Gap: {gap:+.1%}")
        print("\nDetailed Gaps:")
        for metric, value in caps.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:+.1%}")
        
        print("\n" + "=" * 70)
        print(" KEY FINDING")
        print("=" * 70)
        
        if gap > 0.3:
            print("  LLMs CANNOT intrinsically track financial dependencies.")
            print("  They require explicit scaffolding for calculation consistency.")
        elif gap > 0.15:
            print("  LLMs show LIMITED dependency tracking capability.")
            print("  Scaffolding provides significant improvements.")
        else:
            print("  LLMs demonstrate some intrinsic dependency awareness.")
            print("  Scaffolding provides marginal improvements.")
    
    print("\nVisualization files saved in cache/[model]/ directory")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        if args.visualize:
            print(f"Visualizing cached results for {args.visualize}...")
            visualize_cached_results(args.visualize)
            return 0
        
        if args.aggregate:
            print(f"Aggregating results for: {', '.join(args.aggregate)}")
            aggregate_model_results(args.aggregate)
            return 0
        
        return run_evaluation(args)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())