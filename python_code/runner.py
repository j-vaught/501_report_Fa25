# runner.py
"""
Main experiment orchestrator.
Runs all experiments defined in project_restructured.tex Chapter 5.
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.exp_solver_compare import run_all_shapes_all_solvers, print_results_table
from config import OUTPUT_DIR


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_results(results: dict, filename: str = "solver_comparison_results.json"):
    """Save results to JSON file."""
    ensure_output_dir()
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Convert any non-JSON-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return obj
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


def main():
    """Run the complete experiment suite."""
    print("="*60)
    print("PHOTOMETRIC STEREO EXPERIMENT SUITE")
    print("="*60)
    print("\nSection 5.8: Solver Comparison Experiments")
    print("-"*60)
    
    # Run solver comparison on all shapes
    print("\nRunning all 8 shapes with all 3 solvers...")
    results = run_all_shapes_all_solvers()
    
    # Print results table
    print_results_table(results)
    
    # Save to JSON
    save_results(results)
    
    print("\nExperiment suite complete!")
    

if __name__ == "__main__":
    main()
