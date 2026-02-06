"""
Executive Wrapper for Accounting Reconciliation.

This script's sole purpose is to:
1. Read a JSON configuration file.
2. Instantiate the RiconciliatoreContabile class with the correct parameters.
3. Launch the reconciliation process via the run() method.
4. Print the final statistics in JSON format if run in 'silent' mode.
"""
import argparse
import json
import sys
from core import ReconciliationEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the accounting reconciliation based on a configuration file.")
    parser.add_argument('--config', required=True, help="Path to the JSON configuration file.")
    parser.add_argument('--silent', action='store_true', help="Run without verbose output (used by the batch process).")
    
    args = parser.parse_args()
    
    # Load configuration from the specified file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ ERROR: Could not load or parse the configuration file '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    input_file = config.get('file_input')
    output_file = config.get('file_output')

    if not input_file or not output_file:
        print("❌ ERROR: 'file_input' and 'file_output' must be specified in the configuration file.", file=sys.stderr)
        sys.exit(1)

    # Instantiate the reconciler with parameters from the configuration
    engine = ReconciliationEngine(
        tolerance=config.get('tolerance', 0.01),
        days_window=config.get('days_window', 30),
        max_combinations=config.get('max_combinations', 6),
        residual_threshold=config.get('residual_threshold', 100),
        residual_days_window=config.get('residual_days_window', 60),
        sorting_strategy=config.get('sorting_strategy', 'date'),
        search_direction=config.get('search_direction', 'both'),
        column_mapping=config.get('column_mapping', None),
        algorithm=config.get('algorithm', 'subset_sum'),
        use_numba=config.get('use_numba', True),
        ignore_tolerance=config.get('ignore_tolerance', False)
    )

    # Run the entire process
    stats = engine.run(input_file, output_file, verbose=False) # Forza verbose=False

    if args.silent and stats:
        # --- ADDITION: Include the parameters used in the JSON report ---
        # Define the parameters of interest to display in the final summary.
        parameters_to_include = [
            'days_window', 
            'max_combinations', 
            'residual_days_window', 
            'residual_threshold', 
            'sorting_strategy', 
            'search_direction'
        ]
        # Add the parameters found in the configuration file to the statistics dictionary.
        stats['optimal_parameters'] = {key: config.get(key) for key in parameters_to_include}

        print(json.dumps(stats)) # Print statistics in JSON for the parent process
