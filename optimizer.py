"""
Optimizer - Operations Research Script for Reconciliation Parameters.

This script runs multiple simulations on a single input file,
varying key parameters to find the combination that maximizes
reconciliation percentages.
"""

import argparse
import time
from pathlib import Path
import json
import sys
from tqdm import tqdm
import optuna

def load_optimizer_config(config_path='config_optimizer.json'):
    """Loads the optimizer configuration from a JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            print(f"📄 Loading optimizer configuration from '{config_path}'...")
            config = json.load(f)
            print("✓ Optimizer configuration loaded successfully.")
            # Extract the two main sections, providing default values
            settings = config.get('optimizer_settings', {})
            params = config.get('optimization_params', {})
            return settings, params
    except FileNotFoundError:
        print(f"⚠️  File '{config_path}' not found. Using default configuration.")
        # Return a tuple of empty dictionaries
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format in '{config_path}': {e}")
        print("The program will now exit.")
        sys.exit(1)

def generate_dynamic_ranges(base_config, optimizer_config, range_percentage=0.30):
    """
    Generates optimization ranges dynamically.

    For each numeric parameter, it creates a symmetrical interval around the value
    in `base_config`, expanded by a certain percentage.
    Categorical parameters are left unchanged.

    Args:
        base_config (dict): The starting configuration (from config.json).
        optimizer_config (dict): The optimizer configuration with types and steps.
        range_percentage (float): The percentage to define the range width (e.g., 0.3 for ±30%).

    Returns:
        dict: A new configuration for the optimizer with dynamic ranges.
    """
    dynamic_ranges = optimizer_config.copy()
    print(f"🧬  Generating dynamic optimization ranges (width: ±{range_percentage*100:.0f}%)...")

    for param_name, details in optimizer_config.items():
        if details['type'] == 'numeric' and param_name in base_config:
            base_value = base_config[param_name]
            delta = base_value * range_percentage

            new_min = base_value - delta
            new_max = base_value + delta

            # Ensure integer values remain integers and do not fall below 1
            if details.get('value_type') != 'float':
                new_min = max(1, round(new_min))
                new_max = max(1, round(new_max))
            else: # For floats, round and ensure it's not negative
                new_min = max(0.0, round(new_min, 2))
                new_max = max(0.0, round(new_max, 2))

            dynamic_ranges[param_name]['min'] = new_min
            dynamic_ranges[param_name]['max'] = new_max
            print(f"   - Parameter '{param_name}': range set to [{new_min}, {new_max}] (step: {details['step']})")

    return dynamic_ranges

def run_auto_optimization(config, config_path, file_input, is_first_run, sequential=False):
    """Runs automatic optimization based on predefined ranges."""
    print("🔬 Starting optimization in automatic mode...")

    # Load both optimizer settings and parameters
    optimizer_settings, optimization_params = load_optimizer_config()

    if is_first_run:
        print(">>> FIRST RUN DETECTED: Starting WIDE EXPLORATION mode.")
        # Use the min/max ranges defined in config_optimizer.json
        ranges_to_use = optimization_params
        n_trials = optimizer_settings.get('n_trials_first_run', 70) # More trials for the first exploration
    else:
        print(">>> SUBSEQUENT RUN: Starting FOCUSED REFINEMENT mode.")
        # Generate narrow dynamic ranges around the already optimized values
        range_percentage = optimizer_settings.get('range_percentage', 0.15) # Tighter range for refinement
        ranges_to_use = generate_dynamic_ranges(config, optimization_params, range_percentage)
        n_trials = optimizer_settings.get('n_trials_refinement', 40) # Fewer trials for refinement

    best_params = run_simulation(config, ranges_to_use, file_input, n_trials, show_progress=False, sequential=sequential)

    # Write the best parameters to the configuration file
    update_config_file(config_path, best_params)

def run_simulation(base_config, optimizer_config_ranges, file_input, n_trials, show_progress=True, sequential=False):
    """Runs the optimization using Optuna to find the best parameters."""

    def objective(trial, input_data):
        """Objective function that Optuna will try to maximize."""
        # 1. Suggest parameters for this "trial"
        params = {}
        for param_name, details in optimizer_config_ranges.items():
            if details['type'] == 'numeric':
                if details.get('value_type') == 'float':
                    suggested_value = trial.suggest_float(param_name, details['min'], details['max'], step=details['step'])
                    params[param_name] = round(suggested_value, 2)
                else:
                    params[param_name] = trial.suggest_int(param_name, details['min'], details['max'], step=details['step'])
            elif details['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, details['values'])

        # 2. Run the simulation with the suggested parameters
        run_config = base_config.copy()
        run_config.update(params)
        
        # Import here to be compatible with Optuna's parallelization
        from core import ReconciliationEngine

        # Filter the configuration dictionary to pass only the parameters
        # expected by the ReconciliationEngine constructor.
        expected_params = [
            'tolerance', 'days_window', 'max_combinations',
            'residual_threshold', 'residual_days_window',
            'sorting_strategy', 'search_direction',
            'algorithm'
        ]
        engine_config = {
            key: run_config[key] for key in expected_params if key in run_config
        }

        engine_sim = ReconciliationEngine(**engine_config)
        stats = engine_sim.run(input_data, output_file=None, verbose=False)

        # 3. Calculate and return the score to be maximized
        if stats:
            debit_perc = stats.get('_raw_debit_amount_perc', 0.0)
            credit_perc = stats.get('_raw_credit_amount_perc', 0.0)
            
            # We want to maximize the sum of the percentages
            return debit_perc + credit_perc
        
        # If the simulation fails, return a very low score
        return 0.0

    # Disable Optuna's verbose logging to keep the output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create an optimization "study" to maximize the objective function.
    study = optuna.create_study(direction="maximize")

    # Optimization: Load data only once before starting trials to avoid repeated I/O.
    from core import ReconciliationEngine
    input_file_path = file_input
    loader = ReconciliationEngine()
    input_df = loader.load_file(input_file_path) # Assuming a refactored load_file method

    print(f"🚀 Starting optimization with Optuna for {n_trials} trials (in parallel)...")

    if sequential:
        n_jobs = 1
        print("🐌 Running trials in sequential mode (n_jobs=1).")
    else:
        n_jobs = -1 # Use all available CPU cores
        
    if show_progress:
        # Add a progress bar with tqdm only if requested
        with tqdm(total=n_trials, desc="Optimization Trials") as pbar:
            # Define a callback to update the progress bar after each trial
            def callback(study, trial):
                pbar.update(1)

            study.optimize(
                lambda trial: objective(trial, input_data=input_df),
                n_trials=n_trials, 
                n_jobs=n_jobs, 
                callbacks=[callback]
            )
    else:
        # Run without a progress bar (for batch mode)
        study.optimize(
            lambda trial: objective(trial, input_data=input_df),
            n_trials=n_trials, 
            n_jobs=n_jobs
        )

    # Print the final suggestion
    best_params = study.best_params
    best_score = study.best_value

    print("\n\n╔════════════════════════════════════════════════════════════╗")
    print("║              🏆 OPTIMAL PARAMETERS FOUND 🏆             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\nThe parameter combination that produced the best results is:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
    print(f"\nWith these settings, you have achieved:")
    print(f"  - Optimized Score (Sum of Debit % + Credit %): {best_score:.2f}")
    print("\nSuggestion: Update your 'config.json' file with these values for future processing.")
    
    return best_params

def get_user_parameters():
    """Interactive function to define the simulation parameters and ranges."""
    print("\n--- Optimization Parameter Configuration ---")
    available_params = {
        "1": {"name": "days_window", "type": "numeric", "prompt": "Standard time window (days)"},
        "2": {"name": "max_combinazioni", "type": "numeric", "prompt": "Maximum number of combinations"},
        "3": {"name": "residual_days_window", "type": "numeric", "prompt": "Time window for residuals (days)"},
        "4": {"name": "residual_threshold", "type": "numeric", "prompt": "Amount threshold for residual analysis (€)"},
        "5": {"name": "tolerance", "type": "numeric", "prompt": "Amount tolerance (e.g., 0.01)"},
        "6": {"name": "sorting_strategy", "type": "categorical", "values": ["date", "amount"], "prompt": "Sorting strategy"},
        "7": {"name": "search_direction", "type": "categorical", "values": ["future_only", "past_only", "both"], "prompt": "Time search direction"}
    }

    params_to_test = {}
    while True:
        print("\nChoose a parameter to vary (or press Enter to start the simulation):")
        for key, value in available_params.items():
            print(f"  {key}) {value['prompt']}")

        choice = input("> ")
        if not choice:
            if not params_to_test:
                print("❌ No parameter selected. Exiting.")
                exit()
            break

        if choice not in available_params:
            print("❌ Invalid choice.")
            continue

        param_info = available_params.pop(choice) # Remove to avoid choosing it again
        param_name = param_info['name']

        if param_info['type'] == 'numeric':
            try:
                value_type = float if param_name in ['tolerance', 'residual_threshold'] else int
                min_val = value_type(input(f"  - MIN value for '{param_name}': "))
                max_val = value_type(input(f"  - MAX value for '{param_name}': "))
                step = value_type(input(f"  - Step for '{param_name}': "))
                
                # Generate the sequence of values
                values = []
                current = min_val
                while current <= max_val:
                    values.append(current)
                    current += step
                params_to_test[param_name] = values
                print(f"✓ Parameter '{param_name}' configured to test values: {values}")
            except (ValueError, TypeError):
                print("❌ Invalid input. Please try again.")
                available_params[choice] = param_info # Re-insert to allow re-selection
        
        elif param_info['type'] == 'categorical':
            possible_values = param_info['values']
            print(f"  Possible values for '{param_name}':")
            for i, val in enumerate(possible_values):
                print(f"    {i+1}) {val}")
            
            user_choice = input("  Choose values to test (e.g., '1,3' for the first and third, or 'all'): ")
            
            selected_values = []
            if user_choice.lower() == 'all':
                selected_values = possible_values
            else:
                try:
                    indices = [int(i.strip()) - 1 for i in user_choice.split(',')]
                    for idx in indices:
                        if 0 <= idx < len(possible_values):
                            selected_values.append(possible_values[idx])
                        else:
                            print(f"❌ Invalid index '{idx+1}'.")
                    if not selected_values:
                        raise ValueError("No valid value selected.")
                except (ValueError, IndexError):
                    print("❌ Invalid selection. Please try again.")
                    available_params[choice] = param_info # Re-insert
                    continue
            
            params_to_test[param_name] = list(set(selected_values)) # Remove duplicates
            print(f"✓ Parameter '{param_name}' configured to test values: {params_to_test[param_name]}")

    return params_to_test

def update_config_file(config_path, best_params):
    """Updates the JSON configuration file with the best parameters found."""
    with open(config_path, 'r+') as f:
        config_data = json.load(f)
        config_data.update(best_params)
        f.seek(0) # Rewind to the beginning of the file
        json.dump(config_data, f, indent=2)
        f.truncate() # Remove trailing content if the new file is shorter
    print(f"\n✅ Configuration file '{config_path}' updated with optimal parameters.")

def main():
    """Main function that orchestrates the simulation."""
    parser = argparse.ArgumentParser(description="Reconciliation Parameter Optimizer.")
    parser.add_argument('--config', required=True, help="Path to the JSON configuration file to use and update.")
    parser.add_argument('--first-run', action='store_true', help="Indicates that this is the first run for this file, enabling a wider search.")
    parser.add_argument('--auto', action='store_true', help="Run in non-interactive automatic mode.")
    parser.add_argument('--sequential', action='store_true', help="Force Optuna trials to run sequentially (one process at a time).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    # Load the base configuration from the specified file
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║           🚀 STARTING PARAMETER OPTIMIZER 🚀             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n🎯 Configuration file in use: {config_path.resolve()}")
    file_input_for_analysis = config.get('file_input_for_optimization')
    print(f"📄 Input file for analysis: {file_input_for_analysis}")

    if args.auto:
        run_auto_optimization(config, config_path, file_input_for_analysis, args.first_run, sequential=args.sequential)
    else:
        # Interactive mode
        print("\n⚙️  Base configuration loaded from config.json:")
        for key, value in config.items():
            if not isinstance(value, dict): # Print top-level values
                print(f"   - {key}: {value}")
        
        params_to_test = get_user_parameters()
        # For interactive mode, we ask for the number of trials
        n_trials_interactive = int(input("\nHow many trials to run? (e.g., 50): ") or 50)
        
        best_params = run_simulation(config, params_to_test, file_input_for_analysis, n_trials_interactive, show_progress=True, sequential=args.sequential)
        update_config_file(config_path, best_params)

if __name__ == "__main__":
    main()