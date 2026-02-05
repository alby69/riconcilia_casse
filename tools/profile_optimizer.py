import cProfile
import pstats
import io
import json
import os
import sys

# Add the parent directory to the path to import core modules if the script is in tools/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer import run_simulation, load_optimizer_config, generate_dynamic_ranges

def run_profiling():
    """
    Wrapper function that executes the optimization logic to be profiled.
    This is useful because optimization is the most performance-critical part of the process.
    """
    # --- CONFIGURE HERE ---
    # 1. Specify the base configuration file and the input file for profiling
    base_config_path = 'config.json'
    input_file_path = 'input/sample_data.feather' # Use a pre-converted Feather file for speed
    
    # 2. Load configurations
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    optimizer_settings, optimization_params = load_optimizer_config()

    # 3. Choose the optimization mode to profile
    #    'True' for a first run (wide exploration), 'False' for refinement.
    is_first_run = False

    if is_first_run:
        print(">>> Profiling in WIDE EXPLORATION mode.")
        ranges_to_use = optimization_params
        n_trials = optimizer_settings.get('n_trials_first_run', 20) # Reduce trials for faster profiling
    else:
        print(">>> Profiling in FOCUSED REFINEMENT mode.")
        range_percentage = optimizer_settings.get('range_percentage', 0.30)
        ranges_to_use = generate_dynamic_ranges(base_config, optimization_params, range_percentage)
        n_trials = optimizer_settings.get('n_trials_refinement', 10) # Reduce trials

    # Run the simulation (the function you want to measure)
    print(f"Starting profiling of 'run_simulation' for file: {input_file_path}...")
    # Force sequential mode for a cleaner and more readable profiling output
    run_simulation(base_config, ranges_to_use, input_file_path, n_trials, show_progress=False, sequential=True)
    print("Profiling complete.")

if __name__ == "__main__":
    # 1. Create a Profiler object
    profiler = cProfile.Profile()

    # 2. Run your function under the profiler's control
    profiler.run('run_profiling()')

    # 3. Print the stats, sorted by the cumulative time spent in each function
    print("\n--- Profiling Results (sorted by total time 'tottime') ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20) # Show the 20 slowest functions