"""
Optimizer - Operations Research Module for Reconciliation Parameters.

This module runs multiple simulations on a given dataset, varying key
parameters to find the combination that maximizes reconciliation percentages.
It is designed to be called from other parts of the application, such as a
web endpoint.
"""
import pandas as pd
import optuna
from tqdm import tqdm
from core import ReconciliationEngine
import copy
import os
import uuid

def find_best_parameters(input_df: pd.DataFrame, base_config: dict, optimizer_config: dict, show_progress=False, sequential=False):
    """
    Runs the core optimization process using Optuna to find the best parameters.

    This function sets up and executes an Optuna "study" to find the combination
    of reconciliation parameters that maximizes the final reconciled percentage.

    Args:
        input_df (pd.DataFrame): The pre-loaded and pre-processed DataFrame
            containing the transaction data for the simulation.
        base_config (dict): A dictionary with the base reconciliation parameters,
            typically from `config.json['reconciliation_defaults']`.
        optimizer_config (dict): A dictionary defining the optimizer's behavior,
            including the parameter search space. This typically comes from
            `config.json['optimizer_config']`.
        show_progress (bool): If True, displays a `tqdm` progress bar for the
            optimization trials. Defaults to False.
        sequential (bool): If True, forces the trials to run sequentially in a
            single process (`n_jobs=1`). Defaults to False.

    Returns:
        dict: A dictionary containing the best-performing parameter combination
              found by the study.
    """
    
    parameter_space = optimizer_config.get('parameter_space', {})
    optimizer_settings = optimizer_config.get('settings', {})
    n_trials = optimizer_settings.get('n_trials_first_run', 100)

    def objective(trial, data_df):
        """Objective function that Optuna will try to maximize."""
        params = {}
        # Suggest parameters for this trial based on the defined search space
        for param_name, details in parameter_space.items():
            if details['type'] == 'numeric':
                # Handle nested parameters like 'residuals.threshold'
                keys = param_name.split('.')
                # For flat params
                if len(keys) == 1:
                    step = details.get('step')
                    if details.get('value_type') == 'float':
                        params[param_name] = trial.suggest_float(param_name, details['min'], details['max'], step=step)
                    else:
                        step = step if step is not None else 1
                        params[param_name] = trial.suggest_int(param_name, details['min'], details['max'], step=step)

            elif details['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, details['values'])
        
        # Create a deep copy to avoid modifying the base config in-place
        run_config = copy.deepcopy(base_config)

        # Special handling for nested residual parameters
        nested_params = {k: v for k, v in params.items() if '.' in k}
        for key, value in nested_params.items():
            parent, child = key.split('.')
            if parent in run_config and isinstance(run_config[parent], dict):
                run_config[parent][child] = value
            # remove from flat params to avoid clashes
            del params[key]

        # Update the config with the flat parameters suggested by Optuna
        run_config.update(params)

        try:
            # The ReconciliationEngine expects some parameters for residuals inside a nested dict.
            # We construct the final config based on what the engine expects.
            engine_params = {
                'tolerance': run_config.get('tolerance'),
                'days_window': run_config.get('days_window'),
                'max_combinations': run_config.get('max_combinations'),
                'sorting_strategy': run_config.get('sorting_strategy'),
                'search_direction': run_config.get('search_direction'),
                'algorithm': run_config.get('algorithm'),
                'use_numba': run_config.get('use_numba', True),
                'ignore_tolerance': run_config.get('ignore_tolerance', False),
                'enable_best_fit': run_config.get('enable_best_fit', True),
                # Pass residual params in the expected nested structure
                'residual_threshold': run_config.get('residuals', {}).get('threshold'),
                'residual_days_window': run_config.get('residuals', {}).get('days_window'),
            }
            
            # Filter out None values to let the engine use its defaults
            engine_params = {k: v for k, v in engine_params.items() if v is not None}
            
            engine_sim = ReconciliationEngine(**engine_params)
            
            # Run the reconciliation on a copy of the dataframe
            stats = engine_sim.run(data_df.copy(), output_file=None, verbose=False)

            if stats:
                debit_perc = stats.get('_raw_debit_amount_perc', 0.0)
                credit_perc = stats.get('_raw_credit_amount_perc', 0.0)
                # The score to maximize is the sum of reconciled volume percentages
                return debit_perc + credit_perc
        except Exception:
            # If any error occurs during engine run, this trial is a failure
            return 0.0

        # If stats are not produced, trial is a failure
        return 0.0

    # --- Optuna Study Execution ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Configurazione sicura per multiprocessing in Docker
    if not sequential:
        # Usa SQLite per thread-safety con processi multipli
        os.makedirs('output', exist_ok=True)
        db_path = os.path.join(os.getcwd(), 'output', 'optuna.db')
        storage_url = f"sqlite:///{db_path}"
        study_name = f"study_{uuid.uuid4().hex}" # Nome univoco per evitare collisioni
    else:
        storage_url = None
        study_name = None

    study = optuna.create_study(direction="maximize", storage=storage_url, study_name=study_name)

    # Usa 2 job invece di -1 (tutti i core) per evitare crash OOM in Docker
    n_jobs = 1 if sequential else 2
        
    if show_progress:
        with tqdm(total=n_trials, desc="Optimization Trials") as pbar:
            def callback(study, trial):
                pbar.update(1)
            study.optimize(
                lambda trial: objective(trial, data_df=input_df),
                n_trials=n_trials, 
                n_jobs=n_jobs, 
                callbacks=[callback]
            )
    else:
        study.optimize(
            lambda trial: objective(trial, data_df=input_df),
            n_trials=n_trials, 
            n_jobs=n_jobs
        )

    # Return the best parameter set found
    best_params = study.best_params
    
    # Arrotonda i valori float (come la tolleranza) a 2 cifre decimali
    for key, value in best_params.items():
        if isinstance(value, float):
            best_params[key] = round(value, 2)

    # Pulizia dello studio dal DB per non far crescere il file all'infinito
    if storage_url:
        optuna.delete_study(study_name=study_name, storage=storage_url)

    return best_params