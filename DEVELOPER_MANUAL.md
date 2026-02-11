# Cash Reconciliator - Developer Manual

## 1. Introduction

This document provides a technical deep-dive into the Cash Reconciliator project. It is intended for developers who need to understand, maintain, or extend the application's functionality. The manual covers the project's architecture, the role of each module, key classes and functions, and the configuration system.

## 2. Project Architecture

The project is a cash reconciliation system with multiple entry points (web, command-line) that share a common core logic.

### Key Modules and Data Flow

*   **Entry Points (`app.py`, `main.py`, `batch.py`)**: These scripts handle user interaction and file I/O. They parse configurations and user input, prepare the necessary data, and then delegate the core processing to the `ReconciliationEngine`.
*   **Core Engine (`core.py`)**: This is the heart of the application. The `ReconciliationEngine` class contains all the business logic for performing the reconciliation.
*   **Reporting (`reporting.py`)**: This module is responsible for generating the final, user-friendly Excel report from the results produced by the core engine.
*   **Optimizer (`optimizer.py`)**: A utility script that uses `optuna` to find the optimal reconciliation parameters for a given dataset, helping to tune the engine for best performance.
*   **Configuration (`config.json`, `config_optimizer.json`)**: JSON files that control the behavior of the application and the optimizer.

The general data flow is as follows:
1. An entry point loads a configuration and an input data file.
2. It initializes and configures the `ReconciliationEngine`.
3. The engine runs its algorithms to find matches and produces a set of results.
4. The engine uses the `ExcelReporter` to generate a detailed `.xlsx` report.
5. The entry point saves the report and/or displays statistics to the user.

---

## 3. Module Deep Dive

### 3.1. `core.py` - The Reconciliation Engine

This module contains the `ReconciliationEngine` class, which encapsulates all business logic for the reconciliation.

#### `ReconciliationEngine.__init__(...)`

```python
def __init__(self, tolerance=0.01, days_window=7, max_combinations=10, residual_threshold=100.0, residual_days_window=30, sorting_strategy="date", search_direction="past_only", column_mapping=None, algorithm="subset_sum", use_numba=True, ignore_tolerance=False, enable_best_fit=True):
    """Initializes the ReconciliationEngine with its configuration.

    This constructor sets up the core parameters that govern the reconciliation
    algorithms. Amounts are converted from floating-point (Euros) to integers
    (cents) internally to prevent floating-point inaccuracies.

    Args:
        tolerance (float): The maximum acceptable difference between the sum of
            a set of transactions and a target amount to be considered a match.
            Default is 0.01.
        days_window (int): The primary time window (in days) to search for
            matching transactions. The search can be forward, backward, or
            in both directions from the transaction date. Default is 7.
        max_combinations (int): The maximum number of individual transactions
            that can be combined to form a match. Higher numbers increase
            computation time. Default is 10.
        residual_threshold (float): During the residual analysis pass, only
            unmatched transactions with an amount greater than this threshold
            will be considered. Default is 100.0.
        residual_days_window (int): An extended time window (in days) used
            during the final residual reconciliation pass to catch more
            difficult matches. Default is 30.
        sorting_strategy (str): The strategy for sorting transactions before
            processing. Can be 'date' (chronological) or 'amount'
            (descending). Default is 'date'.
        search_direction (str): The temporal direction for the search.
            Can be 'past_only', 'future_only', or 'both'. This determines
            the date range relative to the transaction being matched.
            Default is 'past_only'.
        column_mapping (dict, optional): A dictionary to map custom column
            names from the input file to the internal standard names
            ('Date', 'Debit', 'Credit'). Defaults to a standard mapping.
        algorithm (str): The reconciliation algorithm to use. Can be
            'subset_sum' (a complex combination-finding algorithm),
            'progressive_balance' (a faster, sequential algorithm),
            or 'auto' to let the engine choose the best one. Default is 'subset_sum'.
        use_numba (bool): If True, the engine will leverage the Numba JIT
            compiler for performance-critical calculations, if Numba is
            installed. Default is True.
        ignore_tolerance (bool): Specific to the 'progressive_balance'
            algorithm. If True, forces a block of transactions to be closed
            as a match even if the final balance is not within tolerance,
            once the time window is exceeded. Default is False.
        enable_best_fit (bool): If True, enables a "best fit" or "splitting"
            heuristic. If an exact match for a large transaction cannot be
            found, the algorithm will try to find a combination of smaller
            transactions that partially "fills" it, leaving the rest as a
            residual. Default is True.
    """
```

#### `ReconciliationEngine.run(...)`
```python
def run(self, input_file, output_file=None, verbose=True):
    """Executes the entire end-to-end reconciliation process.

    This is the main public method that orchestrates the workflow, from data
    loading to report generation. It can handle a file path or a pre-loaded
    DataFrame, making it flexible for use in different contexts like batch
    processing or parameter optimization.

    The process includes the following steps:
    1.  **Data Loading**: If `input_file` is a path, it loads the file using
        `load_file`. If it's a DataFrame, it uses it directly.
    2.  **Preprocessing**: Original totals are calculated for later verification,
        and transactions are split into separate Debit and Credit DataFrames.
    3.  **Algorithm Selection**: If `algorithm` is set to 'auto', it runs a
        quick evaluation (`_evaluate_best_configuration`) to determine the
        most effective algorithm ('subset_sum' or 'progressive_balance') for
        the given dataset and applies it.
    4.  **Reconciliation**: Executes the chosen algorithm(s). This may involve
        multiple passes with different strategies (e.g., many-to-one,
        one-to-many, residual analysis).
    5.  **Finalization**: Marks all reconciled transactions as 'used', creates
        a clean DataFrame of all matches, and verifies that the sum of
        reconciled amounts and residuals equals the original totals.
    6.  **Statistics Calculation**: Computes final summary statistics on the
        outcome (e.g., percentage of reconciled amounts, remaining balances).
    7.  **Report Generation**: If `output_file` is provided, it calls the
        `ExcelReporter` to generate a detailed multi-sheet Excel report.

    Args:
        input_file (str or pd.DataFrame): The path to the input data file
            or a pre-loaded pandas DataFrame containing the transactions.
        output_file (str, optional): The path where the final Excel report
            will be saved. If None, no report is generated. Defaults to None.
        verbose (bool): If True, detailed progress and logging information
            will be printed to the console during execution. Defaults to True.

    Returns:
        dict: A dictionary containing key statistics about the reconciliation
              results, such as the number and value of reconciled items,
              percentages, and final imbalances. Returns None if a critical
              error occurs.
    """
```

#### `ReconciliationEngine.load_file(...)`
```python
def load_file(self, file_path):
    """Loads and standardizes data from an Excel, CSV, or Feather file.

    This method is responsible for reading a source file and transforming it
    into a clean, standardized DataFrame ready for reconciliation. It performs
    several key operations:
    
    1.  **File Reading**: Supports '.xlsx', '.csv', and '.feather' formats.
    2.  **Column Mapping**: Renames columns from the source file to the
        engine's internal standard ('Date', 'Debit', 'Credit') based on the
        `column_mapping` provided during initialization.
    3.  **Date Parsing**: Converts the 'Date' column to datetime objects,
        handling common European formats (day-first). It flags rows with
        future dates.
    4.  **Amount Cleaning**: Uses a robust parser to handle various currency
        formats (e.g., "1.234,56" or "1234.56 €"). It strips symbols and
        correctly interprets decimal and thousands separators.
    5.  **Integer Conversion**: Converts 'Debit' and 'Credit' amounts into
        integer cents to eliminate floating-point arithmetic errors during
        reconciliation.
    6.  **Index Preservation**: Stores the original row number in the
        'orig_index' column for traceability in the final report.

    Args:
        file_path (str): The absolute or relative path to the input file.

    Returns:
        pd.DataFrame: A DataFrame with standardized columns ('Date', 'Debit',
        'Credit', 'orig_index'), ready for processing.

    Raises:
        ValueError: If the columns specified in the `column_mapping` are
            not found in the input file.
        FileNotFoundError: If the specified `file_path` does not exist.
    """
```

#### Core Algorithms
The engine contains two main reconciliation algorithms:

*   **`_reconcile_subset_sum(...)`**: Performs reconciliation using a multi-pass subset sum strategy. It's designed to find complex relationships (many-to-one, one-to-many) and is the most powerful but computationally intensive algorithm.
*   **`_reconcile_progressive_balance(...)`**: Performs reconciliation using a sequential "Two Pointers" algorithm. It's much faster and works well for chronologically ordered data that is already mostly balanced.

The engine also contains Numba-optimized functions (`_numba_find_combination` and `_numba_find_best_fit_combination`) to accelerate the subset sum calculations.

### 3.2. `reporting.py` - Excel Report Generator

This module contains the `ExcelReporter` class, which is responsible for creating the final `.xlsx` report.

#### `ExcelReporter.generate_report(...)`
```python
def generate_report(self, output_file, original_df):
    """Generates and saves the complete multi-sheet Excel report.

    This is the main public method of the reporter. It orchestrates the
    creation of the entire Excel workbook by calling a series of internal
    methods, each responsible for a specific sheet. The final file provides
    a comprehensive overview of the reconciliation results.

    The generated sheets include:
    - MANUAL: Explains the algorithm and parameters used.
    - Matches: A detailed list of all successful matches.
    - Unused DEBIT / Unreconciled CREDIT: Lists of transactions that were
      not matched.
    - Original: The original input data for reference.
    - Statistics: A summary of reconciliation figures.
    - Monthly Balance: A table and chart showing month-over-month trends.

    Args:
        output_file (str): The path where the Excel file will be saved.
        original_df (pd.DataFrame): The original, unmodified DataFrame that
            was fed into the reconciliation process.
    """
```

### 3.3. `optimizer.py` - Parameter Optimizer

This script is a powerful utility for finding the best reconciliation parameters for a given dataset.

#### `optimizer.run_simulation(...)`
```python
def run_simulation(base_config, optimizer_config_ranges, file_input, n_trials, show_progress=True, sequential=False):
    """Runs the core optimization process using Optuna to find the best parameters.

    This function sets up and executes an Optuna "study" to find the combination
    of reconciliation parameters that maximizes the final reconciled percentage.
    It is the central part of the optimizer.

    The process involves:
    1.  **Defining an `objective` function**: This nested function is called by
        Optuna for each trial. It suggests a new set of parameters, runs the
        reconciliation engine with them, and returns a score (the sum of the
        reconciled debit and credit volume percentages).
    2.  **Creating a `study`**: An Optuna study is created to maximize the
        score returned by the `objective` function.
    3.  **Efficient Data Loading**: To avoid costly file I/O in every trial,
        the input file is loaded into a pandas DataFrame once before the
        optimization loop begins.
    4.  **Parallel Execution**: By default, it runs trials in parallel across
        all available CPU cores (`n_jobs=-1`) for significant speedup.
    5.  **Optimization**: The `study.optimize` method is called to run the
        specified number of trials.

    Args:
        base_config (dict): A dictionary with the base configuration values,
            which will be overridden by the parameters suggested by Optuna.
        optimizer_config_ranges (dict): A dictionary defining the search space
            (min, max, step, or categories) for each parameter to be optimized.
        file_input (str): The path to the data file to be used for all
            simulation trials.
        n_trials (int): The total number of trials (i.e., different parameter
            combinations) to test.
        show_progress (bool): If True, displays a `tqdm` progress bar for the
            optimization trials. Defaults to True.
        sequential (bool): If True, forces the trials to run sequentially in a
            single process (`n_jobs=1`). Defaults to False.

    Returns:
        dict: A dictionary containing the best-performing parameter combination
              found by the study.
    """
```

### 3.4. Entry Point Scripts

#### `app.py` - Web Application
A Flask-based web server that provides a graphical user interface for the reconciliation tool. It allows users to upload a file, set parameters, and download the resulting report.

#### `main.py` - Single-File Wrapper
A command-line script for processing a single file. It is configured via a specific JSON file passed as an argument and can run in a "silent" mode, making it suitable for being called by other scripts.

#### `batch.py` - Batch Processor
A command-line script for processing an entire directory of files. It reads a central `config.json` and iterates through all matching files in the input folder, saving a result for each one.

## 4. Configuration

The application is primarily configured through two JSON files:

*   **`config.json`**: The main configuration file used by `app.py` and `batch.py`. It defines default parameters for the `ReconciliationEngine` (e.g., `tolerance`, `days_window`), specifies file paths, and sets the column mapping.

*   **`config_optimizer.json`**: The configuration file for `optimizer.py`. It defines the search space for each parameter that Optuna will test (e.g., the min/max values for `days_window`).

By modifying these files, developers can change the default behavior of the application without altering the source code.
