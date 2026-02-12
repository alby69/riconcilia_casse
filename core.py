import pandas as pd
from itertools import combinations
from collections import deque
from datetime import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')
import sys
import io
import contextlib

# --- CHANGE: Management of Numba as an optional dependency ---
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a dummy decorator if Numba is not available
    def jit(signature_or_function=None, locals={}, cache=False, pipeline_class=None, boundscheck=None, **options):
        def decorator(func):
            return func
        return decorator

def _robust_currency_parser(value):
    """
    Robustly converts a string or number into a standard numeric format for pd.to_numeric.
    This helper function is used by `load_file`.
    """
    # If it's already a number, it's fine.
    if isinstance(value, (int, float)):
        return value
    # If it's not a string, we can't do anything.
    if not isinstance(value, str):
        return None # Will be converted to NaN
    
    # Clean the string from spaces and euro symbol
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    
    # Case 1: Full Italian format (e.g., "1.234,56")
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    # Case 2: Italian format with only decimals (e.g., "1234,56")
    elif ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
    # Case 3: Format without commas (e.g., "1234" or "1234.56"). We leave the dot.
    return cleaned_str

class ReconciliationEngine:
    """Contains the business logic for reconciliation."""

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
        # FIX: Converts values from euros (float) to cents (int) for internal consistency
        self.tolerance = int(tolerance * 100)
        self.days_window = days_window
        self.max_combinations = max_combinations
        # FIX: Converts values from euros (float) to cents (int)
        self.residual_threshold = int(residual_threshold * 100)
        self.residual_days_window = residual_days_window
        self.sorting_strategy = sorting_strategy
        self.search_direction = search_direction
        self.algorithm = algorithm
        # ADDITION: Sets the column mapping, with a default if not provided.
        self.column_mapping = column_mapping or {'Date': 'Date', 'Debit': 'Debit', 'Credit': 'Credit'}
        
        # Flag to enable/disable Numba
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        # Flag to force closing blocks in Progressive Balance even if they don't match (on window timeout)
        self.ignore_tolerance = ignore_tolerance

        # ADDITION: Flag to enable best-fit logic (splitting)
        self.enable_best_fit = enable_best_fit

        # Internal state that will be populated during execution
        self.debit_df = self.credit_df = self.matches_df = None
        self.unused_debit_df = self.unreconciled_credit_df = self.original_df = None

        # Optimization: Use sets to keep track of used indices
        self.used_debit_indices = set()
        self.used_credit_indices = set()
        
        # Counter to generate new unique IDs for residuals
        self.max_id_counter = 0

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
        # Common parameters for reading CSV/Excel with European format
        common_read_params = {
            'decimal': ',',
            'thousands': '.'
        }

        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, decimal=',', thousands='.')
        elif str(file_path).endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            # This branch should only be reached if it's not a feather and not a CSV.
            # For Excel files, convert_to_feather.py should have already handled the parsing.
            df = pd.read_excel(file_path, decimal=',', thousands='.', engine='openpyxl')

        # --- CHANGE: Dynamic handling of column names ---
        # Invert the map for renaming: {'Source Column Name': 'Internal Name'} -> {'Internal Name': 'Source Column Name'}
        source_col_names = self.column_mapping.keys()
        
        # Check if the source columns defined in the configuration exist in the file
        if not set(source_col_names).issubset(df.columns):
            missing_cols = set(source_col_names) - set(df.columns)
            raise ValueError(f"The input file does not contain the source columns specified in the configuration: {', '.join(missing_cols)}")

        # Rename the DataFrame columns using the mapping to standardize them to internal names ('Date', 'Debit', 'Credit')
        df.rename(columns=self.column_mapping, inplace=True)

        # After reading, ensure 'Date' is datetime and 'Debit'/'Credit' are numeric.
        # This is a fallback in case the reading parameters were not sufficient
        # or if the DataFrame comes from an already pre-loaded source (e.g., from the optimizer).
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df.dropna(subset=['Date'], inplace=True) # Removes rows with invalid dates

        # --- CHECK DATE FUTURE ---
        today = datetime.now()
        future_rows = df[df['Date'] > today]
        if not future_rows.empty:
            print(f"\n⚠️  WARNING: Found {len(future_rows)} movements with a future date (compared to {today.strftime('%Y-%m-%d')})!")
            print(f"    Example: {future_rows.iloc[0]['Date'].strftime('%Y-%m-%d')} (Row {future_rows.index[0] + 2})")

        # --- ROBUST AMOUNT CLEANING ---
        # Apply the robust parser to each cell, then convert the entire column.
        for col in ['Debit', 'Credit']:
            df[col] = pd.to_numeric(df[col].apply(_robust_currency_parser), errors='coerce')

        # Fill non-numeric values with 0 BEFORE converting to cents
        df[['Debit', 'Credit']] = df[['Debit', 'Credit']].fillna(0)

        # --- OPTIMIZATION: Convert to integers (cents) to avoid floating point errors ---
        # We multiply by 100 and round for safety, then convert to integers.
        df['Debit'] = (df['Debit'] * 100).round().astype(int)
        df['Credit'] = (df['Credit'] * 100).round().astype(int)

        df['orig_index'] = df.index
        
        # Initialize the ID counter with the maximum existing index
        if not df.empty:
            self.max_id_counter = df.index.max()
            
        return df

    def _separate_movements(self, df):
        """Separates the DataFrame into DEBIT and CREDIT movements."""
        self.debit_df = df[df['Debit'] != 0][['orig_index', 'Date', 'Debit']].copy()
        self.credit_df = df[df['Credit'] != 0][['orig_index', 'Date', 'Credit']].copy()
       
        if self.sorting_strategy == "date":
            self.debit_df = self.debit_df.sort_values('Date', ascending=True)
            self.credit_df = self.credit_df.sort_values('Date', ascending=True)
        elif self.sorting_strategy == "amount":
            self.debit_df = self.debit_df.sort_values('Debit', ascending=False)
            self.credit_df = self.credit_df.sort_values('Credit', ascending=False)
        else:
            raise ValueError(f"Invalid sorting strategy: '{self.sorting_strategy}'. Use 'date' or 'amount'.")
            
        return self.debit_df, self.credit_df

    def _find_matches(self, debit_row, credit_candidates_list, unused_map, days_window, max_combinations, enable_best_fit=False):
        """Internal logic to find a match for a single DEBIT. Receives pre-filtered candidates by date."""
        debit_amount = debit_row['Debit']
        debit_date = debit_row['Date']

        # --- CORRECT OPTIMIZATION: Filter the list of dictionaries ---
        # credit_candidates_list is now a list of dictionaries, not a numpy array
        # Candidates are already pre-filtered by date and unused indices. We only filter by amount.
        credit_candidates = [
            c for c in credit_candidates_list if c['Credit'] <= debit_amount + self.tolerance
        ]
        
        if not credit_candidates:
            return None
        
        # 1. Search for exact 1-to-1 match
        exact_match_list = [c for c in credit_candidates if abs(c['Credit'] - debit_amount) <= self.tolerance]
        if exact_match_list:
            best_match = exact_match_list[0] # Takes the first exact match found
            return {
                'debit_indices': [debit_row['orig_index']],
                'debit_dates': [debit_date],
                'debit_amounts': [debit_amount],
                'credit_indices': [best_match['orig_index']],
                'credit_dates': [best_match['Date']],
                'credit_amounts': [best_match['Credit']],
                'total_credit': best_match['Credit'],
                'difference': abs(debit_amount - best_match['Credit']),
                'match_type': '1-to-1'
            }

        # 2. Search for multiple combinations in an optimized way
        credit_candidates = sorted(credit_candidates, key=lambda x: x['Credit'], reverse=True)
        
        # Added cache for memoization
        cache = {}
        total_candidates_sum = sum(c['Credit'] for c in credit_candidates)
        
        match = None
        if self.use_numba:
            # --- NUMBA OPTIMIZATION ---
            candidates_np_numba = np.array([(c['Credit'], c['orig_index']) for c in credit_candidates], dtype=np.int64)
            match_indices = _numba_find_combination(debit_amount, candidates_np_numba, max_combinations, self.tolerance)
            if len(match_indices) > 0:
                match = [c for c in credit_candidates if c['orig_index'] in match_indices]
        else:
            # --- FALLBACK TO PURE PYTHON ---
            for c in credit_candidates: c['Debit'] = c.pop('Credit') # Adapt for generic function
            match = self._find_combinations_recursive_py(debit_amount, credit_candidates, max_combinations, self.tolerance)
            if match:
                for c in match: c['Credit'] = c.pop('Debit') # Restore

        if match:
            return {
                'debit_indices': [debit_row['orig_index']],
                'debit_dates': [debit_date],
                'debit_amounts': [debit_amount],
                'credit_indices': [m['orig_index'] for m in match],
                'credit_dates': [m['Date'] for m in match],
                'credit_amounts': [m['Credit'] for m in match],
                'total_credit': sum(m['Credit'] for m in match),
                'difference': abs(debit_amount - sum(m['Credit'] for m in match)),
                'match_type': f'Combination {len(match)}'
            }
        
        return None

    def _find_combinations_recursive_py(self, target, candidates, max_combinations, tolerance):
        """Iterative function (stack-based) for subset-sum. Pure Python version."""
        stack = deque([(0, 0, [])]) # (start_index, current_sum, current_path)

        while stack:
            idx, current_sum, current_path = stack.pop()

            # --- SUCCESS CONDITION ---
            if abs(target - current_sum) <= tolerance and len(current_path) > 1:
                return current_path

            # --- PRUNING CONDITIONS ---
            if len(current_path) >= max_combinations or idx >= len(candidates):
                continue

            # --- BRANCH 1: Exclude the current candidate ---
            # Continue exploration from the next candidate.
            stack.append((idx + 1, current_sum, current_path))

            # --- BRANCH 2: Include the current candidate ---
            candidate = candidates[idx]
            new_sum = current_sum + candidate['Debit'] # The generic function uses 'Debit'
            
            # Pruning: do not include if the new sum already exceeds too much
            if new_sum > target + tolerance:
                continue

            new_path = current_path + [candidate]
            
            # --- SUCCESS CONDITION (even after addition) ---
            if abs(target - new_sum) <= tolerance and len(new_path) > 1:
                return new_path

            # Continue exploration including the current element
            stack.append((idx + 1, new_sum, new_path))

        return None # No combination found

    def _find_debit_matches(self, credit_row, debit_candidates_list, unused_map, days_window, max_combinations, enable_best_fit=False):
        """Logic to find combinations of DEBIT that match a CREDIT. Receives pre-filtered candidates."""
        credit_amount = credit_row['Credit']
        credit_date = credit_row['Date']

        # Candidates are already pre-filtered by date and unused indices. We only filter by amount.
        debit_candidates = [c for c in debit_candidates_list if c['Debit'] <= credit_amount + self.tolerance]

        if not debit_candidates:
            return None

        # Search for multiple DEBIT combinations in an optimized way
        candidates_to_modify = [c.copy() for c in debit_candidates]
        candidates_to_modify = sorted(candidates_to_modify, key=lambda x: x['Debit'], reverse=True)

        match = None
        is_partial = False
        
        if self.use_numba:
            candidates_np_numba = np.array([(c['Debit'], c['orig_index']) for c in candidates_to_modify], dtype=np.int64)
            # Attempt 1: Exact Match
            match_indices = _numba_find_combination(credit_amount, candidates_np_numba, max_combinations, self.tolerance)
            
            if len(match_indices) > 0:
                match = [c for c in candidates_to_modify if c['orig_index'] in match_indices]
            elif enable_best_fit:
                # Attempt 2: Best Fit (Partial Match)
                # Find the combination that best fills the payment without exceeding it
                match_indices = _numba_find_best_fit_combination(credit_amount, candidates_np_numba, max_combinations, self.tolerance)
                if len(match_indices) > 0:
                    match = [c for c in candidates_to_modify if c['orig_index'] in match_indices]
                    is_partial = True
        else:
            match = self._find_combinations_recursive_py(credit_amount, candidates_to_modify, max_combinations, self.tolerance)

        if match:
            total_debit = sum(m['Debit'] for m in match)
            difference = abs(credit_amount - total_debit)
            
            # If it's a partial best fit, we calculate the residual
            residual = 0
            if is_partial and difference > self.tolerance:
                residual = difference
            
            return {
                'debit_indices': [m['orig_index'] for m in match],
                'debit_dates': [m['Date'] for m in match],
                'debit_amounts': [m['Debit'] for m in match],
                'credit_indices': [credit_row['orig_index']],
                'credit_dates': [credit_date],
                'credit_amounts': [credit_amount],
                'total_debit': total_debit,
                'difference': difference,
                'match_type': f'DEBIT Combination {len(match)}' + (' (Best Fit)' if is_partial else ''),
                'residual': residual if is_partial else 0
            }
        return None

    def _run_reconciliation_pass_debit(self, debit_df, credit_df, days_window, max_combinations, matches_list, title, verbose=True, enable_best_fit=False):
        """Runs a pass looking for DEBIT combinations to match CREDIT (optimized with NumPy)."""
        # Filter CREDITs that have not been used yet
        credit_to_process = credit_df[~credit_df['orig_index'].isin(self.used_credit_indices)].copy() if credit_df is not None and not credit_df.empty else pd.DataFrame()

        self._run_generic_pass(
            df_to_process=credit_to_process,
            df_candidates=debit_df,
            col_to_process='Credit',
            col_candidates='Debit',
            used_indices_candidates=self.used_debit_indices,
            days_window=days_window,
            max_combinations=max_combinations,
            matches_list=matches_list,
            title=title,
            search_direction=self.search_direction, # Use the main direction from the configuration
            find_function=self._find_debit_matches,
            verbose=verbose,
            enable_best_fit=enable_best_fit
        )

    def _run_generic_pass(self, df_to_process, df_candidates, col_to_process, col_candidates, used_indices_candidates, days_window, max_combinations, matches_list, title, search_direction, find_function, verbose=True, enable_best_fit=False):
        """
        Generic helper function that performs a reconciliation pass.
        
        Iterates over each row of `df_to_process` and searches for matches in `df_candidates`
        using the provided `find_function`. Manages time window logic,
        match registration, and splitting (Best Fit).

        Args:
            df_to_process (pd.DataFrame): Main DataFrame to iterate over.
            df_candidates (pd.DataFrame): DataFrame where to search for combinations.
            col_to_process (str): Amount column name in the main DF ('Debit' or 'Credit').
            col_candidates (str): Amount column name in the candidate DF.
            used_indices_candidates (set): Set of already used indices to exclude.
            days_window (int): Time window for the search.
            max_combinations (int): Max combinable elements.
            matches_list (list): List to append found matches to.
            title (str): Title of the pass for logging.
            search_direction (str): Time direction ('past_only', 'future_only', 'both').
            find_function (callable): Function that implements the specific matching logic.
            verbose (bool): If True, prints logs.
            enable_best_fit (bool): If True, enables splitting logic for partial matches.
        """
        if df_to_process is None or df_to_process.empty:
            return

        if verbose:
            print(f"\n{title} (Direction: {search_direction})...")

        # Prepare the record lists once
        records_to_process = df_to_process.to_dict('records')
        records_candidates = sorted(df_candidates.to_dict('records'), key=lambda x: x['Date']) if df_candidates is not None else []

        matches = []
        total_records = len(records_to_process)
        processed_count = 0
        
        # List to collect new residual movements generated by splitting
        new_residuals = []

        for record_row in records_to_process:
            if verbose:
                processed_count += 1
                percentage = (processed_count / total_records) * 100
                sys.stdout.write(f"\r   - Progress: {percentage:.1f}% ({processed_count}/{total_records})")
                sys.stdout.flush()

            # Pre-filter candidates by time window
            min_date, max_date = self._calculate_time_window(record_row['Date'], days_window, search_direction)
            
            candidates_prefiltered = [
                c for c in records_candidates
                if min_date <= c['Date'] <= max_date and c['orig_index'] not in used_indices_candidates
            ]

            if candidates_prefiltered:
                match = find_function(record_row, candidates_prefiltered, None, days_window, max_combinations, enable_best_fit=enable_best_fit)
                if match:
                    # Handle split (Best Fit)
                    residual = match.get('residual', 0)
                    if residual > 0:
                        # Create a new movement for the residual
                        new_movement = self._create_residual_movement(record_row, residual, col_to_process)
                        new_residuals.append(new_movement)

                        # Update the amount of the original row to reflect only the reconciled part
                        # This corrects the statistics by avoiding amount duplication (Original + Residual)
                        idx_orig = record_row['orig_index']
                        new_amount = record_row[col_to_process] - residual
                        
                        if col_to_process == 'Credit':
                            self.credit_df.loc[self.credit_df['orig_index'] == idx_orig, 'Credit'] = new_amount
                            # FIX REPORT: Also update the match to show only the used part in Excel
                            match['credit_amounts'] = [new_amount]
                            match['total_credit'] = new_amount
                            match['difference'] = abs(match.get('total_debit', 0) - new_amount)
                        elif col_to_process == 'Debit':
                            self.debit_df.loc[self.debit_df['orig_index'] == idx_orig, 'Debit'] = new_amount
                            # FIX REPORT
                            match['debit_amounts'] = [new_amount]
                            match['total_debit'] = new_amount
                            match['difference'] = abs(match.get('total_credit', 0) - new_amount)

                    # --- CRITICAL FIX: IMMEDIATE REGISTRATION ---
                    # Immediately register the match to mark indices as used and prevent
                    # them from being reused in the same pass (Double Spending).
                    match['pass_name'] = title
                    self._register_match(match, matches_list)
                    matches.append(match) # Keeps the list only for the final count

        if verbose: print(f"\n   - Registered {len(matches)} matches.")
            
        # Add the generated residuals to the original DataFrame to be processed in subsequent passes
        if new_residuals:
            if verbose: print(f"   - Generated {len(new_residuals)} residual movements from split (Best Fit).")
            df_residuals = pd.DataFrame(new_residuals)
            
            if col_to_process == 'Credit':
                self.credit_df = pd.concat([self.credit_df, df_residuals], ignore_index=True)
                # Ensure types are correct
                self.credit_df['Credit'] = self.credit_df['Credit'].astype(int)
            elif col_to_process == 'Debit':
                self.debit_df = pd.concat([self.debit_df, df_residuals], ignore_index=True)
                self.debit_df['Debit'] = self.debit_df['Debit'].astype(int)
                
        if verbose: sys.stdout.write("\n   ✓ Completed.\n")

    def _create_residual_movement(self, original_record, residual_amount, type_col):
        """Creates a dictionary representing the residual movement."""
        self.max_id_counter += 1
        new_id = self.max_id_counter
        
        new_movement = original_record.copy()
        new_movement['orig_index'] = new_id
        new_movement[type_col] = residual_amount
        # Note: 'used' will be False (or NaN which will be treated as False) by default when added to the DF
        
        return new_movement

    def _calculate_time_window(self, reference_date, days_window, search_direction):
        """Calculates the time window (min_date, max_date) based on the search direction."""
        if search_direction == "future_only":
            min_date = reference_date
            max_date = reference_date + pd.Timedelta(days=days_window)
        elif search_direction == "past_only":
            min_date = reference_date - pd.Timedelta(days=days_window)
            max_date = reference_date
        elif search_direction == "both":
            min_date = reference_date - pd.Timedelta(days=days_window)
            max_date = reference_date + pd.Timedelta(days=days_window)
        else:
            raise ValueError(f"Invalid time search direction: '{search_direction}'. Use 'both', 'future_only' or 'past_only'.")
        return min_date, max_date

    def _run_reconciliation_pass(self, debit_df, credit_df, days_window, max_combinations, matches_list, title, verbose=True):
        """Performs a reconciliation pass and updates the DataFrames (optimized with NumPy)."""
        # Filter DEBITs that have not been used yet
        debit_to_process = debit_df[~debit_df['orig_index'].isin(self.used_debit_indices)].copy() if debit_df is not None and not debit_df.empty else pd.DataFrame()
        
        # --- FIX: Logical inversion of the direction for the DEBIT->CREDIT pass ---
        # If the global strategy is "past_only" (DEBIT before CREDIT),
        # when we start from DEBIT we must search for CREDIT in the future ("future_only").
        direction_for_pass2 = self.search_direction
        if self.search_direction == "past_only":
            direction_for_pass2 = "future_only"
        elif self.search_direction == "future_only":
            direction_for_pass2 = "past_only"
        
        self._run_generic_pass(
            df_to_process=debit_to_process,
            df_candidates=credit_df,
            col_to_process='Debit',
            col_candidates='Credit',
            used_indices_candidates=self.used_credit_indices,
            days_window=days_window,
            max_combinations=max_combinations,
            matches_list=matches_list,
            title=title,
            search_direction=direction_for_pass2, # Use the correct (inverted) direction
            find_function=self._find_matches,
            verbose=verbose
        )

    def _reconcile_subset_sum(self, verbose=True):
        """Performs reconciliation using a multi-pass subset sum strategy."""
        
        matches = []

        # Pass 1: DEBIT combination for CREDIT (Many Receipts -> 1 Deposit)
        self._run_reconciliation_pass_debit(
            self.debit_df, self.credit_df,
            self.days_window,
            self.max_combinations,
            matches,
            "Pass 1: Receipt Aggregation (Many DEBIT -> 1 CREDIT) [with Best Fit]",
            verbose,
            enable_best_fit=self.enable_best_fit
        )

        # Pass 2: Standard Inverse Reconciliation (1 Receipt -> Many Deposits)
        self._run_reconciliation_pass(
            self.debit_df, self.credit_df,
            self.days_window,
            self.max_combinations,
            matches,
            "Pass 2: Split Deposits (1 DEBIT -> Many CREDIT)",
            verbose
        )

        # Pass 3: Residual Analysis (Enlarged Window)
        self._run_reconciliation_pass_debit(
            self.debit_df, self.credit_df,
            self.residual_days_window,
            self.max_combinations,
            matches,
            f"Pass 3: Residual Recovery (Extended window: {self.residual_days_window}d)",
            verbose,
            enable_best_fit=False
        )

        return matches

    def _reconcile_greedy_amount_first(self, verbose=True):
        """
        Performs reconciliation using a greedy, amount-first strategy.
        It sorts both debits and credits by amount (descending) and tries to match
        the largest remaining items first.
        """
        if verbose:
            print("\nStarting reconciliation with 'Greedy Amount First' algorithm...")

        # 1. Prepare data: Filter unused and Sort by Amount (descending)
        df_debit_temp = self.debit_df[~self.debit_df['orig_index'].isin(self.used_debit_indices)].copy()
        df_credit_temp = self.credit_df[~self.credit_df['orig_index'].isin(self.used_credit_indices)].copy()
        
        df_debit_temp.sort_values(by='Debit', ascending=False, inplace=True)
        df_credit_temp.sort_values(by='Credit', ascending=False, inplace=True)

        matches = []
        
        # We iterate over the largest set to ensure we try to match every large transaction
        if len(df_debit_temp) > len(df_credit_temp):
            # Iterate through debits and find credits
            self._run_generic_pass(
                df_to_process=df_debit_temp,
                df_candidates=df_credit_temp,
                col_to_process='Debit',
                col_candidates='Credit',
                used_indices_candidates=self.used_credit_indices,
                days_window=self.days_window,
                max_combinations=self.max_combinations,
                matches_list=matches,
                title="Greedy Pass (Debit -> Credit)",
                search_direction=self.search_direction,
                find_function=self._find_matches,
                verbose=verbose,
                enable_best_fit=True
            )
        else:
            # Iterate through credits and find debits
            self._run_generic_pass(
                df_to_process=df_credit_temp,
                df_candidates=df_debit_temp,
                col_to_process='Credit',
                col_candidates='Debit',
                used_indices_candidates=self.used_debit_indices,
                days_window=self.days_window,
                max_combinations=self.max_combinations,
                matches_list=matches,
                title="Greedy Pass (Credit -> Debit)",
                search_direction=self.search_direction,
                find_function=self._find_debit_matches,
                verbose=verbose,
                enable_best_fit=True
            )
            
        return matches

    def _finalize_matches(self, matches):
        """Creates the final DataFrame, generates IDs, and sorts the results."""
        # Expected columns in the final DataFrame
        final_columns = [
            'Transaction ID', 'debit_indices', 'debit_dates', 'debit_amounts', 
            'credit_date', 'num_credits', 'credit_indices', 'credit_amounts', 
            'total_credit', 'difference', 'days_diff', 'match_type', 'pass_name'
        ]

        # Creation of the final matches DataFrame
        if matches:
            df_matches = pd.DataFrame(matches)
            # Handling of missing columns (e.g., 'somma_dare' vs 'somma_avere')
            if 'total_debit' in df_matches.columns and 'total_credit' not in df_matches.columns:
                df_matches['total_credit'] = df_matches['total_debit']
            
            # Calculation of day difference (Credit - Debit)
            df_matches['days_diff'] = df_matches.apply(
                lambda row: (row['credit_date'] - min(row['debit_dates'])).days 
                if isinstance(row['debit_dates'], list) and len(row['debit_dates']) > 0 and pd.notnull(row['credit_date']) 
                else None, axis=1
            )

            # --- CHANGE: Creation of the Transaction ID with new format D(..)_A(..) ---
            df_matches['Transaction ID'] = df_matches.apply(
                lambda row: "D({})_A({})".format(
                    ','.join(map(str, [i + 2 for i in row['debit_indices']])),
                    ','.join(map(str, [i + 2 for i in row['credit_indices']]))
                ), axis=1
            )
            df_matches['sort_date'] = df_matches['debit_dates'].apply(lambda x: x[0] if isinstance(x, list) and x else pd.NaT)
            df_matches['sort_importo'] = df_matches['debit_amounts'].apply(lambda x: sum(x) if isinstance(x, list) else 0)
            df_matches = df_matches.sort_values(by=['sort_date', 'sort_importo'], ascending=[True, False]).drop(columns=['sort_date', 'sort_importo'])
            df_matches = df_matches.reindex(columns=final_columns) # Ensures all columns exist
        else:
            df_matches = pd.DataFrame(columns=final_columns)
            
        return df_matches

    def _reconcile_progressive_balance(self, verbose=True):
        """Performs reconciliation using a sequential progressive balance algorithm."""
        from datetime import timedelta # Make sure it's imported
        if verbose:
            print("\nStarting reconciliation with 'Progressive Balance' algorithm (Sequential)...")

        # 1. Prepare data: Filter unused and Sort by Date
        df_debit_temp = self.debit_df[~self.debit_df['orig_index'].isin(self.used_debit_indices)].copy()
        df_credit_temp = self.credit_df[~self.credit_df['orig_index'].isin(self.used_credit_indices)].copy()
        
        df_debit_temp.sort_values(by=['Date', 'orig_index'], inplace=True)
        df_credit_temp.sort_values(by=['Date', 'orig_index'], inplace=True)

        debit_rows = df_debit_temp.to_dict('records')
        credit_rows = df_credit_temp.to_dict('records')
        
        n_debit = len(debit_rows)
        n_credit = len(credit_rows)
        
        i = j = start_i = start_j = cum_debit = cum_credit = 0
        matches = []
        
        if verbose:
            print(f"   - Sequential analysis on {n_debit} Debit movements and {n_credit} Credit movements...")

        while i < n_debit or j < n_credit:
            diff = cum_debit - cum_credit
            
            start_date_debit = debit_rows[start_i]['Date'] if start_i < n_debit else None
            start_date_credit = credit_rows[start_j]['Date'] if start_j < n_credit else None
            curr_date_debit = debit_rows[i]['Date'] if i < n_debit else (debit_rows[i-1]['Date'] if i > 0 else None)
            curr_date_credit = credit_rows[j]['Date'] if j < n_credit else (credit_rows[j-1]['Date'] if j > 0 else None)
            
            valid_starts = [d for d in [start_date_debit, start_date_credit] if d is not None]
            valid_ends = [d for d in [curr_date_debit, curr_date_credit] if d is not None]
            
            should_reset = False
            if valid_starts and valid_ends:
                block_duration = (max(valid_ends) - min(valid_starts)).days
                if block_duration > self.days_window and (cum_debit > 0 or cum_credit > 0):
                    should_reset = True

            if should_reset:
                if self.ignore_tolerance:
                    block_debit = debit_rows[start_i:i]
                    block_credit = credit_rows[start_j:j]
                    match = {
                        'debit_indices': [r['orig_index'] for r in block_debit],
                        'debit_dates': [r['Date'] for r in block_debit],
                        'debit_amounts': [r['Debit'] for r in block_debit],
                        'credit_indices': [r['orig_index'] for r in block_credit],
                        'credit_dates': [r['Date'] for r in block_credit],
                        'credit_amounts': [r['Credit'] for r in block_credit],
                        'total_credit': cum_credit,
                        'difference': abs(diff),
                        'match_type': f'Forced (Timeout {self.days_window}d)',
                        'pass_name': 'Progressive Balance (Forced)'
                    }
                    self._register_match(match, matches)
                    start_i, start_j, cum_debit, cum_credit = i, j, 0, 0
                else:
                    start_i, start_j, cum_debit, cum_credit = i, j, 0, 0
            
            if abs(diff) <= self.tolerance and (i > start_i or j > start_j):
                block_debit = debit_rows[start_i:i]
                block_credit = credit_rows[start_j:j]
                match = {
                    'debit_indices': [r['orig_index'] for r in block_debit],
                    'debit_dates': [r['Date'] for r in block_debit],
                    'debit_amounts': [r['Debit'] for r in block_debit],
                    'credit_indices': [r['orig_index'] for r in block_credit],
                    'credit_dates': [r['Date'] for r in block_credit],
                    'credit_amounts': [r['Credit'] for r in block_credit],
                    'total_credit': cum_credit,
                    'difference': abs(diff),
                    'match_type': f'Progressive Balance (Seq. {len(block_debit)}D vs {len(block_credit)}C)',
                    'pass_name': 'Progressive Balance'
                }
                self._register_match(match, matches)
                start_i, start_j, cum_debit, cum_credit = i, j, 0, 0
                if i == n_debit and j == n_credit: break
            
            can_advance_debit = i < n_debit
            can_advance_credit = j < n_credit
            
            if can_advance_debit and can_advance_credit:
                if cum_debit < cum_credit:
                    cum_debit += debit_rows[i]['Debit']; i += 1
                elif cum_credit < cum_debit:
                    cum_credit += credit_rows[j]['Credit']; j += 1
                else:
                    if debit_rows[i]['Date'] <= credit_rows[j]['Date']:
                        cum_debit += debit_rows[i]['Debit']; i += 1
                    else:
                        cum_credit += credit_rows[j]['Credit']; j += 1
            elif can_advance_debit:
                cum_debit += debit_rows[i]['Debit']; i += 1
            elif can_advance_credit:
                cum_credit += credit_rows[j]['Credit']; j += 1
            else:
                break

        if verbose:
            print(f"   - Found {len(matches)} balanced blocks.")
        return matches

    def _register_match(self, match, matches_list):
        """Marks the elements as 'used' and registers the match."""
        if not match: return
        debit_indices_orig = match.get('debit_indices', [])
        credit_indices_orig = match.get('credit_indices', [])
        self.used_debit_indices.update(debit_indices_orig)
        self.used_credit_indices.update(credit_indices_orig)
        credit_dates = match.get('credit_dates')
        matches_list.append({
            'debit_indices': debit_indices_orig,
            'debit_dates': match.get('debit_dates', []),
            'debit_amounts': match.get('debit_amounts', []),
            'credit_date': min(credit_dates) if credit_dates else None,
            'num_credits': len(credit_indices_orig),
            'credit_indices': credit_indices_orig,
            'credit_amounts': match.get('credit_amounts', []),
            'total_credit': match.get('total_credit', match.get('total_debit', 0)),
            'difference': match.get('difference', 0),
            'match_type': match.get('match_type', 'N/D'),
            'pass_name': match.get('pass_name', 'N/D')
        })

    def _calculate_monthly_balance(self):
        """Calculates aggregate statistics by month to identify periodic imbalances."""
        if self.debit_df is None or self.credit_df is None: return pd.DataFrame()

        def aggregate(df, value_col):
            if df.empty: return pd.DataFrame()
            temp = df.copy()
            temp['Month'] = pd.to_datetime(temp['Date']).dt.to_period('M')
            total = temp.groupby('Month')[value_col].sum()
            used = temp[temp['used']].groupby('Month')[value_col].sum()
            res = pd.DataFrame({f'Total {value_col}': total, f'Used {value_col}': used})
            return res.fillna(0)

        stats_debit = aggregate(self.debit_df, 'Debit')
        stats_credit = aggregate(self.credit_df, 'Credit')
        stats = pd.merge(stats_debit, stats_credit, left_index=True, right_index=True, how='outer')
        
        absorbed_imbalance = pd.DataFrame()
        if self.matches_df is not None and not self.matches_df.empty:
            df_temp_matches = self.matches_df.copy()
            df_temp_matches['Month'] = df_temp_matches['debit_dates'].apply(lambda x: x[0].to_period('M') if isinstance(x, list) and x else None)
            df_temp_matches.dropna(subset=['Month'], inplace=True)
            df_temp_matches['total_debit'] = df_temp_matches['debit_amounts'].apply(lambda x: sum(x) if isinstance(x, list) else 0)
            df_temp_matches['block_imbalance'] = df_temp_matches['total_debit'] - df_temp_matches['total_credit']
            absorbed_imbalance = df_temp_matches.groupby('Month')['block_imbalance'].sum().to_frame('Absorbed Imbalance (in match)')

        if not absorbed_imbalance.empty:
            stats = pd.merge(stats, absorbed_imbalance, left_index=True, right_index=True, how='outer')

        stats = stats.fillna(0)
        stats['Unmatched DEBIT'] = stats['Total Debit'] - stats['Used Debit']
        stats['Unmatched CREDIT'] = stats['Total Credit'] - stats['Used Credit']
        stats['Residual Imbalance (DEBIT - CREDIT)'] = stats['Unmatched DEBIT'] - stats['Unmatched CREDIT']
        if 'Absorbed Imbalance (in match)' not in stats.columns:
            stats['Absorbed Imbalance (in match)'] = 0
        stats['Final Monthly Imbalance'] = stats['Residual Imbalance (DEBIT - CREDIT)'] + stats['Absorbed Imbalance (in match)']
        stats = stats[['Total Debit', 'Used Debit', 'Unmatched DEBIT', 'Total Credit', 'Used Credit', 'Unmatched CREDIT', 'Residual Imbalance (DEBIT - CREDIT)', 'Absorbed Imbalance (in match)', 'Final Monthly Imbalance']]
        stats = stats.sort_index()
        stats.index = stats.index.astype(str)
        stats.index.name = 'Month'
        return stats.reset_index()

    def _verify_total_balance(self, tot_debit_orig, tot_credit_orig, verbose=True):
        if self.debit_df is None or self.credit_df is None: return
        tot_debit_final = self.debit_df['Debit'].sum()
        tot_credit_final = self.credit_df['Credit'].sum()
        diff_debit = tot_debit_final - tot_debit_orig
        diff_credit = tot_credit_final - tot_credit_orig
        if verbose:
            print("\n🔍 Verifying Total Balances (Original vs Final):")
            print(f"   DEBIT:  {tot_debit_orig/100:,.2f} € (Orig) vs {tot_debit_final/100:,.2f} € (Fin) -> Delta: {diff_debit/100:,.2f} €")
            print(f"   CREDIT: {tot_credit_orig/100:,.2f} € (Orig) vs {tot_credit_final/100:,.2f} € (Fin) -> Delta: {diff_credit/100:,.2f} €")
        if abs(diff_debit) > 1 or abs(diff_credit) > 1:
             print(f"⚠️  WARNING: Discrepancy detected in totals! DEBIT: {diff_debit}, CREDIT: {diff_credit}", file=sys.stderr)
        elif verbose:
             print("   ✅ Balance confirmed: No loss of amounts during splitting.")

    def create_excel_report(self, output_file, original_df):
        from reporting import ExcelReporter
        reporter = ExcelReporter(self)
        reporter.generate_report(output_file, original_df)

    def get_stats(self):
        if self.debit_df is None or self.credit_df is None or 'used' not in self.debit_df.columns or 'used' not in self.credit_df.columns: return {}
        num_debit_tot, amt_debit_tot = len(self.debit_df), self.debit_df['Debit'].sum()
        num_debit_used, amt_debit_used = int(self.debit_df['used'].sum()), self.debit_df[self.debit_df['used']]['Debit'].sum()
        num_credit_tot, amt_credit_tot = len(self.credit_df), self.credit_df['Credit'].sum()
        num_credit_used, amt_credit_used = int(self.credit_df['used'].sum()), self.credit_df[self.credit_df['used']]['Credit'].sum()
        unused_debit_amount = (self.unused_debit_df['Debit'].sum() / 100) if self.unused_debit_df is not None and not self.unused_debit_df.empty else 0
        unreconciled_credit_amount = (self.unreconciled_credit_df['Credit'].sum() / 100) if self.unreconciled_credit_df is not None and not self.unreconciled_credit_df.empty else 0
        structural_imbalance = amt_debit_tot - amt_credit_tot
        return {
            'Total Receipts (DEBIT)': num_debit_tot, 'Used Receipts (DEBIT)': num_debit_used,
            '% Used Receipts (DEBIT) (Num)': f"{(num_debit_used / num_debit_tot * 100) if num_debit_tot > 0 else 0:.1f}%",
            '% Covered Receipts (DEBIT) (Vol)': f"{(amt_debit_used / amt_debit_tot * 100) if amt_debit_tot > 0 else 0:.1f}%",
            'Unused Receipts (DEBIT)': num_debit_tot - num_debit_used,
            'Total Deposits (CREDIT)': num_credit_tot, 'Reconciled Deposits (CREDIT)': num_credit_used,
            '% Reconciled Deposits (CREDIT) (Num)': f"{(num_credit_used / num_credit_tot * 100) if num_credit_tot > 0 else 0:.1f}%",
            '% Covered Deposits (CREDIT) (Vol)': f"{(amt_credit_used / amt_credit_tot * 100) if amt_credit_tot > 0 else 0:.1f}%",
            'Unreconciled Deposits (CREDIT)': num_credit_tot - num_credit_used,
            'Final delta (DEBIT - CREDIT)': f"{(unused_debit_amount - unreconciled_credit_amount):,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            'Structural Imbalance (Source)': f"{(structural_imbalance / 100):,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            '_raw_unused_debit_amount': unused_debit_amount, '_raw_unreconciled_credit_amount': unreconciled_credit_amount,
            '_raw_debit_amount_perc': (amt_debit_used / amt_debit_tot * 100) if amt_debit_tot > 0 else 0,
            '_raw_credit_amount_perc': (amt_credit_used / amt_credit_tot * 100) if amt_credit_tot > 0 else 0,
        }

    def _evaluate_best_configuration(self, df, verbose=True):
        if verbose: print("\n🧠 AUTO-EVALUATION: Analyzing data to select the best strategy...")
        strategies = [
            {'name': 'Progressive Balance (Strict)', 'params': {'algorithm': 'progressive_balance', 'sorting_strategy': 'date', 'search_direction': 'past_only'}},
            {'name': 'Subset Sum (Standard)', 'params': {'algorithm': 'subset_sum', 'sorting_strategy': self.sorting_strategy, 'search_direction': self.search_direction}},
            {'name': 'Greedy Amount First', 'params': {'algorithm': 'greedy_amount_first'}}
        ]
        if len(df) < 5000:
             strategies.append({'name': 'Subset Sum (Aggressive)', 'params': {'algorithm': 'subset_sum', 'days_window': max(self.days_window, 30), 'max_combinations': max(self.max_combinations, 12)}})

        best_score, best_params = -1, {}
        for strat in strategies:
            if verbose: print(f"   👉 Testing: {strat['name']}...", end="")
            cfg = { 'tolerance': self.tolerance / 100.0, 'days_window': self.days_window, 'max_combinations': self.max_combinations, 'residual_threshold': self.residual_threshold / 100.0, 'residual_days_window': self.residual_days_window, 'sorting_strategy': self.sorting_strategy, 'search_direction': self.search_direction, 'column_mapping': self.column_mapping, 'use_numba': self.use_numba, 'ignore_tolerance': self.ignore_tolerance, 'enable_best_fit': self.enable_best_fit }
            cfg.update(strat['params'])
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                     sim_engine = ReconciliationEngine(**cfg)
                     stats = sim_engine.run(df.copy(), output_file=None, verbose=False)
                if stats:
                    score = stats.get('_raw_debit_amount_perc', 0) + stats.get('_raw_credit_amount_perc', 0)
                    if strat['params']['algorithm'] == 'progressive_balance' and score > 190: score += 5
                    if verbose: print(f" Score: {score:.2f}")
                    if score > best_score:
                        best_score, best_params = score, strat['params']
                else:
                    if verbose: print(" Failed.")
            except Exception as e:
                if verbose: print(f" Error: {e}")
        if verbose: print(f"   🏆 Selected Strategy: {best_params.get('algorithm')} (Score: {best_score:.2f})")
        return best_params

    def run(self, input_file, output_file=None, verbose=True):
        if not NUMBA_AVAILABLE and verbose:
            print("\n⚠️  WARNING: 'numba' library not found. Running in non-optimized mode (slower).")
            print("   For better performance, install it with: pip install numba\n")
        try:
            self.used_debit_indices, self.used_credit_indices = set(), set()
            if isinstance(input_file, pd.DataFrame):
                if verbose: print("1. Using pre-loaded DataFrame.")
                df = input_file
            else:
                if verbose: print(f"1. Loading and validating file: {input_file}")
                df = self.load_file(input_file)

            tot_debit_orig, tot_credit_orig = df['Debit'].sum(), df['Credit'].sum()
            if verbose: print("2. Separating and sorting DEBIT/CREDIT movements...")
            self._separate_movements(df)
            if verbose: print("3. Starting reconciliation passes...")
            all_matches = []

            if self.algorithm == 'auto':
                best_params = self._evaluate_best_configuration(df, verbose=verbose)
                if best_params:
                    if verbose: print(f"   ⚙️  Applying optimal parameters: {best_params}")
                    for k, v in best_params.items():
                        if hasattr(self, k): setattr(self, k, v)
                    if verbose: print(f"   -> Proceeding with algorithm: {self.algorithm}")

            algorithms_to_run = []
            if self.algorithm == 'all':
                algorithms_to_run = ['progressive_balance', 'subset_sum', 'greedy_amount_first']
            elif self.algorithm in ['progressive_balance', 'subset_sum', 'greedy_amount_first']:
                algorithms_to_run = [self.algorithm]
            else: # default
                algorithms_to_run = ['subset_sum']

            for algo in algorithms_to_run:
                if algo == 'progressive_balance':
                    all_matches.extend(self._reconcile_progressive_balance(verbose=verbose))
                elif algo == 'subset_sum':
                    all_matches.extend(self._reconcile_subset_sum(verbose=verbose))
                elif algo == 'greedy_amount_first':
                    all_matches.extend(self._reconcile_greedy_amount_first(verbose=verbose))

            self.debit_df['used'] = self.debit_df['orig_index'].isin(self.used_debit_indices)
            self.credit_df['used'] = self.credit_df['orig_index'].isin(self.used_credit_indices)
            self.matches_df = self._finalize_matches(all_matches)
            self._verify_total_balance(tot_debit_orig, tot_credit_orig, verbose=verbose)

            structural_diff = tot_debit_orig - tot_credit_orig
            if verbose and abs(structural_diff) > 100:
                 print(f"\n⚖️  INITIAL DATA ANALYSIS: Structural imbalance detected!")
                 print(f"    Total DEBIT (Receipts):    {tot_debit_orig/100:,.2f} €")
                 print(f"    Total CREDIT (Deposits): {tot_credit_orig/100:,.2f} €")
                 print(f"    Difference at source:    {structural_diff/100:,.2f} € (This amount can never be reconciled)")

            self.unused_debit_df = self.debit_df[~self.debit_df['used']].copy()
            self.unreconciled_credit_df = self.credit_df[~self.credit_df['used']].copy()

            if verbose: print("4. Calculating final statistics...")
            stats = self.get_stats()
            if output_file:
                if verbose: print(f"5. Generating Excel report in: {output_file}")
                self.create_excel_report(output_file, df)
                if verbose: print("✓ Excel report created successfully.")
            if verbose: print("\n🎉 Reconciliation completed successfully!")
            return stats
        except (FileNotFoundError, ValueError, IndexError) as e:
            print(f"\n❌ CRITICAL ERROR during processing of '{input_file}': {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

# --- FUNCTION COMPILED WITH NUMBA ---
@jit(nopython=True)
def _numba_find_combination(target, candidates_np, max_combinations, tolerance):
    """Finds an exact combination of candidates that sum to a target amount."""
    stack, n_candidates = [], len(candidates_np)
    for i in range(n_candidates - 1, -1, -1):
        val = candidates_np[i, 0]
        if val <= target + tolerance:
            stack.append((i, val, 1))
    path = np.full(max_combinations, -1, dtype=np.int64)
    while len(stack) > 0:
        idx, current_sum, level = stack.pop()
        path[level-1] = idx
        if abs(target - current_sum) <= tolerance:
             result_indices = np.full(level, 0, dtype=np.int64)
             for k in range(level):
                 result_indices[k] = candidates_np[path[k], 1]
             return result_indices
        if level >= max_combinations: continue
        remaining_slots = max_combinations - level
        if idx + 1 < n_candidates:
            max_add = candidates_np[idx+1, 0] * remaining_slots
            if current_sum + max_add < target - tolerance: continue
        elif current_sum < target - tolerance:
            continue
        for i in range(n_candidates - 1, idx, -1):
            val = candidates_np[i, 0]
            new_sum = current_sum + val
            if new_sum <= target + tolerance:
                 stack.append((i, new_sum, level + 1))
    return np.empty(0, dtype=np.int64)

@jit(nopython=True)
def _numba_find_best_fit_combination(target, candidates_np, max_combinations, tolerance):
    """Finds the best-fitting combination of candidates that maximizes the sum without exceeding the target amount."""
    stack, n_candidates = [], len(candidates_np)
    for i in range(n_candidates - 1, -1, -1):
        val = candidates_np[i, 0]
        if val <= target + tolerance:
            stack.append((i, val, 1))
    path = np.full(max_combinations, -1, dtype=np.int64)
    best_sum, best_path_len = 0, 0
    best_path = np.full(max_combinations, -1, dtype=np.int64)
    min_fill_threshold = target * 0.01
    while len(stack) > 0:
        idx, current_sum, level = stack.pop()
        path[level-1] = idx
        if current_sum > best_sum:
            best_sum, best_path_len = current_sum, level
            for k in range(level): best_path[k] = path[k]
            if abs(target - best_sum) <= tolerance: break
        if level >= max_combinations: continue
        remaining_slots = max_combinations - level
        if idx + 1 < n_candidates:
             max_potential = current_sum + candidates_np[idx+1, 0] * remaining_slots
             if max_potential <= best_sum: continue
        else:
             continue
        for i in range(n_candidates - 1, idx, -1):
            val = candidates_np[i, 0]
            new_sum = current_sum + val
            if new_sum > target + tolerance: continue
            if new_sum + (val * (remaining_slots - 1)) <= best_sum: continue
            stack.append((i, new_sum, level + 1))
    if best_path_len > 0 and best_sum >= min_fill_threshold:
        result_indices = np.full(best_path_len, 0, dtype=np.int64)
        for k in range(best_path_len):
            result_indices[k] = candidates_np[best_path[k], 1]
        return result_indices
    return np.empty(0, dtype=np.int64)

# Aliases
RiconciliatoreContabile = ReconciliationEngine
AccountingReconciler = ReconciliationEngine