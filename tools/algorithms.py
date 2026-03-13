import itertools
import pandas as pd

def find_simple_match(debit_row, credit_df, tolerance):
    """
    Simple 1-to-1 reconciliation algorithm.
    Finds a single credit transaction that matches a debit transaction
    within the specified tolerance.
    """
    debit_amount = debit_row['Debit']
    
    # Calculate the absolute difference and find the best match
    potential_matches = credit_df.copy()
    potential_matches['diff'] = abs(potential_matches['Credit'] - debit_amount)
    
    best_match = potential_matches.loc[potential_matches['diff'].idxmin()]
    
    if best_match['diff'] <= tolerance:
        # Return a DataFrame for consistency with the other algorithm
        return pd.DataFrame([best_match])
        
    return None

def find_subset_sum_match(debit_row, credit_df, tolerance, max_combinations):
    """
    Greedy Subset Sum algorithm (N-to-1).
    Searches for a combination of credit transactions whose sum matches
    a single debit transaction within the specified tolerance.
    """
    debit_amount = debit_row['Debit']
    
    # Filter credits that are smaller than or equal to the debit amount + tolerance
    # to reduce the search space.
    candidate_credits = credit_df[credit_df['Credit'] <= debit_amount + tolerance].copy()

    # Iterate on the number of items to combine (from 1 to max_combinations)
    for n in range(1, max_combinations + 1):
        # If the number of candidates is less than n, we cannot create combinations
        if len(candidate_credits) < n:
            break
            
        # Generate all combinations of 'n' credits
        for combo_indices in itertools.combinations(candidate_credits.index, n):
            combo_df = candidate_credits.loc[list(combo_indices)]
            combo_sum = combo_df['Credit'].sum()
            
            # Check if the sum of the combination is within the tolerance range
            if abs(combo_sum - debit_amount) <= tolerance:
                # First valid combination found, return it (greedy approach)
                return combo_df
                
    return None

# Dictionary to map configuration strings to algorithm functions
RECONCILIATION_ALGORITHMS = {
    "simple": find_simple_match,
    "subset_sum": find_subset_sum_match
}

def get_algorithm(strategy_name: str):
    """Returns the reconciliation function based on the strategy name."""
    func = RECONCILIATION_ALGORITHMS.get(strategy_name)
    if not func:
        raise ValueError(f"Invalid reconciliation strategy: '{strategy_name}'. Possible values: {list(RECONCILIATION_ALGORITHMS.keys())}")
    return func