import itertools
import pandas as pd

def find_simple_match(dare_row, avere_df, tolleranza):
    """
    Algoritmo di riconciliazione semplice (1-a-1).
    Cerca un singolo versamento (Avere) che corrisponda all'incasso (Dare)
    entro la tolleranza specificata.
    """
    dare_importo = dare_row['Dare']
    
    # Calcola la differenza assoluta e trova il miglior match
    potential_matches = avere_df.copy()
    potential_matches['diff'] = abs(potential_matches['Avere'] - dare_importo)
    
    best_match = potential_matches.loc[potential_matches['diff'].idxmin()]
    
    if best_match['diff'] <= tolleranza:
        # Restituisce un DataFrame per coerenza con l'altro algoritmo
        return pd.DataFrame([best_match])
        
    return None

def find_greedy_subset_sum_match(dare_row, avere_df, tolleranza, max_combinazioni):
    """
    Algoritmo "Greedy Subset Sum" (N-M).
    Cerca una combinazione di versamenti (Avere) la cui somma corrisponda
    all'incasso (Dare) entro la tolleranza.
    """
    dare_importo = dare_row['Dare']
    
    # Filtra gli Avere che sono più piccoli o uguali al Dare + tolleranza
    # per ridurre lo spazio di ricerca.
    candidate_avere = avere_df[avere_df['Avere'] <= dare_importo + tolleranza].copy()

    # Itera sul numero di elementi da combinare (da 1 a max_combinazioni)
    for n in range(1, max_combinazioni + 1):
        # Se il numero di candidati è inferiore a n, non possiamo creare combinazioni
        if len(candidate_avere) < n:
            break
            
        # Genera tutte le combinazioni di 'n' versamenti
        for combo_indices in itertools.combinations(candidate_avere.index, n):
            combo_df = candidate_avere.loc[list(combo_indices)]
            combo_sum = combo_df['Avere'].sum()
            
            # Controlla se la somma della combinazione è nel range di tolleranza
            if abs(combo_sum - dare_importo) <= tolleranza:
                # Trovata la prima combinazione valida, la restituiamo (approccio greedy)
                return combo_df
                
    return None

# Dizionario per mappare le stringhe di configurazione alle funzioni dell'algoritmo
RECONCILIATION_ALGORITHMS = {
    "simple": find_simple_match,
    "greedy_subset_sum": find_greedy_subset_sum_match
}

def get_reconciliation_function(strategy_name: str):
    """Restituisce la funzione di riconciliazione basata sul nome della strategia."""
    func = RECONCILIATION_ALGORITHMS.get(strategy_name)
    if not func:
        raise ValueError(f"Strategia di riconciliazione non valida: '{strategy_name}'. Valori possibili: {list(RECONCILIATION_ALGORITHMS.keys())}")
    return func