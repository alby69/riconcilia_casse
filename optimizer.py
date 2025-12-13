"""
Optimizer - Libreria per la Ricerca Operativa dei Parametri di Riconciliazione.

Questo modulo contiene le funzioni per eseguire simulazioni multiple,
variando i parametri chiave per trovare la combinazione che massimizza
le percentuali di riconciliazione.
Utilizza Optuna per l'ottimizzazione.
"""

import pandas as pd
import json
import sys
import optuna
from tqdm import tqdm

from core import RiconciliatoreContabile


def load_optimizer_config(config_path='config_optimizer.json'):
    """Carica la configurazione dell'ottimizzatore da un file JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        settings = config.get('optimizer_settings', {})
        params = config.get('optimization_params', {})
        return settings, params
    except FileNotFoundError:
        print(f"⚠️  File di configurazione dell'ottimizzatore '{config_path}' non trovato.")
        return {}, {}
    except json.JSONDecodeError as e:
        print(f"❌ ERRORE: Formato JSON non valido in '{config_path}': {e}")
        return {}, {}


def run_simulation(base_config, file_input_df, n_trials=50, show_progress=True, sequential=False):
    """
    Esegue l'ottimizzazione usando Optuna per trovare i parametri migliori.
    
    Args:
        base_config (dict): La configurazione di base da cui partire.
        file_input_df (pd.DataFrame): Il DataFrame già caricato e pre-processato su cui eseguire le simulazioni.
        n_trials (int): Il numero di simulazioni (trial) da eseguire.
        show_progress (bool): Se mostrare una barra di progresso (non implementato per il web).
        sequential (bool): Se forzare l'esecuzione sequenziale (1 solo processo).

    Returns:
        dict: Un dizionario contenente i parametri migliori ('best_params') e il punteggio migliore ('best_score').
    """
    
    _, optimizer_config_ranges = load_optimizer_config()
    if not optimizer_config_ranges:
        raise ValueError("Impossibile caricare i range dei parametri per l'ottimizzazione.")

    def objective(trial):
        """Funzione obiettivo che Optuna cercherà di massimizzare."""
        params = {}
        for param_name, details in optimizer_config_ranges.items():
            if details['type'] == 'numeric':
                if details.get('value_type') == 'float':
                    params[param_name] = trial.suggest_float(param_name, details['min'], details['max'], step=details['step'])
                else:
                    params[param_name] = trial.suggest_int(param_name, details['min'], details['max'], step=details['step'])
            elif details['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, details['values'])

        run_config = base_config.copy()
        run_config.update(params)
        
        expected_params = [
            'tolleranza', 'giorni_finestra', 'max_combinazioni', 
            'soglia_residui', 'giorni_finestra_residui', 
            'sorting_strategy', 'search_direction'
        ]
        riconciliatore_config = {key: run_config[key] for key in expected_params if key in run_config}
        
        # Gestione speciale per la tolleranza e soglia che sono in Euro nel config e in centesimi nel core
        if 'tolleranza' in riconciliatore_config:
            riconciliatore_config['tolleranza'] = int(riconciliatore_config['tolleranza'] * 100)
        if 'soglia_residui' in riconciliatore_config:
             riconciliatore_config['soglia_residui'] = int(riconciliatore_config['soglia_residui'] * 100)


        riconciliatore_sim = RiconciliatoreContabile(**riconciliatore_config)
        
        # Passiamo il DataFrame direttamente per evitare riletture
        stats = riconciliatore_sim.run(file_input_df.copy(), output_file=None, verbose=False)

        if stats:
            perc_dare = stats.get('_raw_perc_dare_importo', 0.0)
            perc_avere = stats.get('_raw_perc_avere_importo', 0.0)
            return perc_dare + perc_avere
        
        return 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    n_jobs = 1 if sequential else -1
        
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress # Mostra la barra di progresso di Optuna
    )

    return {
        "best_params": study.best_params,
        "best_score": study.best_value
    }


def update_config_file(config_path, best_params):
    """Aggiorna il file di configurazione JSON con i parametri migliori trovati."""
    try:
        with open(config_path, 'r+') as f:
            config_data = json.load(f)
            
            # Gestione speciale per i parametri annidati come 'residui'
            residui_params = {}
            params_to_update = best_params.copy()

            for key in list(params_to_update.keys()):
                if key.startswith('residui_'):
                    residui_key = key.replace('residui_', '')
                    residui_params[residui_key] = params_to_update.pop(key)
            
            # Aggiorna i parametri di primo livello
            config_data.update(params_to_update)

            # Aggiorna l'oggetto 'residui' se ci sono parametri per esso
            if residui_params:
                if 'residui' not in config_data:
                    config_data['residui'] = {}
                config_data['residui'].update(residui_params)

            f.seek(0)
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            f.truncate()
        return True
    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento del file di configurazione: {e}")
        return False