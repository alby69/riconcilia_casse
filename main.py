"""
Wrapper Esecutivo per Riconciliazione Contabile.

Questo script ha il solo scopo di:
1. Leggere un file di configurazione JSON.
2. Istanziare la classe RiconciliatoreContabile con i parametri corretti.
3. Lanciare il processo di riconciliazione tramite il metodo run().
4. Stampare le statistiche finali in formato JSON se eseguito in modalità 'silent'.
"""
import argparse
import json
import sys
from core import RiconciliatoreContabile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue la riconciliazione contabile basata su un file di configurazione.")
    parser.add_argument('--config', required=True, help="Percorso del file di configurazione JSON.")
    parser.add_argument('--silent', action='store_true', help="Esegui senza output verboso (usato dal batch).")
    
    args = parser.parse_args()
    
    # Carica la configurazione dal file specificato
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ ERRORE: Impossibile caricare o parsare il file di configurazione '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    input_file = config.get('file_input')
    output_file = config.get('file_output')

    if not input_file or not output_file:
        print("❌ ERRORE: 'file_input' e 'file_output' devono essere specificati nel file di configurazione.", file=sys.stderr)
        sys.exit(1)

    # Istanzia il riconciliatore con i parametri dalla configurazione
    riconciliatore = RiconciliatoreContabile(
        tolleranza=config.get('tolleranza', 0.01),
        giorni_finestra=config.get('giorni_finestra', 30),
        max_combinazioni=config.get('max_combinazioni', 6),
        soglia_residui=config.get('soglia_residui', 100),
        giorni_finestra_residui=config.get('giorni_finestra_residui', 60),
        sorting_strategy=config.get('sorting_strategy', 'date'),
        search_direction=config.get('search_direction', 'both'),
        column_mapping=config.get('column_mapping', None) # AGGIUNTA: Legge la mappatura delle colonne
    )

    # Esegui l'intero processo
    stats = riconciliatore.run(input_file, output_file, verbose=False) # Forza verbose=False

    if args.silent and stats:
        # --- AGGIUNTA: Includi i parametri usati nel report JSON ---
        # Definisci i parametri di interesse che vuoi visualizzare nel riepilogo finale.
        parametri_da_includere = [
            'giorni_finestra', 
            'max_combinazioni', 
            'giorni_finestra_residui', 
            'soglia_residui', 
            'sorting_strategy', 
            'search_direction'
        ]
        # Aggiungi i parametri trovati nel file di configurazione al dizionario delle statistiche.
        stats['parametri_ottimali'] = {key: config.get(key) for key in parametri_da_includere}

        print(json.dumps(stats)) # Stampa le statistiche in JSON per il processo padre
