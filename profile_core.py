import cProfile
import pstats
import io
from core import RiconciliatoreContabile

def run_profiling():
    """
    Funzione wrapper che esegue la logica da profilare.
    """
    # --- CONFIGURA QUI ---
    # Specifica il percorso di un file di test. Usa un file di dimensioni
    # realistiche per ottenere risultati significativi.
    file_di_test = 'input/sancesareo_311025.feather' # OTTIMIZZATO: Usa il file Feather
    
    # Istanzia il riconciliatore con parametri standard o ottimizzati
    riconciliatore = RiconciliatoreContabile(
        tolleranza=0.10,
        giorni_finestra=20,
        max_combinazioni=7,
        soglia_residui=20,
        giorni_finestra_residui=30,
        sorting_strategy="date",
        search_direction="future_only"
    )

    # Esegui la funzione che vuoi misurare
    print(f"Avvio profilazione per il file: {file_di_test}...")
    riconciliatore.run(file_di_test, output_file=None, verbose=False)
    print("Profilazione completata.")


if __name__ == "__main__":
    # 1. Crea un oggetto Profiler
    profiler = cProfile.Profile()

    # 2. Esegui la tua funzione sotto il controllo del profiler
    profiler.run('run_profiling()')

    # 3. Stampa le statistiche ordinate per il tempo totale speso in ogni funzione
    print("\n--- Risultati Profilazione (ordinate per Tempo Totale 'tottime') ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(15) # Mostra le 15 funzioni più lente