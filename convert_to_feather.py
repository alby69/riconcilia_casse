# convert_to_feather.py
import pandas as pd
import os

def convert_excel_to_feather(excel_path, feather_path=None, force_conversion=False):
    """Converts an Excel file to the much faster Feather format."""
    
    # If no feather_path is provided, create one from the excel_path
    if feather_path is None:
        base, _ = os.path.splitext(excel_path)
        feather_path = f"{base}.feather"

    # Skip conversion if the Feather file already exists and is newer, unless forced
    if not force_conversion and os.path.exists(feather_path) and \
       os.path.getmtime(feather_path) > os.path.getmtime(excel_path):
        print(f"Feather file '{feather_path}' is up to date. Skipping conversion.")
        return feather_path

    # Messaggio di lettura rimosso per ridurre la verbosità
    # --- CORREZIONE: Aggiunti i parametri per parsing date e numeri europei ---
    # parse_dates=['Data'] per interpretare la colonna 'Data' come data.
    # dayfirst=True per interpretare 'dd/mm/yyyy' se ambiguo.
    # decimal=',' e thousands='.' per numeri in formato europeo.
    df = pd.read_excel(
        excel_path,
        # Rimuoviamo il parsing delle date da qui, lo facciamo dopo.
        decimal=',',
        thousands='.'
    )

    # --- CORREZIONE: Applica la conversione della data qui ---
    # pd.to_datetime supporta correttamente l'argomento 'dayfirst'.
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Data'], inplace=True) # Rimuove le righe dove la data non è valida

    # Reset index to make it a regular column, which Feather supports.
    df = df.reset_index()
    
    # Messaggio di scrittura rimosso
    df.to_feather(feather_path)
    return feather_path

if __name__ == "__main__":
    # --- CONFIGURE ---
    excel_file = 'input/sancesareo_311025.xlsx'
    # --- RUN ---
    convert_excel_to_feather(excel_file, force_conversion=True)
