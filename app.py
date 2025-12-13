import io
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory, session
from core import RiconciliatoreContabile

# --- Configurazione dell'App Flask ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_dev' # Cambiare in produzione

# --- Configurazione delle cartelle ---
LOG_FOLDER = 'log'
OUTPUT_FOLDER = os.path.join('output', 'processed_files')
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Assicura che le cartelle esistano all'avvio
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- Pagina Principale ---
@app.route('/')
def index():
    """Mostra la pagina iniziale con il form di upload."""
    return render_template('index.html')


# --- Endpoint per l'Elaborazione (modificato per AJAX) ---
@app.route('/processa', methods=['POST'])
def processa_file():
    """
    Gestisce il caricamento del file, l'elaborazione, salva i risultati
    e restituisce un JSON con il link per il download e il log.
    """
    if 'file_input' not in request.files:
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    file = request.files['file_input']

    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Formato file non supportato. Si prega di caricare un file Excel (.xlsx o .xls).'}), 400
    
    try:
        # --- 1. Preparazione del DataFrame ---
        df_input = pd.read_excel(file.stream)
        df_input['Data'] = pd.to_datetime(df_input['Data'], errors='coerce', dayfirst=True)
        df_input.dropna(subset=['Data'], inplace=True)
        if df_input['Dare'].dtype == 'object':
            df_input['Dare'] = df_input['Dare'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        if df_input['Avere'].dtype == 'object':
            df_input['Avere'] = df_input['Avere'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_input['Dare'] = pd.to_numeric(df_input['Dare'], errors='coerce')
        df_input['Avere'] = pd.to_numeric(df_input['Avere'], errors='coerce')
        df_input[['Dare', 'Avere']] = df_input[['Dare', 'Avere']].fillna(0)
        df_input['Dare'] = (df_input['Dare'] * 100).round().astype(int)
        df_input['Avere'] = (df_input['Avere'] * 100).round().astype(int)
        df_input['indice_orig'] = df_input.index

        # --- 2. Esecuzione della logica di riconciliazione ---
        riconciliatore = RiconciliatoreContabile()
        stats = riconciliatore.run(df_input, output_file=None, verbose=False)

        # --- 3. Salvataggio del file di output con nome unico ---
        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        riconciliatore._crea_report_excel(output_filepath, df_input)

        # --- 4. Creazione e salvataggio del file di log ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{timestamp}_{sanitized_filename}_summary.log"
        log_filepath = os.path.join(LOG_FOLDER, log_filename)
        
        formatted_stats = stats.copy()
        for key, value in stats.items():
            if '_raw_importo' in key:
                formatted_stats[key] = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
            elif '_raw_perc' in key:
                formatted_stats[key] = f"{value:.2f} %".replace('.', ',')

        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_stats, f, indent=4, ensure_ascii=False)
        
        # --- 5. Preparazione dei nomi e salvataggio in sessione ---
        base_name, extension = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_risultato{extension}"
        
        # Salva la mappa {nome_bello: nome_unico} nella sessione utente
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        # --- 6. Restituzione del JSON per il frontend ---
        return jsonify({
            'log_content': json.dumps(formatted_stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename),
            'version': '3.0' # Versione incrementata per il nuovo fix
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Si è verificato un errore critico: {str(e)}"}), 500


# --- Nuovo Endpoint per il Download ---
@app.route('/download/<filename>')
def download_file(filename):
    """
    Serve il file di output usando una mappa salvata in sessione
    per trovare il file fisico con nome univoco.
    """
    download_map = session.get('download_map', {})
    actual_filename = download_map.get(filename)
    
    if not actual_filename:
        return "File non trovato o sessione scaduta.", 404
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        actual_filename, 
        as_attachment=True,
        download_name=filename  # FIX: Specifica il nome del file per il download
    )


# --- Avvio del Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
