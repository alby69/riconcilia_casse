import io
import os
import json
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session, url_for
import uuid
from core import ReconciliationEngine

# --- Flask App Configuration ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_dev' # Change in production

# --- Folder Configuration ---
LOG_FOLDER = 'log'
OUTPUT_FOLDER = 'output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
CONFIG_FILE_PATH = 'config.json'

# Ensure folders exist on startup
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Configuration Helper Functions ---
def load_config():
    """Loads the configuration from config.json."""
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(new_config):
    """Saves the configuration to config.json."""
    # First, load the existing configuration to avoid losing keys not present in the UI
    current_config = load_config()
    # Update the configuration with the new values
    current_config.update(new_config)
    # Save the complete file
    with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_config, f, indent=2, ensure_ascii=False)

def robust_currency_parser(value):
    """Robustly converts a string or number into a standard numeric format for pd.to_numeric."""
    # If it's already a number, it's fine.
    if isinstance(value, (int, float)):
        return value
    # If it's not a string, we can't do anything.
    if not isinstance(value, str):
        return None # Will be converted to NaN
    
    # Clean the string from spaces and euro symbol
    cleaned_str = str(value).strip().replace('€', '').replace(' ', '')
    
    # Case 1: Full Italian format (e.g., "1.234,56")
    # The presence of both separators is a strong indicator.
    if '.' in cleaned_str and ',' in cleaned_str:
        return cleaned_str.replace('.', '').replace(',', '.')
    
    # Case 2: Italian format with only decimals (e.g., "1234,56")
    if ',' in cleaned_str:
        return cleaned_str.replace(',', '.')
        
    # Case 3: Format without commas (e.g., "1234" or "1234.56"). We leave the dot.
    return cleaned_str

# --- Pagina Principale ---
@app.route('/')
def index():
    """Displays the initial page with the upload form."""
    return render_template('index.html')

# --- API per la Configurazione ---
@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the current configuration in JSON format."""
    try:
        return jsonify(load_config())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return jsonify({'error': f"Unable to read config.json: {e}"}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Updates and saves the configuration from a submitted JSON."""
    new_config_data = request.json
    save_config(new_config_data)
    return jsonify({'status': 'success', 'message': 'Configuration saved successfully.'})

# --- Endpoint per l'Elaborazione (modificato per AJAX) ---
@app.route('/processa', methods=['POST'])
def processa_file():
    """Handles the main file processing via an AJAX request.

    This endpoint is the workhorse of the web application. It orchestrates
    the entire reconciliation process based on user input from the web form.

    Workflow:
    1.  **File Upload**: Receives an Excel file from the POST request.
    2.  **Parameter Extraction**: Retrieves all reconciliation parameters
        (tolerance, days_window, etc.) from the submitted form data.
    3.  **Data Preparation**: Reads the Excel file into a pandas DataFrame,
        applies the column name mapping from `config.json`, and standardizes
        the data (parses dates, cleans currency, converts to cents).
    4.  **Engine Execution**: Initializes the `ReconciliationEngine` with the
        user-provided parameters and runs the reconciliation on the DataFrame.
    5.  **Report Generation**: Creates the detailed Excel report and saves it
        to the `OUTPUT_FOLDER` with a unique, non-guessable filename.
    6.  **Session Management**: Stores a mapping between a user-friendly
        download name and the unique physical filename in the user's session
        to facilitate secure downloads.
    7.  **JSON Response**: Returns a JSON object to the frontend containing
        the reconciliation statistics and the secure URL for downloading the
        generated report.

    Returns:
        A Flask JSON response containing either the results or an error message.
    """
    if 'file_input' not in request.files:
        return jsonify({'error': 'No file selected.'}), 400

    file = request.files['file_input']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Unsupported file format. Please upload an Excel file (.xlsx or .xls).'}), 400
    
    try:
        # --- 1. Extract configuration parameters from the form ---
        # Values sent from the form are strings, they need to be converted to the correct type.
        # We extract 'save_log' separately because it's not passed to the RiconciliatoreContabile constructor
        save_log = request.form.get('save_log') == 'true'

        config_params = {
            'tolerance': request.form.get('tolerance', 0.01, type=float),
            'days_window': request.form.get('days_window', 7, type=int),
            'max_combinations': request.form.get('max_combinations', 10, type=int),
            'residual_threshold': request.form.get('residual_threshold', 100.0, type=float),
            'residual_days_window': request.form.get('residual_days_window', 30, type=int),
            'sorting_strategy': request.form.get('sorting_strategy', 'date', type=str),
            'search_direction': request.form.get('search_direction', 'past_only', type=str),
            'algorithm': request.form.get('algorithm', 'auto', type=str),
            'ignore_tolerance': request.form.get('ignore_tolerance') == 'true',
            'enable_best_fit': request.form.get('enable_best_fit') == 'true'
        }

        # --- 2. DataFrame Preparation ---
        # Fix for Python 3.9 SpooledTemporaryFile error: read into BytesIO
        # --- 2. Load and Prepare DataFrame ---
        file.stream.seek(0)
        df_input = pd.read_excel(io.BytesIO(file.stream.read()))

        # CLEANING: Strip whitespace from column names to avoid "DATA " != "DATA" issues
        if not df_input.empty:
            df_input.columns = df_input.columns.str.strip()

        # Apply column mapping (External -> Internal) before accessing data
        # Apply column mapping from config.json
        full_config = load_config()
        # Support both new ('column_mapping') and legacy ('mapping_colonne') keys
        mapping_conf = full_config.get('column_mapping', full_config.get('mapping_colonne', {}))
        
        # Map internal keys to English (Date, Debit, Credit) if they are in Italian
        key_translation = {
            'Data': 'Date',
            'Dare': 'Debit',
            'Avere': 'Credit',
            'Incassi': 'Debit',
            'Versamenti': 'Credit'
        }

        # Invert the mapping: {'Date': 'DATE'} -> {'DATE': 'Date'} to rename correctly
        # And ensure the target column name is the English one expected by core.py
        rename_mapping = {}
        for internal_key, external_col in mapping_conf.items():
            # If the config uses Italian keys (Data, Dare, Avere), translate them to English
            target_key = key_translation.get(internal_key, internal_key)
            rename_mapping[external_col] = target_key

        df_input.rename(columns=rename_mapping, inplace=True)
        
        # Validation: Check if required columns exist
        required_columns = ['Date', 'Debit', 'Credit']
        missing_columns = [col for col in required_columns if col not in df_input.columns]
        if missing_columns:
            found_cols = df_input.columns.tolist()
            print(f"⚠️  Validation Failed. Missing: {missing_columns}")
            print(f"   Columns found in file (after mapping): {found_cols}")
            return jsonify({'error': f"Missing columns after mapping: {', '.join(missing_columns)}. Columns found in file: {found_cols}. Check config.json mapping."}), 400

        df_input['Date'] = pd.to_datetime(df_input['Date'], errors='coerce', dayfirst=True)
        df_input.dropna(subset=['Date'], inplace=True)

        # --- NUOVA LOGICA DI PARSING ROBUSTA ---
        # Apply the parser to each cell, then convert the entire column.
        df_input['Debit'] = pd.to_numeric(df_input['Debit'].apply(robust_currency_parser), errors='coerce')
        df_input['Credit'] = pd.to_numeric(df_input['Credit'].apply(robust_currency_parser), errors='coerce')

        df_input[['Debit', 'Credit']] = df_input[['Debit', 'Credit']].fillna(0)
        df_input['Debit'] = (df_input['Debit'] * 100).round().astype(int)
        df_input['Credit'] = (df_input['Credit'] * 100).round().astype(int)
        df_input['orig_index'] = df_input.index

        # --- 3. Execute reconciliation logic with parameters from the UI ---
        engine = ReconciliationEngine(**config_params)
        stats = engine.run(df_input, output_file=None, verbose=False)

        # --- 4. Save the output file with a unique name ---
        unique_id = uuid.uuid4()
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ('.', '_')).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_output_filename)
        
        engine.create_excel_report(output_filepath, df_input)

        # --- 5. Create and save the log file ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"{timestamp}_{sanitized_filename}_summary.log"
        log_filepath = os.path.join(LOG_FOLDER, log_filename)
        
        formatted_stats = stats.copy()
        for key, value in stats.items():
            if '_raw_importo' in key:
                formatted_stats[key] = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + " €"
            elif '_raw_perc' in key:
                formatted_stats[key] = f"{value:.2f} %".replace('.', ',')

        # Save to disk only if explicitly requested
        if save_log:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                json.dump(formatted_stats, f, indent=4, ensure_ascii=False)
        
        # --- 6. Prepare names and save in session ---
        base_name, extension = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_result{extension}"
        
        # Save the map {pretty_name: unique_name} in the user session
        session['download_map'] = {pretty_download_filename: unique_output_filename}
        
        # --- 7. Return JSON for the frontend ---
        return jsonify({
            'log_content': json.dumps(formatted_stats, indent=4, ensure_ascii=False),
            'download_url': url_for('download_file', filename=pretty_download_filename),
            'version': '3.2'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"A critical error occurred: {str(e)}"}), 500


# --- Nuovo Endpoint per il Download ---
@app.route('/download/<filename>')
def download_file(filename):
    """Handles secure downloading of the generated report file.

    This endpoint serves the Excel report generated by the `/processa`
    route. To prevent users from guessing filenames and accessing reports
    that are not their own, it uses a security mechanism based on the Flask
    session:

    1.  The user-facing `filename` from the URL is not the direct path.
    2.  It looks up this `filename` in a dictionary (`download_map`) stored
        in the user's session.
    3.  This map translates the friendly name to the actual, unique filename
        stored on the server's filesystem.
    4.  If the mapping is found, it serves the file as an attachment for
        download. Otherwise, it returns a 404 error.

    Args:
        filename (str): The user-friendly filename provided in the URL.

    Returns:
        A Flask file response or a 404 error if the file is not found
        in the session map.
    """
    download_map = session.get('download_map', {})
    actual_filename = download_map.get(filename)
    
    if not actual_filename:
        return "File not found or session expired.", 404
        
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 
        actual_filename, 
        as_attachment=True,
        download_name=filename  # FIX: Specify the filename for the download
    )


# --- Avvio del Server ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
