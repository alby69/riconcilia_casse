import io
import os
import json
import pandas as pd
from datetime import datetime
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    send_from_directory,
    session,
    url_for,
)
import uuid
from core import ReconciliationEngine
from optimizer import find_best_parameters


# --- Helper Functions for Form Parsing ---
def _get_float(value, default):
    """Safely convert form value to float, returning default if empty or None."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _get_int(value, default):
    """Safely convert form value to int, returning default if empty or None."""
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# --- Flask App Configuration ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key_change_in_prod")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# --- Folder Configuration ---
LOG_FOLDER = "log"
OUTPUT_FOLDER = "output"
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
CONFIG_FILE_PATH = "config.json"
PROFILES_FILE_PATH = "profiles.json"


def cleanup_old_files(folder_path, max_age_seconds=86400):
    """Deletes files in folder_path older than max_age_seconds (default 24h)."""
    if not os.path.exists(folder_path):
        return
    now = datetime.now().timestamp()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                if now - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")

# Ensure folders exist on startup
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DEFAULT_PROFILES = {
    "Operatore Punto Vendita (Default)": {
        "algorithm": "progressive_balance",
        "days_window": 5,
        "tolerance": 50.0,
        "search_direction": "past_only",
        "max_combinations": 10,
        "residual_threshold": 50.0,
        "residual_days_window": 5
    },
    "Riconciliazione Mensile Fine Anno": {
        "algorithm": "subset_sum",
        "days_window": 15,
        "tolerance": 1.0,
        "search_direction": "both",
        "max_combinations": 10,
        "residual_threshold": 50.0,
        "residual_days_window": 30
    },
    "Greedy Importi Elevati": {
        "algorithm": "greedy_amount_first",
        "days_window": 15,
        "tolerance": 50.0,
        "search_direction": "both",
        "max_combinations": 10,
        "residual_threshold": 50.0,
        "residual_days_window": 5
    }
}


# --- Helper Functions ---
def load_config():
    """Loads the configuration from config.json."""
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_profiles():
    """Loads profiles from profiles.json or initializes default ones."""
    if not os.path.exists(PROFILES_FILE_PATH):
        save_profiles(DEFAULT_PROFILES)
        return DEFAULT_PROFILES
    try:
        with open(PROFILES_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PROFILES


def save_profiles(profiles_data):
    """Saves profiles to profiles.json."""
    with open(PROFILES_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles_data, f, indent=2, ensure_ascii=False)


def robust_currency_parser(value):
    """Robustly converts a string or number into a standard numeric format."""
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    cleaned_str = str(value).strip().replace("€", "").replace(" ", "")
    if "." in cleaned_str and "," in cleaned_str:
        return cleaned_str.replace(".", "").replace(",", ".")
    if "," in cleaned_str:
        return cleaned_str.replace(",", ".")
    return cleaned_str


def prepare_dataframe(file_stream, column_mapping=None):
    """Reads an Excel file from a stream and prepares the DataFrame for the engine.

    Args:
        file_stream: The uploaded file stream.
        column_mapping (dict, optional): Mapping { "Nome Colonna File": "Nome Interno" }
            built from the form fields. Merged on top of the config mapping; the form
            values take priority. Source columns not present in the file are skipped.
    """
    df = pd.read_excel(io.BytesIO(file_stream.read()))
    df.columns = df.columns.str.strip()

    config = load_config()
    config_mapping = config.get("common", {}).get("column_mapping", {})

    # Merge config mapping with the form mapping (form wins).
    # Skip empty source column names, i.e. mapping fields left blank.
    effective_mapping = dict(config_mapping)
    if column_mapping:
        effective_mapping.update(
            {k: v for k, v in column_mapping.items() if k and str(k).strip()}
        )

    # Apply renaming. df.rename silently ignores source columns not in the file,
    # so unmapped/absent columns are simply skipped.
    df.rename(columns=effective_mapping, inplace=True)

    required_columns = ["Date", "Debit", "Credit"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonne mancanti dopo il mapping: {', '.join(missing)}. "
            f"Colonne trovate: {df.columns.tolist()}. "
            f"Mappatura applicata: {effective_mapping}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df.dropna(subset=["Date"], inplace=True)

    # Parse currency and convert to cents in the final columns
    df["Debit"] = pd.to_numeric(
        df["Debit"].apply(robust_currency_parser), errors="coerce"
    )
    df["Credit"] = pd.to_numeric(
        df["Credit"].apply(robust_currency_parser), errors="coerce"
    )
    df[["Debit", "Credit"]] = df[["Debit", "Credit"]].fillna(0)

    # The engine expects integer cents
    df["Debit"] = (df["Debit"] * 100).round().astype(int)
    df["Credit"] = (df["Credit"] * 100).round().astype(int)

    df["orig_index"] = df.index
    return df


# --- Routes ---
@app.route("/")
def index():
    """Displays the main page, passing the full configuration to the template."""
    config = load_config()
    return render_template("index.html", config=config)


@app.route("/optimize", methods=["POST"])
def optimize_parameters():
    """Analyzes the uploaded file and returns optimal parameters."""
    if "file_input" not in request.files:
        return jsonify({"error": "Nessun file selezionato."}), 400
    file = request.files["file_input"]
    if file.filename == "":
        return jsonify({"error": "Nessun file selezionato."}), 400

    try:
        file.stream.seek(0)
        df = prepare_dataframe(file.stream)
        config = load_config()
        # Usa 'common' come fallback se 'reconciliation_defaults' non esiste
        base_config = config.get("reconciliation_defaults", config.get("common", {}))
        optimizer_config = config.get("optimizer", {})

        # Run optimization
        # Pass both base parameters and optimizer-specific configurations
        # sequential=False abilita il multiprocessing (configurato in modo sicuro in optimizer.py)
        best_params = find_best_parameters(
            df, base_config, optimizer_config, sequential=False
        )

        return jsonify(best_params)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(
            {"error": f"Errore critico durante l'ottimizzazione: {str(e)}"}
        ), 500


@app.route("/processa", methods=["POST"])
def processa_file():
    """Handles the main file processing with user-provided or optimized parameters."""
    if "file_input" not in request.files:
        return jsonify({"error": "Nessun file selezionato."}), 400
    file = request.files["file_input"]
    if file.filename == "":
        return jsonify({"error": "Nessun file selezionato."}), 400

    try:
        form_data = request.form.to_dict()

        # Build column mapping from form inputs
        col_date = form_data.get("col_date", "Data")
        col_debit = form_data.get("col_debit", "Dare")
        col_credit = form_data.get("col_credit", "Avere")
        col_store_id = form_data.get("col_store_id")
        col_valuta_date = form_data.get("col_valuta_date")

        # Only use valuta_date if explicitly provided and not empty
        valuta_date_col = (
            col_valuta_date if col_valuta_date and col_valuta_date.strip() else None
        )

        # Build column mapping dict (source -> internal), skipping empty fields
        column_mapping = {}
        if col_date and col_date.strip():
            column_mapping[col_date] = "Date"
        if col_debit and col_debit.strip():
            column_mapping[col_debit] = "Debit"
        if col_credit and col_credit.strip():
            column_mapping[col_credit] = "Credit"

        config = load_config()
        common_cfg = config.get("common", {})

        engine_params = {
            "tolerance": _get_float(form_data.get("tolerance"), common_cfg.get("tolerance", 50.0)),
            "days_window": _get_int(form_data.get("days_window"), common_cfg.get("days_window", 5)),
            "max_combinations": _get_int(form_data.get("max_combinations"), common_cfg.get("max_combinations", 10)),
            "residual_threshold": _get_float(form_data.get("residual_threshold"), common_cfg.get("residual_threshold", 50.0)),
            "residual_days_window": _get_int(form_data.get("residual_days_window"), common_cfg.get("residual_days_window", 5)),
            "search_direction": form_data.get("search_direction") or common_cfg.get("search_direction", "past_only"),
            "algorithm": form_data.get("algorithm") or common_cfg.get("algorithm", "progressive_balance"),
            "ignore_tolerance": form_data.get("ignore_tolerance") == "true",
            "store_id_column": col_store_id
            if col_store_id and col_store_id.strip()
            else common_cfg.get("store_id_column"),
            "valuta_date_column": valuta_date_col or common_cfg.get("valuta_date_column"),
            "handover_days": common_cfg.get("handover_days", 5),
            "column_mapping": column_mapping,
        }

        file.stream.seek(0)
        df_input = prepare_dataframe(file.stream, column_mapping)

        # Cleanup old output/log files
        cleanup_old_files(app.config["OUTPUT_FOLDER"])
        cleanup_old_files(LOG_FOLDER)

        # The engine receives the dataframe with amounts already in cents
        engine = ReconciliationEngine(**engine_params)
        stats = engine.run(df_input, verbose=False)

        unique_id = uuid.uuid4()
        sanitized_filename = "".join(
            c for c in file.filename if c.isalnum() or c in (".", "_")
        ).rstrip()
        unique_output_filename = f"{unique_id}_{sanitized_filename}"
        output_filepath = os.path.join(
            app.config["OUTPUT_FOLDER"], unique_output_filename
        )

        engine.create_excel_report(output_filepath, df_input)

        base_name, _ = os.path.splitext(sanitized_filename)
        pretty_download_filename = f"{base_name}_result.xlsx"
        session["download_map"] = {pretty_download_filename: unique_output_filename}

        return jsonify(
            {
                "log_content": json.dumps(stats, indent=4, ensure_ascii=False),
                "download_url": url_for(
                    "download_file", filename=pretty_download_filename
                ),
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify(
            {"error": f"Errore critico durante l'elaborazione: {str(e)}"}
        ), 500


@app.route("/download/<filename>")
def download_file(filename):
    """Handles secure downloading of the generated report file."""
    download_map = session.get("download_map", {})
    actual_filename = download_map.get(filename)

    if not actual_filename:
        return "File non trovato o sessione scaduta.", 404

    return send_from_directory(
        app.config["OUTPUT_FOLDER"],
        actual_filename,
        as_attachment=True,
        download_name=filename,
    )


# --- API Routes for Configuration & Profile Management ---
@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Gets or updates the global configuration in config.json."""
    if request.method == "POST":
        new_config = request.get_json()
        if not new_config:
            return jsonify({"error": "Dati configurazione non validi."}), 400
        try:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            return jsonify({"status": "ok", "message": "Configurazione salvata con successo."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(load_config())


@app.route("/api/profiles", methods=["GET", "POST"])
def api_profiles():
    """Gets or saves named configuration profiles."""
    if request.method == "POST":
        data = request.get_json() or {}
        name = data.get("name")
        params = data.get("params")
        if not name or not params:
            return jsonify({"error": "Nome profilo e parametri obbligatori."}), 400
        profiles = load_profiles()
        profiles[name] = params
        save_profiles(profiles)
        return jsonify({"status": "ok", "profiles": profiles})
    return jsonify(load_profiles())


@app.route("/api/profiles/<profile_name>", methods=["DELETE"])
def delete_profile(profile_name):
    """Deletes a saved profile."""
    profiles = load_profiles()
    if profile_name in profiles:
        del profiles[profile_name]
        save_profiles(profiles)
        return jsonify({"status": "ok", "profiles": profiles})
    return jsonify({"error": "Profilo non trovato."}), 404


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host=host, port=port)
