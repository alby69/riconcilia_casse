"""
Accounting Reconciliation System

Main script for batch processing of one or more accounting files.

Logic:
1. Load configuration from `config.json`.
2. Find files to process in the input folder.
3. For each file:
    a. Load data (Excel or CSV).
    b. Apply column mapping defined in `config.json`.
    c. Run the reconciliation algorithm via the `core` module.
    d. Save the results to a detailed Excel file in the output folder.
4. Print a final summary.
"""

from pathlib import Path
import json
import sys
from tqdm import tqdm
from core import ReconciliationEngine  # Assuming core.py is refactored

def load_config():
    """Loads configuration from config.json or uses default values."""
    config_file = Path('config.json')
    default_config = {
        "tolerance": 0.01,
        "days_window": 10,
        "input_folder": "input",
        "output_folder": "output",
        "file_patterns": ["*.xlsx", "*.csv"],
        "column_mapping": {
            "Date": "date",
            "Debit": "debit",
            "Credit": "credit"
        },
        "algorithm": {
            "name": "subset_sum",
            "max_combinations": 6
        },
        "residuals": {
            "enabled": True,
            "amount_threshold": 100,
            "days_window": 90
        }
    }
    
    if not config_file.exists():
        print(f"⚠️  '{config_file}' not found. Using default configuration.")
        return default_config
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            print(f"📄 Loading configuration from '{config_file}'...")
            user_config = json.load(f)
            default_config.update(user_config)
            print("✓ Configuration loaded successfully.")
            return default_config
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format in '{config_file}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function"""
    config = load_config()
    input_folder = Path(config['input_folder'])
    output_folder = Path(config['output_folder'])
    output_folder.mkdir(exist_ok=True)

    patterns = config.get('file_patterns', ['*.xlsx', '*.csv'])
    files_to_process = []
    for p in patterns:
        files_to_process.extend(input_folder.glob(p))
    
    if not files_to_process:
        print(f"⚠️ No files found in '{input_folder}' with patterns: {patterns}")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for file_path in tqdm(files_to_process, desc="Processing files"):
        print(f"\n{'='*20} Processing: {file_path.name} {'='*20}")
        try:
            engine = ReconciliationEngine(
                tolerance=config['tolerance'],
                days_window=config['days_window'],
                max_combinations=config['algorithm']['max_combinations'],
                residual_threshold=config['residuals']['amount_threshold'],
                residual_days_window=config['residuals']['days_window'],
                column_mapping=config.get('column_mapping')
            )

            output_file = output_folder / f"result_{file_path.stem}.xlsx"
            
            engine.run(str(file_path), str(output_file))

        except (IOError, ValueError, FileNotFoundError) as e:
            print(f"❌ ERROR while processing {file_path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
