# Accounting Reconciliation Web Service (CashRec)

This project provides a powerful and flexible accounting reconciliation service, accessible via a web interface or as a batch processing script. It allows users to upload financial data, apply sophisticated matching algorithms, and generate detailed reports.

## 📖 Documentation

- **[Roadmap](./ROADMAP.md)** - Improvement phases and implementation status
- **[Progressive Balance Algorithm](./docs/PROGRESSIVE_BALANCE.md)** - Detailed explanation of the main algorithm, parameters, and how to interpret results
- **[Developer Manual](./docs/DEVELOPER_MANUAL.md)** - Technical documentation for developers
- **[Git Tutorial](./docs/GIT_TUTORIAL.md)** - Git workflow guide
- **[Cloudflare Tunnel](./docs/CLOUDFLARE_TUNNEL_TUTORIAL.md)** - Deployment guide

## ✨ Key Features

- **Intuitive Web Interface**: Clean UI for uploading files, managing profiles, and customizing processing settings.
- **POS Operator Profile Default**: Pre-configured defaults optimized for POS operators (deposits matching cash receipts 1-5 days prior with `past_only` search direction and 5-day window).
- **Single Source of Truth (SSOT)**: Centralized configuration in `config.json` shared across CLI, Web UI, and batch processing.
- **Profile Management**: Save, load, and delete custom parameter profiles directly from the Web UI or via REST API.
- **Multiple Algorithms**: Supports various reconciliation algorithms, including "Subset Sum", "Progressive Balance", and "Greedy Amount First".
- **Smart Residual Recovery**: Automatically recovers differences from forced blocks.
- **Capienza Logic**: Supports GDO-style matching where credit >= debits (anticipi, incassi extra).
- **Multi-Store Support**: Optional store ID column for prioritized matching within the same store.
- **Data Valuta**: Handles year-end transitions where January deposits may refer to December.
- **Flexible Column Mapping**: Map any Excel column names to the internal format via Web UI.
- **Secure Processing**: Environment-based secret keys, file size limits (50MB), and automatic retention cleanup for generated reports.
- **Detailed Excel Reports**: Multi-sheet output with Summary, Matches, Unreconciled items, and Monthly Balance charts.
- **Batch Processing**: Command-line script (`batch.py`) for automatic multiple file processing.
- **Parameter Optimizer**: Script (`optimizer.py`) to find optimal reconciliation parameters.

---

## 📚 How the Algorithms Work

This section explains each reconciliation algorithm with simple examples.

### Basic Concepts

Before diving into algorithms, let's clarify the terminology:

- **DEBIT (Dare)**: Money received from sales (cash register receipts)
- **CREDIT (Avere)**: Money deposited in bank (versamenti)
- **Match**: An association between one or more DEBIT movements and one or more CREDIT movements
- **Tolerance**: Maximum acceptable difference (in euros) to consider a match valid

### 1. Progressive Balance Algorithm (POS Operator Default)

**Philosophy**: "Walk through chronologically with a backward/past time window"

This algorithm processes CREDITs sequentially, matching them with DEBITs within a time window (5 days in the past by default):

```
Logic:
1. Create Data_Analisi = Data_Valuta (if present) else Data Registrazione
2. Sort by Data_Analisi ascending
3. For each CREDIT:
   - Search for unused DEBITs within 5 days in the past (past_only)
   - If total DEBITs >= CREDIT: create match (using partial if needed)
   - If total DEBITs < CREDIT within tolerance: create match with tolerance
   - If total DEBITs < CREDIT beyond tolerance: create ANOMALY block (not carried forward)
4. Mark used items

Example:
CREDIT €150 on Jan 10 with days_window=5, search_direction=past_only:
- Searches DEBITs from Jan 5 to Jan 10
- Finds DEBIT €100 + €50 = €150 → MATCH ✅
- Finds only DEBIT €80 → ANOMALY €70 (not carried forward)
```

**Key Features**:
- **Past-Only Time Window**: Focuses on cash register receipts occurring 1-5 days before deposit
- **Anomaly Detection**: When CREDIT cannot be matched within tolerance, it's flagged as anomaly (residue NOT carried to next CREDIT)
- **Partial Usage**: When a DEBIT is larger than needed, it's split - the used portion goes to the match, the remainder stays available
- **Data Valuta Support**: Uses Data_Valuta for CREDIT transactions to handle year-end transitions
- **Tolerance Support**: Matches within tolerance are accepted (default: 50€)

### 2. Subset Sum Algorithm

**Philosophy**: "Find combinations that add up to the target"

Runs in **3 passes**:
1. **Receipt Aggregation**: Many DEBITs → 1 CREDIT
2. **Split Deposits**: 1 DEBIT → Many CREDITs
3. **Residual Recovery**: Extended window analysis

### 3. Greedy Amount First Algorithm

**Philosophy**: "Match largest amounts first"

Sorts transactions by amount descending and matches largest movements first.

---

## ⚙️ Installation & Testing

1. **Prerequisites**: Python 3.9+ and Git.

2. **Clone the Repository**:
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd accounting-reconciliation
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run Unit Tests**:
    ```bash
    ./run_tests.sh
    ```

---

## 🚀 Usage

### Command Line (Single File)

```bash
python main.py --config config.json
```

### Web Interface

```bash
python app.py
# Access at http://localhost:5001
```

Or using Docker:
```bash
docker compose up -d --build
# Access at http://localhost:5000
```

### Profile & Configuration Management

Configurations are loaded directly from `config.json`. You can manage named profiles via Web UI or REST API:

- **Get Current Config**: `GET /api/config`
- **Update Config**: `POST /api/config`
- **List Profiles**: `GET /api/profiles`
- **Save Profile**: `POST /api/profiles`
- **Delete Profile**: `DELETE /api/profiles/<profile_name>`

---

## 🔧 Configuration Parameters (Single Source of Truth)

| Parameter | Default (POS Operator) | Description |
|-----------|------------------------|-------------|
| `algorithm` | `progressive_balance` | Reconciliation strategy (`progressive_balance`, `subset_sum`, `greedy_amount_first`, `auto`) |
| `tolerance` | `50.0 €` | Maximum acceptable difference |
| `days_window` | `5 days` | Matching time window |
| `search_direction` | `past_only` | Temporal search direction (`past_only`, `future_only`, `both`) |
| `max_combinations` | `10` | Maximum elements combined in subset sum |
| `residual_threshold` | `50.0 €` | Threshold for residual recovery |
| `residual_days_window` | `5 days` | Extended window for residual recovery |

---

## 📂 Project Structure

```
├── core.py              # ReconciliationEngine (core logic & algorithms)
├── reporting.py         # Excel report generation
├── app.py               # Flask web interface & REST API
├── main.py              # Single file CLI worker
├── batch.py             # Batch processing CLI
├── optimizer.py         # Parameter optimization with Optuna
├── config.json          # Single Source of Truth configuration
├── profiles.json        # Saved configuration profiles
├── tests/               # Modernized unit test suite
├── .github/workflows/   # CI GitHub Actions workflow
└── ROADMAP.md           # Improvement roadmap and status
```

---

## 📜 Changelog

### v5.1 (March 2026)
- **POS Operator Defaults**: Default parameters optimized for store operators (`progressive_balance`, 5-day window, `past_only` direction, 50€ tolerance).
- **Single Source of Truth**: Centralized configuration management in `config.json`.
- **Profile Management**: Save and apply named configuration profiles via Web UI and REST API.
- **Engine Hardening**: Deterministic exact matching by date proximity and side-effect free candidate processing.
- **Security & Hygiene**: Environment-based secret key management, upload file size limits, auto cleanup of generated reports, and CI workflow.
