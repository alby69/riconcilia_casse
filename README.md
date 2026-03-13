# Accounting Reconciliation Web Service

This project provides a powerful and flexible accounting reconciliation service, accessible via a web interface or as a batch processing script. It allows users to upload financial data, apply sophisticated matching algorithms, and generate detailed reports.

## ✨ Key Features

- **Intuitive Web Interface**: A clean, tab-organized UI for uploading files and customizing processing settings.
- **Multiple Algorithms**: Supports various reconciliation algorithms, including "Subset Sum", "Progressive Balance", and "Greedy Amount First".
- **Smart Residual Recovery**: Automatically recovers differences from forced blocks.
- **Capienza Logic**: Supports GDO-style matching where credit >= debits (anticipi, incassi extra).
- **Multi-Store Support**: Optional store ID column for prioritized matching within the same store.
- **Data Valuta**: Handles year-end transitions where January deposits may refer to December.
- **Flexible Column Mapping**: Map any Excel column names to the internal format via web UI.
- **Dynamic Configuration**: Allows real-time modification of key parameters like tolerance, time windows, and search strategies.
- **Secure In-Memory Processing**: Files are processed in memory to ensure speed and data privacy.
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

### 1. Subset Sum Algorithm

**Philosophy**: "Find combinations that add up to the target"

This is the most powerful algorithm. It runs in **3 passes**:

#### Pass 1: Receipt Aggregation (Many DEBIT → 1 CREDIT)
```
Example:
- CREDIT: €300 deposited on Jan 5
- DEBITs: €100 + €150 + €50 from Jan 1-4
- Result: Match! (100 + 150 + 50 = 300)
```

The algorithm searches for combinations of DEBITs that sum to match a CREDIT within the time window.

#### Pass 2: Split Deposits (1 DEBIT → Many CREDIT)
```
Example:
- DEBIT: €500 from sales on Jan 3
- CREDITs: €200 + €300 deposited on Jan 5 and Jan 6
- Result: Match! (200 + 300 = 500)
```

Inverse of Pass 1: splits one large DEBIT into multiple CREDITs.

#### Pass 3: Residual Recovery (Extended Window)
```
Same as Pass 1 but with a larger time window (default 30 days)
to catch difficult matches that were missed in earlier passes.
```

#### Best Fit (Partial Match)

When exact match is impossible, Best Fit finds the combination that best fills the target without exceeding it:

```
Example:
- CREDIT: €300 deposited on Jan 5
- DEBITs: €100 + €150 + €80 from previous days
- Best Fit: Uses €100 + €150 = €250
- Residual: €50 left unmatched (will be processed in next passes)
```

### 2. Progressive Balance Algorithm

**Philosophy**: "Walk through chronologically like a human would"

This algorithm simulates how a person would reconcile by scanning through sorted transactions:

```
Example (sorted by date):
Jan 1:  DEBIT €100
Jan 2:  DEBIT €200
Jan 3:  CREDIT €150
Jan 4:  DEBIT €50
Jan 5:  CREDIT €200

Step-by-step:
1. Start: cum_debit=0, cum_credit=0
2. Add DEBIT €100 → cum_debit=100
3. Add DEBIT €200 → cum_debit=300
4. Add CREDIT €150 → cum_credit=150, diff=150
5. Add DEBIT €50  → cum_debit=350, diff=200
6. Add CREDIT €200 → cum_credit=350, diff=0 ✅ MATCH!

Block: DEBITs (100+200+50) = CREDITs (150+200) = €350
```

**Key Feature - Forced Blocks**: When the time window expires before balance is reached, the algorithm **always** registers the match (forced) instead of discarding data. This prevents losing information.

### 3. Greedy Amount First Algorithm

**Philosophy**: "Match largest amounts first"

Instead of sorting by date, this algorithm sorts by amount and processes the largest transactions first:

```
Example:
- All DEBITs sorted descending: €500, €300, €200, €100...
- All CREDITs sorted descending: €450, €350, €200...

Process:
1. Try to match €500 (largest DEBIT)
2. Search for CREDITs that sum to €500
3. Continue with next largest...
```

This is useful when large transactions are the most important to reconcile.

---

## 🆕 New Features (v5.0)

### Data Valuta (Value Date)

In GDO (Grande Distribuzione) retail, it's common for the bank deposit to be **greater than** the cash receipts (due to anticipi, extra income, rounding adjustments).

```
Traditional: CREDIT must equal DEBIT ± tolerance
Capienza:     CREDIT can be >= DEBIT (credit has excess)

Example:
- DEBITs (receipts): €100 + €150 = €250
- CREDIT (deposit):  €300
- Result: MATCH with Capienza! (300 - 250 = €50 excess)
```

### Smart Residual Recovery

After main algorithms complete, this new phase tries to recover unmatched differences:

1. Takes all "forced" blocks (from Progressive Balance timeout)
2. Analyzes the difference for each block
3. Searches for unused movements that can compensate the difference
4. Creates new matches to recover residual amounts

```
Example:
- Forced block: DEBITs €500, CREDIT €450, difference €50
- Unused credit: €60 from nearby date
- Recovery: Match the €60 to compensate the €50 difference
```

### Multi-Store Support

Optional store ID column enables:

1. **Priority matching**: First try to match within the same store
2. **Cross-store matching**: If no match found, try other stores
3. **Store-level reporting**: Statistics per store

```
Configuration:
store_id_column: "CodiceNegozio"  # or any column name

Example data:
| Date       | Debit | Credit | CodiceNegozio |
|------------|-------|--------|---------------|
| 2025-01-01 | 100   | 0      | STORE_01      |
| 2025-01-02 | 200   | 0      | STORE_01      |
| 2025-01-03 | 0     | 300    | STORE_01      |  ← Match: 100+200 = 300

| 2025-01-01 | 150   | 0      | STORE_02      |
| 2025-01-03 | 0     | 150    | STORE_02      |  ← Match: same store
```

### Data Valuta (Value Date)

The "Data Valuta" feature handles **year-end transitions** common in GDO retail:

- **Problem**: Deposits made in early January may actually belong to December (or previous year)
- **Solution**: Use "Data Valuta" to specify the actual reference period

```
Example:
| Data Registrazione | Dare | Avere | Data Valuta |
|-------------------|------|-------|-------------|
| 2025-01-02        | 100  | 0     | (none)      |  ← Incasso gen
| 2025-01-02        | 200  | 0     | (none)      |  ← Incasso gen  
| 2025-01-03        | 0    | 300   | 2024-12-31  |  ← Versamento di gen, ma riferito a dic!

Matching Logic:
- CREDIT with valuta Dec 31 → matches only with DEBITs from December
- CREDIT with valuta Jan 1 → matches only with DEBITs from January
- CREDIT without valuta → uses registration Date (backward compatible)
```

### Data Analisi (Analysis Date)

For better chronological ordering, the system creates an internal **Data Analisi** column:

- **Data Analisi = Data Valuta** if present
- **Data Analisi = Data** (registration) if no valuta

This ensures deposits with December valuta are processed BEFORE January transactions, even if registered in January.

---

## ⚙️ Installation

1. **Prerequisites**: Python 3.9+ and Git.

2. **Clone the Repository**:
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd accounting-reconciliation
    ```

3. **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

### Command Line (Single File)

```bash
python main.py --input input/myfile.xlsx --output output/result.xlsx
```

### Web Interface

```bash
docker compose up -d
# Access at http://localhost:5000
```

### Batch Processing

Configure `config.json`:
```json
{
  "tolerance": 0.01,
  "days_window": 7,
  "max_combinations": 10,
  "residual_days_window": 30,
  "store_id_column": "CodiceNegozio",
  "valuta_date_column": "Data Valuta",
  "column_mapping": {
    "Data": "Date",
    "Dare": "Debit", 
    "Avere": "Credit"
  }
}
```

### Column Mapping

The system can map **any column names** from your Excel file to the internal format. This is useful if your files use different naming conventions.

**Via Web Interface:**
In the "Impostazioni Avanzate" section, fill in the column names from your Excel file:

| Field | Description | Default |
|-------|-------------|---------|
| Data | Date column | Data |
| Dare (Incassi) | Receipts (cash in) | Dare |
| Avere (Versamenti) | Deposits (cash out) | Avere |
| Codice Negozio | Store ID (optional) | - |
| Data Valuta | Value date (optional) | - |

**Example:** If your Excel has:
- `DataMovimento` instead of `Data`
- `Incassi` instead of `Dare`
- `Versamenti` instead of `Avere`
- `DataRif` for value date

Just enter these names in the mapping fields!

Then run:
```bash
python batch.py
```

---

## 📊 Output Example

The Excel output contains multiple sheets:

| Sheet | Description |
|-------|-------------|
| **Summary** | KPIs, top unreconciled items, automated analysis |
| **Matches** | All found matches with dates, amounts, types |
| **Unused DEBIT** | Unmatched receipt transactions |
| **Unreconciled CREDIT** | Unmatched deposit transactions |
| **Monthly Balance** | Monthly aggregation with charts |

---

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | 0.01 € | Maximum acceptable difference |
| `days_window` | 7 days | Time window for matching |
| `max_combinations` | 10 | Max elements in a combination |
| `residual_days_window` | 30 days | Extended window for residual recovery |
| `algorithm` | subset_sum | Algorithm to use (subset_sum, progressive_balance, greedy, auto) |
| `search_direction` | past_only | Direction (past_only, future_only, both) |
| `store_id_column` | None | Column name for store identification |
| `enable_best_fit` | True | Enable partial matching |

---

## 🐳 Docker Usage

```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## 📂 Project Structure

```
├── core.py              # ReconciliationEngine (algorithms + logic)
├── reporting.py         # Excel report generation
├── app.py              # Flask web interface
├── main.py             # CLI for single file
├── batch.py            # Batch processing
├── optimizer.py         # Parameter optimization
├── config.json         # Configuration file
├── tests/              # Unit tests
└── tools/              # Utility scripts
```

---

## 📜 Changelog

### v5.0 (March 2026)
- **Data Valuta**: New field for "value date" - handles year-end transitions where January deposits refer to December
- **Data Analisi**: Auto-calculated column that uses valuta_date if present, otherwise registration date
- **Column Mapping**: Full support for custom column names via web interface
- **Smart Filtering**: Deposits with December valuta won't match January receipts (and vice versa)

### v4.0 (March 2026)
- **Smart Residual Recovery**: New phase that recovers differences from forced blocks
- **Capienza Logic**: Support for GDO-style matching (credit >= debits)
- **Forced Blocks**: Progressive Balance now always registers matches on timeout (never loses data)
- **Multi-Store Support**: New `store_id_column` parameter for store-level matching
- **Better Excel Formatting**: Fixed currency/integer formatting for all columns
- **New Match Types**: Added "Forced", "Residual Recovery", "Capienza" indicators

### v3.1.0 (February 2026)
- Optimizer with sorting_strategy
- Docker Gunicorn optimization
- Monthly Performance chart improvements

### v3.0.0
- Complete rewrite with Pandas
- Best Fit logic
- Docker support
