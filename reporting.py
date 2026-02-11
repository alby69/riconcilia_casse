import pandas as pd
from openpyxl.styles import PatternFill, Alignment, Font
from openpyxl.chart import BarChart, Reference

class ExcelReporter:
    """
    Gestisce la generazione dei report Excel per il motore di riconciliazione.
    Separa la logica di presentazione dalla logica di business.
    """

    def __init__(self, engine):
        """Initializes the ExcelReporter.

        Args:
            engine (ReconciliationEngine): An instance of the executed
                reconciliation engine. This engine object holds all the results
                (e.g., matches, unreconciled items) and configuration parameters
                required to build the report.
        """
        self.engine = engine

    def generate_report(self, output_file, original_df):
        """Generates and saves the complete multi-sheet Excel report.

        This is the main public method of the reporter. It orchestrates the
        creation of the entire Excel workbook by calling a series of internal
        methods, each responsible for a specific sheet. The final file provides
        a comprehensive overview of the reconciliation results.

        The generated sheets include:
        - MANUAL: Explains the algorithm and parameters used.
        - Matches: A detailed list of all successful matches.
        - Unused DEBIT / Unreconciled CREDIT: Lists of transactions that were
          not matched.
        - Original: The original input data for reference.
        - Statistics: A summary of reconciliation figures.
        - Monthly Balance: A table and chart showing month-over-month trends.

        Args:
            output_file (str): The path where the Excel file will be saved.
            original_df (pd.DataFrame): The original, unmodified DataFrame that
                was fed into the reconciliation process.
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. Manuale e Parametri
            self._create_manual_sheet(writer)

            # 2. Matches (Abbinamenti)
            self._create_matches_sheet(writer)

            # 3. Non Riconciliati
            self._create_unreconciled_sheets(writer)

            # 4. Dati Originali
            self._create_original_sheet(writer, original_df)

            # 5. Statistiche
            self._create_statistics_sheet(writer)

            # 6. Bilancio Mensile e Grafici
            self._create_monthly_balance_sheet(writer)

    def _create_manual_sheet(self, writer):
        """Creates the 'MANUAL' sheet with algorithm and parameter details.

        This sheet serves as a guide to the report, providing context on how
        the reconciliation was performed. It includes a high-level description
        of the algorithm that was used (e.g., 'Subset Sum' or 'Progressive
        Balance') and a detailed list of all the configuration parameters
        (e.g., tolerance, time window) and their values for the specific run.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
        """
        ws = writer.book.create_sheet("MANUAL", 0)
        title_font = Font(bold=True, size=14)
        header_font = Font(bold=True, size=12)
        
        manual_content = {}
        if self.engine.algorithm == 'subset_sum':
            manual_content = {
                "title": "Algorithm: Subset Sum (Combination Search)",
                "description": [
                    ("General Description", "This algorithm attempts to solve the 'subset sum problem'. For each movement on one side, it searches for a combination of movements on the other side."),
                    ("How it Works", "1. Receipt Aggregation (Many DEBIT -> 1 CREDIT)\n2. Split Deposits (1 DEBIT -> Many CREDIT)\n3. Residual Recovery"),
                ],
                "params": [
                    ("Tolerance", f"{self.engine.tolerance / 100:.2f} €", "Maximum error margin."),
                    ("Time Window", f"{self.engine.days_window} days", "Search interval."),
                    ("Max Combinations", f"{self.engine.max_combinations}", "Max elements combined."),
                ]
            }
        elif self.engine.algorithm == 'progressive_balance':
            manual_content = {
                "title": "Algorithm: Progressive Balance",
                "description": [
                    ("General Description", "Simulates an operator scrolling through lists and closing blocks when totals match."),
                ],
                "params": [
                    ("Tolerance", f"{self.engine.tolerance / 100:.2f} €", "Maximum error margin."),
                ]
            }
            
        common_params = [
             ("Sorting Strategy", self.engine.sorting_strategy, "Sort criterion."),
             ("Search Direction", self.engine.search_direction, "Time direction."),
             ("Numba Optimization", "Enabled" if self.engine.use_numba else "Disabled", "Accelerated calculation."),
             ("Column Mapping", str(self.engine.column_mapping), "Column mapping."),
             ("Force Close on Timeout", "Yes" if self.engine.ignore_tolerance else "No", "Accept non-squared blocks on timeout.")
        ]
        
        if 'params' in manual_content:
            manual_content['params'].extend(common_params)
        else:
            manual_content['params'] = common_params

        row_cursor = 1
        ws.cell(row=row_cursor, column=1, value=manual_content.get('title')).font = title_font
        row_cursor += 2

        for header, text in manual_content.get('description', []):
            ws.cell(row=row_cursor, column=1, value=header).font = header_font
            row_cursor += 1
            cell = ws.cell(row=row_cursor, column=1, value=text)
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=5)
            row_cursor += 2
        
        ws.cell(row=row_cursor, column=1, value="Parameters Used").font = title_font
        row_cursor += 1
        for name, value, desc in manual_content.get('params', []):
            ws.cell(row=row_cursor, column=1, value=name).font = header_font
            ws.cell(row=row_cursor, column=2, value=value)
            ws.cell(row=row_cursor, column=3, value=desc)
            row_cursor += 1

        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 60

    def _create_matches_sheet(self, writer):
        """Creates the 'Matches' sheet with all successful reconciliations.

        This method populates the 'Matches' sheet with a detailed row for each
        successful match found by the engine. It formats the raw data for
        clarity, converting lists of indices and amounts into readable,
        comma-separated strings.

        A key feature of this sheet is the color-coding of rows. Each row is
        highlighted with a different color based on the reconciliation pass
        that identified the match (e.g., green for 'Pass 1', yellow for 'Pass 2'),
        providing an intuitive visual cue about how each match was found.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
        """
        if self.engine.matches_df is None or self.engine.matches_df.empty:
            return

        df = self.engine.matches_df.copy()
        
        # Formatting helpers
        def format_list(data, is_float=False):
            if not isinstance(data, list): return data
            items = [f"{i/100:.2f}".replace('.', ',') for i in data] if is_float else [i + 2 for i in data]
            return ', '.join(map(str, items))

        for col in ['debit_indices', 'credit_indices']: df[col] = df[col].apply(lambda x: format_list(x, is_float=False))
        for col in ['debit_amounts', 'credit_amounts']: df[col] = df[col].apply(lambda x: format_list(x, is_float=True))
        
        df['debit_dates'] = df['debit_dates'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]) if isinstance(x, list) else x.strftime('%d/%m/%y'))
        df['credit_date'] = pd.to_datetime(df['credit_date']).dt.strftime('%d/%m/%y')
        df['total_credit'] = (df['total_credit'] / 100).map('{:,.2f}'.format).str.replace('.', ',')
        df['difference'] = (df['difference'] / 100).map('{:,.2f}'.format).str.replace('.', ',')

        df.to_excel(writer, sheet_name='Matches', index=False)

        # Coloring
        ws = writer.sheets['Matches']
        fill_pass1 = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid") # Green
        fill_pass2 = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # Yellow
        fill_pass3 = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid") # Red
        fill_prog = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid") # Blue

        if 'pass_name' in df.columns:
            for i, row in df.iterrows():
                pass_name = str(row['pass_name'])
                fill = None
                if "Pass 1" in pass_name: fill = fill_pass1
                elif "Pass 2" in pass_name: fill = fill_pass2
                elif "Pass 3" in pass_name: fill = fill_pass3
                elif "Progressive" in pass_name: fill = fill_prog
                
                if fill:
                    for col in range(1, len(df.columns) + 1):
                        ws.cell(row=i + 2, column=col).fill = fill

    def _create_unreconciled_sheets(self, writer):
        """Creates sheets for all unreconciled debit and credit items.

        This method generates two separate sheets:
        - 'Unused DEBIT': Lists all debit transactions that were not part of any
          successful match.
        - 'Unreconciled CREDIT': Lists all credit transactions that were not part
          of any successful match.

        These sheets provide a clear and focused view of all items that remain
        unreconciled, which is essential for manual review and follow-up.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
        """
        if self.engine.unused_debit_df is not None and not self.engine.unused_debit_df.empty:
            df = self.engine.unused_debit_df[['orig_index', 'Date', 'Debit']].copy()
            df['orig_index'] = df['orig_index'] + 2
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%y')
            df['Debit'] = df['Debit'] / 100.0
            df.rename(columns={'orig_index': 'Row Index', 'Debit': 'Amount'}).to_excel(writer, sheet_name='Unused DEBIT', index=False)

        if self.engine.unreconciled_credit_df is not None and not self.engine.unreconciled_credit_df.empty:
            df = self.engine.unreconciled_credit_df[['orig_index', 'Date', 'Credit']].copy()
            df['orig_index'] = df['orig_index'] + 2
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%y')
            df['Credit'] = df['Credit'] / 100.0
            df.rename(columns={'orig_index': 'Row Index', 'Credit': 'Amount'}).to_excel(writer, sheet_name='Unreconciled CREDIT', index=False)

    def _create_original_sheet(self, writer, original_df):
        """Creates the 'Original' sheet with a copy of the input data.

        This method writes the initial, unmodified transaction data to the
        'Original' sheet. The data is sorted by date for consistency.
        This provides a valuable reference for users, allowing them to easily
        compare the reconciliation output against the source transactions.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
            original_df (pd.DataFrame): The original, unmodified DataFrame that
                was used as input for the reconciliation process.
        """
        df = original_df.copy()
        if 'Date' in df.columns: df.sort_values(by=['Date', 'orig_index'], inplace=True)
        if 'Debit' in df.columns: df['Debit'] = df['Debit'] / 100
        if 'Credit' in df.columns: df['Credit'] = df['Credit'] / 100
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
        if 'orig_index' in df.columns: df.drop(columns=['orig_index'], inplace=True)
        df.to_excel(writer, sheet_name='Original', index=False)

    def _create_statistics_sheet(self, writer):
        """Creates the 'Statistics' sheet with a high-level summary.

        This method generates the 'Statistics' sheet, which provides a concise,
        at-a-glance overview of the reconciliation results. It presents the
        summary in two distinct tables: one for Debit (Receipts) and one for
        Credit (Deposits).

        Each table breaks down the total number and value of transactions, the
        portion that was successfully reconciled, and the resulting delta of
        unreconciled items.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
        """
        stats = self.engine.get_stats()
        if not stats: return

        def fmt(v): return f"{v:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
        
        # Helper to create small tables
        def write_table(data, start_row, title):
            df = pd.DataFrame(data, index=['Numero', 'Importo'])
            sheet = writer.sheets['Statistics'] if 'Statistics' in writer.sheets else writer.book.create_sheet('Statistics')
            sheet.cell(row=start_row-1, column=1, value=title).font = Font(bold=True)
            df.to_excel(writer, sheet_name='Statistics', startrow=start_row)

        # Receipts
        receipts_data = {
            'TOT': [stats.get('Total Receipts (DEBIT)'), fmt(self.engine.debit_df['Debit'].sum() / 100)],
            'USED': [stats.get('Used Receipts (DEBIT)'), fmt(self.engine.debit_df[self.engine.debit_df['used']]['Debit'].sum() / 100)],
            'Delta': [stats.get('Unused Receipts (DEBIT)'), fmt(stats.get('_raw_unused_debit_amount', 0))]
        }
        write_table(receipts_data, 2, "Receipts Summary (DEBIT)")

        # Deposits
        deposits_data = {
            'TOT': [stats.get('Total Deposits (CREDIT)'), fmt(self.engine.credit_df['Credit'].sum() / 100)],
            'USED': [stats.get('Reconciled Deposits (CREDIT)'), fmt(self.engine.credit_df[self.engine.credit_df['used']]['Credit'].sum() / 100)],
            'Delta': [stats.get('Unreconciled Deposits (CREDIT)'), fmt(stats.get('_raw_unreconciled_credit_amount', 0))]
        }
        write_table(deposits_data, 8, "Deposits Summary (CREDIT)")

    def _calculate_monthly_balance(self):
        """Calculates monthly aggregate statistics for the balance sheet.

        This helper method processes the full set of transaction data to
        aggregate it by month. For each month, it calculates:
        - Total and reconciled amounts for both Debits and Credits.
        - The "absorbed imbalance" (the net difference within matched blocks).
        - The final unmatched amounts.

        This data forms the basis for the 'Monthly Balance' sheet and its
        accompanying chart.

        Returns:
            pd.DataFrame: A DataFrame indexed by month with columns for each
                calculated statistic.
        """
        if self.engine.debit_df is None or self.engine.credit_df is None: return pd.DataFrame()

        def agg(df, col):
            if df.empty: return pd.DataFrame()
            t = df.copy()
            t['Date'] = pd.to_datetime(t['Date'])
            t['Month'] = t['Date'].dt.to_period('M')
            return pd.DataFrame({
                f'Total {col}': t.groupby('Month')[col].sum(),
                f'Used {col}': t[t['used']].groupby('Month')[col].sum()
            }).fillna(0)

        s_d = agg(self.engine.debit_df, 'Debit')
        s_c = agg(self.engine.credit_df, 'Credit')
        stats = pd.merge(s_d, s_c, left_index=True, right_index=True, how='outer').fillna(0)

        # Absorbed imbalance
        absorbed = pd.DataFrame()
        if self.engine.matches_df is not None and not self.engine.matches_df.empty:
            m = self.engine.matches_df.copy()
            m['Month'] = m['debit_dates'].apply(lambda x: x[0].to_period('M') if isinstance(x, list) and x else None)
            m.dropna(subset=['Month'], inplace=True)
            m['tot_d'] = m['debit_amounts'].apply(sum)
            absorbed = m.groupby('Month').apply(lambda x: (x['tot_d'] - x['total_credit']).sum()).to_frame('Absorbed Imbalance')

        if not absorbed.empty: stats = pd.merge(stats, absorbed, left_index=True, right_index=True, how='outer').fillna(0)
        
        stats['Unmatched DEBIT'] = stats['Total Debit'] - stats['Used Debit']
        stats['Unmatched CREDIT'] = stats['Total Credit'] - stats['Used Credit']
        if 'Absorbed Imbalance' not in stats: stats['Absorbed Imbalance'] = 0
        
        return stats.sort_index().reset_index().astype({'Month': str})

    def _create_monthly_balance_sheet(self, writer):
        """Creates the 'Monthly Balance' sheet with a summary table and chart.

        This method is responsible for visualizing the reconciliation performance
        over time. It calls the `_calculate_monthly_balance` helper to get
        month-over-month statistics.

        It then populates the 'Monthly Balance' sheet with:
        1.  A summary table of the monthly data.
        2.  A stacked column chart that visually represents the unmatched debit
            and credit amounts for each month, making it easy to spot trends
            and periods with significant imbalances.

        Args:
            writer (pd.ExcelWriter): The pandas ExcelWriter object for the report.
        """
        df = self._calculate_monthly_balance()
        if df.empty: return

        # Convert to float for Excel
        for c in df.columns: 
            if c != 'Month': df[c] = df[c] / 100.0
        
        df['Unmatched CREDIT (Chart)'] = -df['Unmatched CREDIT']
        df.to_excel(writer, sheet_name='Monthly Balance', index=False)
        
        # Chart
        ws = writer.sheets['Monthly Balance']
        chart = BarChart()
        chart.type, chart.style, chart.grouping, chart.overlap = "col", 10, "stacked", 100
        chart.title, chart.y_axis.title, chart.x_axis.title = "Monthly Imbalance", "Amount (€)", "Month"
        
        data_cols = [df.columns.get_loc(c)+1 for c in ['Unmatched DEBIT', 'Unmatched CREDIT (Chart)', 'Absorbed Imbalance']]
        for col in data_cols:
            chart.add_data(Reference(ws, min_col=col, min_row=1, max_row=len(df)+1), titles_from_data=True)
        
        chart.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(df)+1))
        ws.add_chart(chart, f"A{len(df)+4}")