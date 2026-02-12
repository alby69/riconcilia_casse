import pandas as pd
from openpyxl.styles import PatternFill, Alignment, Font, NamedStyle
from openpyxl.formatting.rule import DataBarRule, ColorScaleRule
from openpyxl.chart import BarChart, Reference, Series

class ExcelReporter:
    """
    Manages the generation of Excel reports for the reconciliation engine.
    Separates presentation logic from business logic.
    """

    def __init__(self, engine):
        self.engine = engine
        self.currency_style = self._create_styles()

    def _create_styles(self):
        """Creates named styles for reuse in the workbook."""
        currency_style = NamedStyle(name='currency_style', number_format='#,##0.00 €')
        return currency_style

    def generate_report(self, output_file, original_df):
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Register styles
            writer.book.add_named_style(self.currency_style)

            self._create_summary_sheet(writer)
            self._create_manual_sheet(writer)
            self._create_matches_sheet(writer)
            self._create_unreconciled_sheets(writer)
            self._create_original_sheet(writer, original_df)
            self._create_monthly_balance_sheet(writer)
            
            # Set Summary as the active sheet
            writer.book.active = writer.book['Summary']

    def _create_summary_sheet(self, writer):
        """Creates the main 'Summary' sheet with KPIs and actionable insights."""
        ws = writer.book.create_sheet("Summary", 0)
        title_font = Font(bold=True, size=16, color="1F4E78")
        header_font = Font(bold=True, size=12, color="1F4E78")
        
        ws.cell(row=1, column=1, value="Reconciliation Summary").font = title_font
        
        # --- KPIs ---
        stats = self.engine.get_stats()
        debit_coverage = stats.get('_raw_debit_amount_perc', 0)
        credit_coverage = stats.get('_raw_credit_amount_perc', 0)
        
        ws.cell(row=3, column=1, value="Key Performance Indicators").font = header_font
        kpis = [
            ("Debit Coverage (Volume)", f"{debit_coverage:.2f}%"),
            ("Credit Coverage (Volume)", f"{credit_coverage:.2f}%"),
            ("Unreconciled Debits", stats.get('Unused Receipts (DEBIT)')),
            ("Unreconciled Credits", stats.get('Unreconciled Deposits (CREDIT)')),
            ("Final Delta", stats.get('Final delta (DEBIT - CREDIT)'))
        ]
        for i, (label, value) in enumerate(kpis, 4):
            ws.cell(row=i, column=1, value=label).font = Font(bold=True)
            ws.cell(row=i, column=2, value=value)

        # --- Textual Summary ---
        ws.cell(row=10, column=1, value="Automated Analysis").font = header_font
        summary_text = f"Reconciliation resulted in {debit_coverage:.1f}% of debit volume and {credit_coverage:.1f}% of credit volume being matched. "
        if debit_coverage > 95 and credit_coverage > 95:
            summary_text += "This is a great result, with very few items left to check."
        elif debit_coverage < 80 or credit_coverage < 80:
            summary_text += "There is a significant amount of unreconciled transactions. Focus on the largest unmatched items listed below."
        else:
            summary_text += "Good result, but some items require manual review."
            
        ws.cell(row=11, column=1, value=summary_text).alignment = Alignment(wrap_text=True)
        ws.merge_cells('A11:E11')

        # --- Top 5 Unreconciled Items ---
        ws.cell(row=14, column=1, value="Top 5 Largest Unreconciled Debits").font = header_font
        if self.engine.unused_debit_df is not None and not self.engine.unused_debit_df.empty:
            top_debits = self.engine.unused_debit_df.nlargest(5, 'Debit')
            ws.cell(row=15, column=1, value="Date").font = Font(bold=True)
            ws.cell(row=15, column=2, value="Amount").font = Font(bold=True)
            for i, row in enumerate(top_debits.itertuples(), 16):
                ws.cell(row=i, column=1, value=row.Date.strftime('%d/%m/%Y'))
                cell = ws.cell(row=i, column=2, value=row.Debit / 100)
                cell.style = 'currency_style'

        ws.cell(row=14, column=4, value="Top 5 Largest Unreconciled Credits").font = header_font
        if self.engine.unreconciled_credit_df is not None and not self.engine.unreconciled_credit_df.empty:
            top_credits = self.engine.unreconciled_credit_df.nlargest(5, 'Credit')
            ws.cell(row=15, column=4, value="Date").font = Font(bold=True)
            ws.cell(row=15, column=5, value="Amount").font = Font(bold=True)
            for i, row in enumerate(top_credits.itertuples(), 16):
                ws.cell(row=i, column=4, value=row.Date.strftime('%d/%m/%Y'))
                cell = ws.cell(row=i, column=5, value=row.Credit / 100)
                cell.style = 'currency_style'

        # Column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 5
        ws.column_dimensions['D'].width = 25
        ws.column_dimensions['E'].width = 15

    def _create_manual_sheet(self, writer):
        ws = writer.book.create_sheet("MANUAL", 1)
        # ... (rest of the logic is similar, just update the content)
        title_font = Font(bold=True, size=14)
        header_font = Font(bold=True, size=12)
        
        algo_map = {
            'subset_sum': {
                "title": "Algorithm: Subset Sum (Combination Search)",
                "description": [("How it Works", "1. Receipt Aggregation (Many DEBIT -> 1 CREDIT)\n2. Split Deposits (1 DEBIT -> Many CREDIT)\n3. Residual Recovery")],
            },
            'progressive_balance': {
                "title": "Algorithm: Progressive Balance (Sequential)",
                "description": [("How it Works", "Simulates an operator scrolling through lists and closing blocks when totals match.")],
            },
            'greedy_amount_first': {
                "title": "Algorithm: Greedy Amount First",
                "description": [("How it Works", "Sorts all transactions by amount and tries to match the largest items first, which is useful for finding key matches quickly.")],
            }
        }
        manual_content = algo_map.get(self.engine.algorithm, {"title": f"Algorithm: {self.engine.algorithm}", "description": []})

        params = [
            ("Tolerance", f"{self.engine.tolerance / 100:.2f} €", "Maximum error margin."),
            ("Time Window", f"{self.engine.days_window} days", "Search interval for matches."),
            ("Max Combinations", f"{self.engine.max_combinations}", "Max elements in a combination (for Subset Sum)."),
            ("Search Direction", self.engine.search_direction, "Time direction for search."),
            ("Force Close on Timeout", "Yes" if self.engine.ignore_tolerance else "No", "For Progressive Balance: accepts non-squared blocks on timeout."),
        ]
        manual_content['params'] = params

        row_cursor = 1
        ws.cell(row=row_cursor, column=1, value=manual_content.get('title')).font = title_font
        row_cursor += 2

        for header, text in manual_content.get('description', []):
            ws.cell(row=row_cursor, column=1, value=header).font = header_font
            row_cursor += 1
            cell = ws.cell(row=row_cursor, column=1, value=text); cell.alignment = Alignment(wrap_text=True, vertical='top')
            ws.merge_cells(start_row=row_cursor, start_column=1, end_row=row_cursor, end_column=5)
            row_cursor += 2
        
        ws.cell(row=row_cursor, column=1, value="Parameters Used").font = title_font
        row_cursor += 1
        for name, value, desc in manual_content.get('params', []):
            ws.cell(row=row_cursor, column=1, value=name).font = header_font
            ws.cell(row=row_cursor, column=2, value=value)
            ws.cell(row=row_cursor, column=3, value=desc)
            row_cursor += 1

        ws.column_dimensions['A'].width = 30; ws.column_dimensions['B'].width = 20; ws.column_dimensions['C'].width = 60
    
    def _create_matches_sheet(self, writer):
        if self.engine.matches_df is None or self.engine.matches_df.empty: return
        df = self.engine.matches_df.copy()
        
        def format_list(data, is_float=False):
            if not isinstance(data, list): return data
            items = [f"{i/100:.2f}".replace('.', ',') if is_float else str(i + 2) for i in data]
            return ', '.join(items)

        df['debit_indices'] = df['debit_indices'].apply(format_list)
        df['credit_indices'] = df['credit_indices'].apply(format_list)
        df['debit_amounts'] = df['debit_amounts'].apply(lambda x: format_list(x, is_float=True))
        df['credit_amounts'] = df['credit_amounts'].apply(lambda x: format_list(x, is_float=True))
        
        df['debit_dates'] = df['debit_dates'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]) if isinstance(x, list) and x else (x.strftime('%d/%m/%y') if pd.notna(x) else ''))
        df['credit_date'] = pd.to_datetime(df['credit_date']).dt.strftime('%d/%m/%y')
        df['total_credit'] = (df['total_credit'] / 100)
        df['difference'] = (df['difference'] / 100)
        
        df.to_excel(writer, sheet_name='Matches', index=False)
        ws = writer.sheets['Matches']

        for c_idx, col_name in enumerate(df.columns, 1):
            if 'credit' in col_name or 'diff' in col_name:
                for r_idx in range(2, len(df) + 2):
                    ws.cell(row=r_idx, column=c_idx).style = self.currency_style

        fills = {
            "Pass 1": PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"),
            "Pass 2": PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"),
            "Pass 3": PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"),
            "Progressive": PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid"),
            "Greedy": PatternFill(start_color="E9D8F5", end_color="E9D8F5", fill_type="solid")
        }
        if 'pass_name' in df.columns:
            for i, row in df.iterrows():
                pass_name = str(row['pass_name'])
                for key, fill in fills.items():
                    if key in pass_name:
                        for col in range(1, len(df.columns) + 1):
                            ws.cell(row=i + 2, column=col).fill = fill
                        break

    def _create_unreconciled_sheets(self, writer):
        # ... identical logic to before, just applying currency style
        if self.engine.unused_debit_df is not None and not self.engine.unused_debit_df.empty:
            df = self.engine.unused_debit_df[['orig_index', 'Date', 'Debit']].copy()
            df.rename(columns={'orig_index': 'Row Index', 'Debit': 'Amount'}, inplace=True)
            df['Row Index'] += 2
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%y')
            df['Amount'] /= 100.0
            df.to_excel(writer, sheet_name='Unused DEBIT', index=False)
            ws = writer.sheets['Unused DEBIT']
            for row in ws.iter_rows(min_row=2, max_row=len(df)+1, min_col=3, max_col=3):
                for cell in row: cell.style = self.currency_style

        if self.engine.unreconciled_credit_df is not None and not self.engine.unreconciled_credit_df.empty:
            df = self.engine.unreconciled_credit_df[['orig_index', 'Date', 'Credit']].copy()
            df.rename(columns={'orig_index': 'Row Index', 'Credit': 'Amount'}, inplace=True)
            df['Row Index'] += 2
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%y')
            df['Amount'] /= 100.0
            df.to_excel(writer, sheet_name='Unreconciled CREDIT', index=False)
            ws = writer.sheets['Unreconciled CREDIT']
            for row in ws.iter_rows(min_row=2, max_row=len(df)+1, min_col=3, max_col=3):
                for cell in row: cell.style = self.currency_style

    def _create_original_sheet(self, writer, original_df):
        df = original_df.copy()
        if 'Date' in df.columns: df.sort_values(by=['Date', 'orig_index'], inplace=True)
        if 'Debit' in df.columns: df['Debit'] = df['Debit'] / 100
        if 'Credit' in df.columns: df['Credit'] = df['Credit'] / 100
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%d/%m/%Y')
        if 'orig_index' in df.columns: df.drop(columns=['orig_index'], inplace=True)
        df.to_excel(writer, sheet_name='Original', index=False)

    def _create_monthly_balance_sheet(self, writer):
        df = self.engine._calculate_monthly_balance()
        if df.empty: return

        for c in df.columns: 
            if c != 'Month': df[c] = df[c] / 100.0
        
        df.to_excel(writer, sheet_name='Monthly Balance', index=False)
        ws = writer.sheets['Monthly Balance']
        
        # Apply currency style
        for row in ws.iter_rows(min_row=2, max_row=len(df)+1, min_col=2):
            for cell in row:
                cell.style = self.currency_style

        # Conditional formatting for Unmatched columns
        red_scale_rule = ColorScaleRule(start_type='min', start_color='FFFFE0', end_type='max', end_color='FF0000')
        unmatched_debit_range = f"D2:D{len(df)+1}"
        unmatched_credit_range = f"G2:G{len(df)+1}"
        ws.conditional_formatting.add(unmatched_debit_range, red_scale_rule)
        ws.conditional_formatting.add(unmatched_credit_range, red_scale_rule)

        # --- New Clustered Bar Chart (Total vs Used) ---
        chart = BarChart()
        chart.type, chart.style, chart.grouping = "col", 10, "clustered"
        chart.title, chart.y_axis.title, chart.x_axis.title = "Monthly Performance (Total vs. Used)", "Amount (€)", "Month"
        
        categories = Reference(ws, min_col=1, min_row=2, max_row=len(df)+1)
        chart.set_categories(categories)
        
        data_cols = {
            'Total Debit': 2, 'Used Debit': 3,
            'Total Credit': 5, 'Used Credit': 6
        }
        for title, col_idx in data_cols.items():
            series_ref = Reference(ws, min_col=col_idx, min_row=1, max_row=len(df)+1)
            series = Series(series_ref, title_from_data=True)
            # Assegna colori diversi per distinguere Totale vs Usato
            if 'Debit' in title:
                if 'Total' in title:
                    series.graphicalProperties.solidFill = "4472C4"  # Dark Blue
                else: # Used Debit
                    series.graphicalProperties.solidFill = "8EB4E3"  # Light Blue
            else:
                if 'Total' in title:
                    series.graphicalProperties.solidFill = "ED7D31"  # Dark Orange
                else: # Used Credit
                    series.graphicalProperties.solidFill = "F7B68A"  # Light Orange
            chart.series.append(series)

        ws.add_chart(chart, f"A{len(df)+4}")