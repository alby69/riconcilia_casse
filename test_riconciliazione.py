import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent folder to the path to import core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import RiconciliatoreContabile

class TestReconciliationCore(unittest.TestCase):
    """
    Test suite to verify the correctness of the reconciliation logic
    contained in the RiconciliatoreContabile class.
    Updated for core.py v3.0 (Pandas DataFrame).
    """

    def _create_df(self, debits_list, credits_list):
        """Helper to create the input DataFrame in the format expected by core.py."""
        rows = []
        # debits_list: [(date, amount), ...]
        for d, amt in debits_list:
            rows.append({'Data': d, 'Dare': amt, 'Avere': 0})
        # credits_list: [(date, amount), ...]
        for d, amt in credits_list:
            rows.append({'Data': d, 'Dare': 0, 'Avere': amt})
        
        df = pd.DataFrame(rows)
        # Conversion to cents as expected by core.run() if passed as a DataFrame
        df['Dare'] = (df['Dare'] * 100).round().astype(int)
        df['Avere'] = (df['Avere'] * 100).round().astype(int)
        df['indice_orig'] = df.index
        return df

    def test_exact_1_to_1_match(self):
        """Verifies a simple 1-to-1 match."""
        print("\n--- Running test: Exact 1-to-1 Match ---")
        credits = [(datetime(2025, 1, 10), 100)]
        debits = [
            (datetime(2025, 1, 9), 50),
            (datetime(2025, 1, 10), 100), # Exact match
            (datetime(2025, 1, 11), 20)
        ]

        df = self._create_df(debits, credits)
        r = RiconciliatoreContabile(tolleranza=0.0)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "There should be only one result.")
        
        # Verify amounts (in cents)
        self.assertEqual(matches.iloc[0]['somma_avere'], 10000)
        
        # Verify that the correct elements are marked as used
        # Debit of 100 is the second one inserted (original index 1)
        self.assertTrue(r.debit_df.loc[r.debit_df['indice_orig']==1, 'usato'].values)
        # Credit of 100 is the fourth one inserted (original index 3)
        self.assertTrue(r.credit_df.loc[r.credit_df['indice_orig']==3, 'usato'].values)

    def test_2_to_1_combination(self):
        """Verifies a combined 2-to-1 match."""
        print("\n--- Running test: 2-to-1 Combination ---")
        credits = [(datetime(2025, 2, 15), 150)]
        debits = [
            (datetime(2025, 2, 14), 100), # Part of the combination
            (datetime(2025, 2, 15), 50),  # Part of the combination
            (datetime(2025, 2, 16), 150)
        ]

        df = self._create_df(debits, credits)
        r = RiconciliatoreContabile(max_combinazioni=2, tolleranza=0.0)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1)
        # Verify that it took 2 DEBIT elements
        self.assertEqual(len(matches.iloc[0]['dare_indices']), 2)
        self.assertEqual(matches.iloc[0]['somma_avere'], 15000)

    def test_match_with_tolerance(self):
        """Verifies that a match occurs within the defined tolerance."""
        print("\n--- Running test: Match with tolerance ---")
        credits = [(datetime(2025, 3, 10), 99.99)]
        debits = [(datetime(2025, 3, 10), 100.00)]

        df = self._create_df(debits, credits)
        r = RiconciliatoreContabile(tolleranza=0.02)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1)
        # Expected difference: 1 cent
        self.assertEqual(matches.iloc[0]['differenza'], 1)

    def test_no_match_found(self):
        """Verifies that no match is created if there are no valid candidates."""
        print("\n--- Running test: No match ---")
        credits = [(datetime(2025, 4, 1), 1000)]
        debits = [(datetime(2025, 4, 1), 100)]

        df = self._create_df(debits, credits)
        r = RiconciliatoreContabile()
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty)
        # Verify that the credit is not used
        self.assertFalse(r.credit_df['usato'].any())

    def test_greedy_residuals_reconciliation(self):
        """Verifies Phase 2 of reconciliation with small amounts (residuals)."""
        print("\n--- Running test: Residuals Reconciliation (Greedy) ---")
        credits = [(datetime(2025, 6, 20), 125)]
        debits = [
            (datetime(2025, 6, 1), 80),
            (datetime(2025, 6, 2), 30),
            (datetime(2025, 6, 3), 15),
            (datetime(2025, 6, 4), 500) # Not a residual
        ]
        
        df = self._create_df(debits, credits)
        
        # Configure to use residuals (high threshold to include 80, 30, 15)
        # The main reconciliation fails if max_combinazioni=2, but the residuals phase should catch them
        r = RiconciliatoreContabile(
            max_combinazioni=2, 
            soglia_residui=100.0, 
            giorni_finestra_residui=30
        )
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "The residuals phase should find a match.")
        # Verify that it combined 3 elements (80+30+15)
        self.assertEqual(len(matches.iloc[0]['dare_indices']), 3)
        self.assertEqual(matches.iloc[0]['somma_avere'], 12500)

    def test_time_window_compliance(self):
        """Verifies that a debit outside the window is not considered."""
        print("\n--- Running test: Time window compliance ---")
        credits = [(datetime(2025, 7, 15), 200)]
        debits = [(datetime(2025, 7, 1), 200)] # 14 days before

        df = self._create_df(debits, credits)
        # 10-day window, so July 1st is too old for July 15th
        r = RiconciliatoreContabile(giorni_finestra=10, search_direction='past_only')
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "No match should be found.")

    def test_max_combinations_limit(self):
        """Verifies that the algorithm does not exceed the combination limit."""
        print("\n--- Running test: Max combinations limit ---")
        credits = [(datetime(2025, 8, 10), 60)]
        debits = [
            (datetime(2025, 8, 9), 10),
            (datetime(2025, 8, 9), 20),
            (datetime(2025, 8, 9), 30)
        ]

        df = self._create_df(debits, credits)
        # The match 10+20+30=60 requires 3 combinations, but the limit is 2
        r = RiconciliatoreContabile(max_combinazioni=2)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "The match should not be found due to the combination limit.")

if __name__ == '__main__':
    """
    Runs the test suite.
    You can run this script directly from the terminal with:
    python test_reconciliation.py
    """
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestReconciliationCore))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)