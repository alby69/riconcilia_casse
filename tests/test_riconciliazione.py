import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent folder to the path to import core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import RiconciliatoreContabile

class TestRiconciliazioneCore(unittest.TestCase):
    """
    Test suite to verify the correctness of the reconciliation logic
    contained in the RiconciliatoreContabile class.
    Updated for core.py v3.0 (Pandas DataFrame).
    """

    def _create_df(self, incassi_list, versamenti_list):
        """Helper to create the input DataFrame in the format expected by core.py."""
        rows = []
        # receipts_list: [(date, amount), ...]
        for d, amt in incassi_list:
            rows.append({'Data': d, 'Dare': amt, 'Avere': 0})
        # deposits_list: [(date, amount), ...]
        for d, amt in versamenti_list:
            rows.append({'Data': d, 'Dare': 0, 'Avere': amt})
        
        df = pd.DataFrame(rows)
        # Conversion to cents as expected by core.run() if passed as a DF
        df['Dare'] = (df['Dare'] * 100).round().astype(int)
        df['Avere'] = (df['Avere'] * 100).round().astype(int)
        df['indice_orig'] = df.index
        return df

    def test_exact_1_to_1_match(self):
        """Verifies a simple 1-to-1 match."""
        print("\n--- Running test: Exact 1-to-1 match ---")
        versamenti = [(datetime(2025, 1, 10), 100)]
        incassi = [
            (datetime(2025, 1, 9), 50),
            (datetime(2025, 1, 10), 100), # Exact match
            (datetime(2025, 1, 11), 20)
        ]

        df = self._create_df(incassi, versamenti)
        r = RiconciliatoreContabile(tolleranza=0.0)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "There should be only one result.")
        
        # Verify amounts (in cents)
        self.assertEqual(matches.iloc[0]['somma_avere'], 10000)
        
        # Verify that the correct elements are marked as used
        # Receipt of 100 is the second one inserted (index 1)
        self.assertTrue(r.dare_df.loc[r.dare_df['indice_orig']==1, 'usato'].values[0])
        # Deposit of 100 is the fourth one inserted (index 3)
        self.assertTrue(r.avere_df.loc[r.avere_df['indice_orig']==3, 'usato'].values[0])

    def test_2_to_1_combination(self):
        """Verifies a combined 2-to-1 match."""
        print("\n--- Running test: 2-to-1 Combination ---")
        versamenti = [(datetime(2025, 2, 15), 150)]
        incassi = [
            (datetime(2025, 2, 14), 100), # Parte della combinazione
            (datetime(2025, 2, 15), 50),  # Parte della combinazione
            (datetime(2025, 2, 16), 150)
        ]

        df = self._create_df(incassi, versamenti)
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
        versamenti = [(datetime(2025, 3, 10), 99.99)]
        incassi = [(datetime(2025, 3, 10), 100.00)]

        df = self._create_df(incassi, versamenti)
        r = RiconciliatoreContabile(tolleranza=0.02)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1)
        # Expected difference: 1 cent
        self.assertEqual(matches.iloc[0]['differenza'], 1)

    def test_no_match_found(self):
        """Verifies that no match is created if there are no valid candidates."""
        print("\n--- Running test: No match ---")
        versamenti = [(datetime(2025, 4, 1), 1000)]
        incassi = [(datetime(2025, 4, 1), 100)]

        df = self._create_df(incassi, versamenti)
        # We disable best fit to prevent the 100 from being partially matched to the 1000
        r = RiconciliatoreContabile(enable_best_fit=False)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty)
        # Verify that the deposit is not used
        self.assertFalse(r.avere_df['usato'].any())

    def test_greedy_residuals_reconciliation(self):
        """Verifies Phase 2 of reconciliation with small amounts (residuals)."""
        print("\n--- Running test: Residuals Reconciliation (Greedy) ---")
        versamenti = [(datetime(2025, 6, 20), 125)]
        incassi = [
            (datetime(2025, 6, 1), 80),
            (datetime(2025, 6, 2), 30),
            (datetime(2025, 6, 3), 15),
            (datetime(2025, 6, 4), 500) # Not a residual
        ]
        
        df = self._create_df(incassi, versamenti)
        
        # NOTE: This test verifies that a 3-to-1 combination is found.
        # With max_combinations=3, the match is found by Pass 1.
        r = RiconciliatoreContabile(
            max_combinazioni=3, 
            soglia_residui=100.0, 
            giorni_finestra_residui=30
        )
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "The residual phase should find a match.")
        # Verify that it has combined 3 elements (80+30+15)
        self.assertEqual(len(matches.iloc[0]['dare_indices']), 3)
        self.assertEqual(matches.iloc[0]['somma_avere'], 12500)

    def test_respects_time_window(self):
        """Verifies that a receipt outside the window is not considered."""
        print("\n--- Running test: Respect time window ---")
        versamenti = [(datetime(2025, 7, 15), 200)]
        incassi = [(datetime(2025, 7, 1), 200)] # 14 giorni prima

        df = self._create_df(incassi, versamenti)
        # 10-day window, so July 1st is too old for July 15th
        # We set giorni_finestra_residui to be the same to prevent Pass 3 from finding the match.
        r = RiconciliatoreContabile(giorni_finestra=10, giorni_finestra_residui=10, search_direction='past_only')
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "Nessun match dovrebbe essere trovato.")

    def test_max_combinations_limit(self):
        """Verifies that the algorithm does not exceed the combination limit."""
        print("\n--- Running test: Max combinations limit ---")
        versamenti = [(datetime(2025, 8, 10), 60)]
        incassi = [
            (datetime(2025, 8, 9), 10),
            (datetime(2025, 8, 9), 20),
            (datetime(2025, 8, 9), 30)
        ]

        df = self._create_df(incassi, versamenti)
        # The match 10+20+30=60 requires 3 combinations, but the limit is 2
        # We disable best_fit to prevent it from finding a partial match (e.g., 20+30)
        r = RiconciliatoreContabile(max_combinazioni=2, enable_best_fit=False)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "The match should not be found due to the combination limit.")

if __name__ == '__main__':
    """
    Esegue la suite di test quando lo script viene lanciato direttamente.
    Questo è utile per il debug individuale del file di test.
    """
    unittest.main(verbosity=2)