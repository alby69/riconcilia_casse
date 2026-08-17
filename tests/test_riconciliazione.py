import unittest
import os
import sys
import pandas as pd
from datetime import datetime

# Add root folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import ReconciliationEngine


class TestReconciliationCore(unittest.TestCase):
    """
    Unit test suite for ReconciliationEngine in core.py.
    """

    def _create_df(self, debits_list, credits_list):
        """Helper to create input DataFrame in the format expected by core.py."""
        rows = []
        for d, amt in debits_list:
            rows.append({'Date': d, 'Debit': amt, 'Credit': 0.0})
        for d, amt in credits_list:
            rows.append({'Date': d, 'Debit': 0.0, 'Credit': amt})
        
        df = pd.DataFrame(rows)
        return df

    def test_exact_1_to_1_match(self):
        """Verifies a simple 1-to-1 match."""
        credits = [(datetime(2025, 1, 10), 100.0)]
        debits = [
            (datetime(2025, 1, 9), 50.0),
            (datetime(2025, 1, 9), 100.0),
            (datetime(2025, 1, 11), 20.0)
        ]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            tolerance=0.0,
            days_window=5,
            search_direction="past_only",
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertEqual(len(matches), 1, "There should be exactly one match.")
        self.assertEqual(matches.iloc[0]['total_credit'], 10000)  # Cents

        # Check used flags
        self.assertTrue(engine.debit_df.loc[engine.debit_df['orig_index'] == 1, 'used'].values[0])
        self.assertTrue(engine.credit_df.loc[engine.credit_df['orig_index'] == 3, 'used'].values[0])

    def test_nearest_date_match(self):
        """Verifies that among multiple exact matches, the candidate closest in date is selected."""
        debits = [(datetime(2025, 1, 1), 100.0)]
        credits = [
            (datetime(2025, 1, 10), 100.0),  # 9 days diff
            (datetime(2025, 1, 2), 100.0),   # 1 day diff (closest)
        ]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            tolerance=0.0,
            days_window=10,
            search_direction="past_only",
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]['credit_indices'], [2])

    def test_2_to_1_combination(self):
        """Verifies a combined 2 DEBITs to 1 CREDIT match."""
        credits = [(datetime(2025, 2, 15), 150.0)]
        debits = [
            (datetime(2025, 2, 14), 100.0),
            (datetime(2025, 2, 14), 50.0),
            (datetime(2025, 2, 16), 150.0)
        ]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            max_combinations=2,
            tolerance=0.0,
            days_window=5,
            search_direction="past_only",
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(matches.iloc[0]['debit_indices']), 2)
        self.assertEqual(matches.iloc[0]['total_credit'], 15000)

    def test_match_with_tolerance(self):
        """Verifies that a match occurs within the defined tolerance."""
        credits = [(datetime(2025, 3, 10), 99.99)]
        debits = [(datetime(2025, 3, 10), 100.00)]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(tolerance=0.02)
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]['difference'], 1)  # 1 cent

    def test_no_match_found(self):
        """Verifies that no match is created if there are no valid candidates."""
        credits = [(datetime(2025, 4, 1), 1000.0)]
        debits = [(datetime(2025, 4, 1), 100.0)]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            tolerance=0.0,
            enable_best_fit=False,
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertTrue(matches.empty)
        self.assertFalse(engine.credit_df['used'].any())

    def test_progressive_balance_operator_defaults(self):
        """Verifies Progressive Balance with POS operator default settings."""
        credits = [(datetime(2025, 5, 5), 300.0)]
        debits = [
            (datetime(2025, 5, 2), 100.0),
            (datetime(2025, 5, 3), 200.0)
        ]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            algorithm="progressive_balance",
            days_window=5,
            tolerance=50.0,
            search_direction="past_only"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]['total_debit'], 30000)
        self.assertEqual(matches.iloc[0]['total_credit'], 30000)

    def test_time_window_compliance(self):
        """Verifies that receipts outside days_window are not matched."""
        credits = [(datetime(2025, 7, 15), 200.0)]
        debits = [(datetime(2025, 7, 1), 200.0)]  # 14 days prior

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            tolerance=0.0,
            days_window=5,
            residual_days_window=5,
            search_direction="past_only",
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertTrue(matches.empty, "No match should be found outside window.")

    def test_max_combinations_limit(self):
        """Verifies combination limits are respected."""
        credits = [(datetime(2025, 8, 10), 60.0)]
        debits = [
            (datetime(2025, 8, 9), 10.0),
            (datetime(2025, 8, 9), 20.0),
            (datetime(2025, 8, 9), 30.0)
        ]

        df = self._create_df(debits, credits)
        engine = ReconciliationEngine(
            tolerance=0.0,
            max_combinations=2,
            enable_best_fit=False,
            algorithm="subset_sum"
        )
        engine.run(df, verbose=False)

        matches = engine.matches_df
        self.assertTrue(matches.empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
