import unittest
import sys
import os
import pandas as pd
from datetime import datetime

# Aggiungi la cartella padre al path per importare core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import RiconciliatoreContabile

class TestRiconciliazioneCore(unittest.TestCase):
    """
    Suite di test per verificare la correttezza della logica di riconciliazione
    contenuta nella classe RiconciliatoreContabile.
    Aggiornata per core.py v3.0 (Pandas DataFrame).
    """

    def _create_df(self, incassi_list, versamenti_list):
        """Helper per creare il DataFrame di input nel formato atteso da core.py."""
        rows = []
        # incassi_list: [(date, amount), ...]
        for d, amt in incassi_list:
            rows.append({'Data': d, 'Dare': amt, 'Avere': 0})
        # versamenti_list: [(date, amount), ...]
        for d, amt in versamenti_list:
            rows.append({'Data': d, 'Dare': 0, 'Avere': amt})
        
        df = pd.DataFrame(rows)
        # Conversione in centesimi come atteso da core.run() se passato come DF
        df['Dare'] = (df['Dare'] * 100).round().astype(int)
        df['Avere'] = (df['Avere'] * 100).round().astype(int)
        df['indice_orig'] = df.index
        return df

    def test_match_esatto_1_a_1(self):
        """Verifica un abbinamento semplice 1 a 1."""
        print("\n--- Eseguo test: Match esatto 1 a 1 ---")
        versamenti = [(datetime(2025, 1, 10), 100)]
        incassi = [
            (datetime(2025, 1, 9), 50),
            (datetime(2025, 1, 10), 100), # Match esatto
            (datetime(2025, 1, 11), 20)
        ]

        df = self._create_df(incassi, versamenti)
        r = RiconciliatoreContabile(tolleranza=0.0)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "Dovrebbe esserci un solo risultato.")
        
        # Verifica importi (in centesimi)
        self.assertEqual(matches.iloc[0]['somma_avere'], 10000)
        
        # Verifica che gli elementi corretti siano marcati come usati
        # Incasso da 100 è il secondo inserito (indice 1)
        self.assertTrue(r.dare_df.loc[r.dare_df['indice_orig']==1, 'usato'].values[0])
        # Versamento da 100 è il quarto inserito (indice 3)
        self.assertTrue(r.avere_df.loc[r.avere_df['indice_orig']==3, 'usato'].values[0])

    def test_combinazione_2_a_1(self):
        """Verifica un abbinamento combinato 2 a 1."""
        print("\n--- Eseguo test: Combinazione 2 a 1 ---")
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
        # Verifica che abbia preso 2 elementi DARE
        self.assertEqual(len(matches.iloc[0]['dare_indices']), 2)
        self.assertEqual(matches.iloc[0]['somma_avere'], 15000)

    def test_match_con_tolleranza(self):
        """Verifica che un match avvenga entro la tolleranza definita."""
        print("\n--- Eseguo test: Match con tolleranza ---")
        versamenti = [(datetime(2025, 3, 10), 99.99)]
        incassi = [(datetime(2025, 3, 10), 100.00)]

        df = self._create_df(incassi, versamenti)
        r = RiconciliatoreContabile(tolleranza=0.02)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1)
        # Differenza attesa: 1 centesimo
        self.assertEqual(matches.iloc[0]['differenza'], 1)

    def test_nessun_match_trovato(self):
        """Verifica che nessun match venga creato se non ci sono candidati validi."""
        print("\n--- Eseguo test: Nessun match ---")
        versamenti = [(datetime(2025, 4, 1), 1000)]
        incassi = [(datetime(2025, 4, 1), 100)]

        df = self._create_df(incassi, versamenti)
        # Disabilitiamo il best fit per evitare che il 100 venga abbinato parzialmente al 1000
        r = RiconciliatoreContabile(enable_best_fit=False)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty)
        # Verifica che il versamento non sia usato
        self.assertFalse(r.avere_df['usato'].any())

    def test_riconciliazione_residui_greedy(self):
        """Verifica la Fase 2 di riconciliazione con piccoli importi (residui)."""
        print("\n--- Eseguo test: Riconciliazione residui (Greedy) ---")
        versamenti = [(datetime(2025, 6, 20), 125)]
        incassi = [
            (datetime(2025, 6, 1), 80),
            (datetime(2025, 6, 2), 30),
            (datetime(2025, 6, 3), 15),
            (datetime(2025, 6, 4), 500) # Non è un residuo
        ]
        
        df = self._create_df(incassi, versamenti)
        
        # NOTA: Questo test verifica che una combinazione 3-a-1 venga trovata.
        # Con max_combinazioni=3, il match viene trovato dalla Passata 1.
        r = RiconciliatoreContabile(
            max_combinazioni=3, 
            soglia_residui=100.0, 
            giorni_finestra_residui=30
        )
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertEqual(len(matches), 1, "La fase residui dovrebbe trovare un match.")
        # Verifica che abbia combinato 3 elementi (80+30+15)
        self.assertEqual(len(matches.iloc[0]['dare_indices']), 3)
        self.assertEqual(matches.iloc[0]['somma_avere'], 12500)

    def test_rispetto_finestra_temporale(self):
        """Verifica che un incasso fuori finestra non venga considerato."""
        print("\n--- Eseguo test: Rispetto finestra temporale ---")
        versamenti = [(datetime(2025, 7, 15), 200)]
        incassi = [(datetime(2025, 7, 1), 200)] # 14 giorni prima

        df = self._create_df(incassi, versamenti)
        # Finestra di 10 giorni, quindi il 1° luglio è troppo vecchio per il 15 luglio
        # Impostiamo giorni_finestra_residui uguale per evitare che la Passata 3 trovi il match.
        r = RiconciliatoreContabile(giorni_finestra=10, giorni_finestra_residui=10, search_direction='past_only')
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "Nessun match dovrebbe essere trovato.")

    def test_limite_max_combinazioni(self):
        """Verifica che l'algoritmo non superi il limite di combinazioni."""
        print("\n--- Eseguo test: Limite max combinazioni ---")
        versamenti = [(datetime(2025, 8, 10), 60)]
        incassi = [
            (datetime(2025, 8, 9), 10),
            (datetime(2025, 8, 9), 20),
            (datetime(2025, 8, 9), 30)
        ]

        df = self._create_df(incassi, versamenti)
        # Il match 10+20+30=60 richiede 3 combinazioni, ma il limite è 2
        # Disabilitiamo il best_fit per evitare che trovi un match parziale (es. 20+30)
        r = RiconciliatoreContabile(max_combinazioni=2, enable_best_fit=False)
        r.run(df, verbose=False)

        matches = r.df_abbinamenti
        self.assertTrue(matches.empty, "Il match non deve essere trovato a causa del limite di combinazioni.")

if __name__ == '__main__':
    """
    Esegue la suite di test quando lo script viene lanciato direttamente.
    Questo è utile per il debug individuale del file di test.
    """
    unittest.main(verbosity=2)