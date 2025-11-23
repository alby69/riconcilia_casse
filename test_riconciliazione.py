import unittest
from datetime import datetime, timedelta
from core import Movimento, RiconciliatoreContabile

class TestRiconciliazioneCore(unittest.TestCase):
    """
    Suite di test per verificare la correttezza della logica di riconciliazione
    contenuta nella classe RiconciliatoreContabile.
    """

    def test_match_esatto_1_a_1(self):
        """Verifica un abbinamento semplice 1 a 1."""
        print("\n--- Eseguo test: Match esatto 1 a 1 ---")
        versamenti = [Movimento(0, datetime(2025, 1, 10), 100, 'versamento')]
        incassi = [
            Movimento(1, datetime(2025, 1, 9), 50, 'incasso'),
            Movimento(2, datetime(2025, 1, 10), 100, 'incasso'), # Match esatto
            Movimento(3, datetime(2025, 1, 11), 20, 'incasso')
        ]

        r = RiconciliatoreContabile(incassi, versamenti)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 1, "Dovrebbe esserci un solo risultato.")
        self.assertTrue(versamenti[0].usato, "Il versamento dovrebbe essere marcato come usato.")
        self.assertTrue(incassi[1].usato, "L'incasso corrispondente dovrebbe essere usato.")
        self.assertFalse(incassi[0].usato, "Gli altri incassi non dovrebbero essere usati.")
        self.assertEqual(r.risultati[0]['somma_incassi'], 100)

    def test_combinazione_2_a_1(self):
        """Verifica un abbinamento combinato 2 a 1."""
        print("\n--- Eseguo test: Combinazione 2 a 1 ---")
        versamenti = [Movimento(0, datetime(2025, 2, 15), 150, 'versamento')]
        incassi = [
            Movimento(1, datetime(2025, 2, 14), 100, 'incasso'), # Parte della combinazione
            Movimento(2, datetime(2025, 2, 15), 50, 'incasso'),  # Parte della combinazione
            Movimento(3, datetime(2025, 2, 16), 150, 'incasso')
        ]

        r = RiconciliatoreContabile(incassi, versamenti, max_combinazioni=2)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 1)
        self.assertEqual(r.risultati[0]['num_elementi'], 2)
        self.assertEqual(r.risultati[0]['somma_incassi'], 150)
        self.assertTrue(incassi[0].usato and incassi[1].usato)
        self.assertFalse(incassi[2].usato)

    def test_match_con_tolleranza(self):
        """Verifica che un match avvenga entro la tolleranza definita."""
        print("\n--- Eseguo test: Match con tolleranza ---")
        versamenti = [Movimento(0, datetime(2025, 3, 10), 99.99, 'versamento')]
        incassi = [Movimento(1, datetime(2025, 3, 10), 100.00, 'incasso')]

        r = RiconciliatoreContabile(incassi, versamenti, tolleranza=0.02)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 1)
        self.assertAlmostEqual(r.risultati[0]['differenza'], -0.01, places=2)

    def test_nessun_match_trovato(self):
        """Verifica che nessun match venga creato se non ci sono candidati validi."""
        print("\n--- Eseguo test: Nessun match ---")
        versamenti = [Movimento(0, datetime(2025, 4, 1), 1000, 'versamento')]
        incassi = [Movimento(1, datetime(2025, 4, 1), 100, 'incasso')]

        r = RiconciliatoreContabile(incassi, versamenti)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 0)
        self.assertFalse(versamenti[0].usato)

    def test_scelta_match_piu_vicino_nel_tempo(self):
        """Verifica che venga scelto il match con la data più vicina."""
        print("\n--- Eseguo test: Scelta match più vicino ---")
        versamenti = [Movimento(0, datetime(2025, 5, 15), 50, 'versamento')]
        incassi = [
            Movimento(1, datetime(2025, 5, 1), 50, 'incasso'),   # Lontano
            Movimento(2, datetime(2025, 5, 14), 50, 'incasso'),  # Vicino
            Movimento(3, datetime(2025, 5, 25), 50, 'incasso')  # Lontano
        ]

        r = RiconciliatoreContabile(incassi, versamenti)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 1)
        self.assertEqual(r.risultati[0]['incassi_indices'][0], 2, "Deve scegliere l'incasso con indice 2.")
        self.assertTrue(incassi[1].usato)
        self.assertFalse(incassi[0].usato)

    def test_riconciliazione_residui_greedy(self):
        """Verifica la Fase 2 di riconciliazione con piccoli importi (residui)."""
        print("\n--- Eseguo test: Riconciliazione residui (Greedy) ---")
        versamenti = [Movimento(0, datetime(2025, 6, 20), 125, 'versamento')]
        incassi = [
            Movimento(1, datetime(2025, 6, 1), 80, 'incasso'),
            Movimento(2, datetime(2025, 6, 2), 30, 'incasso'),
            Movimento(3, datetime(2025, 6, 3), 15, 'incasso'),
            Movimento(4, datetime(2025, 6, 4), 500, 'incasso') # Non è un residuo
        ]
        
        residui_config = {"attiva": True, "soglia_importo": 100}
        # La riconciliazione principale fallisce perché 80+30+15 richiede 3 combinazioni
        r = RiconciliatoreContabile(incassi, versamenti, max_combinazioni=2, residui_config=residui_config)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 1, "La fase residui dovrebbe trovare un match.")
        self.assertEqual(r.risultati[0]['num_elementi'], 3)
        self.assertAlmostEqual(r.risultati[0]['somma_incassi'], 125)
        # 80+30+15 = 125. L'algoritmo greedy li prende.
        self.assertTrue(incassi[0].usato and incassi[1].usato and incassi[2].usato)

    def test_rispetto_finestra_temporale(self):
        """Verifica che un incasso fuori finestra non venga considerato."""
        print("\n--- Eseguo test: Rispetto finestra temporale ---")
        versamenti = [Movimento(0, datetime(2025, 7, 15), 200, 'versamento')]
        incassi = [Movimento(1, datetime(2025, 7, 1), 200, 'incasso')] # Fuori finestra

        r = RiconciliatoreContabile(incassi, versamenti, giorni_finestra=10)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 0, "Nessun match dovrebbe essere trovato.")

    def test_limite_max_combinazioni(self):
        """Verifica che l'algoritmo non superi il limite di combinazioni."""
        print("\n--- Eseguo test: Limite max combinazioni ---")
        versamenti = [Movimento(0, datetime(2025, 8, 10), 60, 'versamento')]
        incassi = [
            Movimento(1, datetime(2025, 8, 9), 10, 'incasso'),
            Movimento(2, datetime(2025, 8, 9), 20, 'incasso'),
            Movimento(3, datetime(2025, 8, 9), 30, 'incasso')
        ]

        # Il match 10+20+30=60 richiede 3 combinazioni, ma il limite è 2
        r = RiconciliatoreContabile(incassi, versamenti, max_combinazioni=2)
        r.esegui_riconciliazione()

        self.assertEqual(len(r.risultati), 0, "Il match non deve essere trovato a causa del limite di combinazioni.")

if __name__ == '__main__':
    """
    Esegue la suite di test.
    Puoi lanciare questo script direttamente dal terminale con:
    python test_riconciliazione.py
    """
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRiconciliazioneCore))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)