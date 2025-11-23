"""
Core del sistema di riconciliazione.
Contiene la classe RiconciliatoreContabile con tutta la logica di matching.
"""

import pandas as pd
from datetime import timedelta
from itertools import combinations
import math
from tqdm import tqdm

class Movimento:
    """Rappresenta un singolo movimento contabile (incasso o versamento)."""
    def __init__(self, indice, data, importo, tipo):
        self.indice = indice
        self.data = data
        self.importo = importo
        self.tipo = tipo
        self.usato = False

class RiconciliatoreContabile:
    """
    Classe che orchestra la riconciliazione tra incassi e versamenti.
    Utilizza un approccio a due fasi:
    1. Matching principale (esatto e per combinazioni).
    2. Matching dei residui (per piccoli importi non abbinati).
    """
    def __init__(self, incassi, versamenti, tolleranza=0.01, giorni_finestra=10, max_combinazioni=6, residui_config=None):
        self.incassi = sorted(incassi, key=lambda x: x.importo, reverse=True)
        self.versamenti = sorted(versamenti, key=lambda x: x.importo, reverse=True)
        self.tolleranza = tolleranza
        self.giorni_finestra = giorni_finestra
        self.max_combinazioni = max_combinazioni
        self.residui_config = residui_config or {'attiva': False}
        self.risultati = []

    def trova_match_esatto(self, versamento, incassi_disponibili):
        """Cerca un singolo incasso che corrisponda esattamente al versamento."""
        limite_data_inf = versamento.data - timedelta(days=self.giorni_finestra)
        limite_data_sup = versamento.data + timedelta(days=self.giorni_finestra)

        candidati = [
            inc for inc in incassi_disponibili
            if not inc.usato and
               limite_data_inf <= inc.data <= limite_data_sup and
               abs(inc.importo - versamento.importo) <= self.tolleranza
        ]

        if not candidati:
            return None

        # Se ci sono più candidati, sceglie quello più vicino nel tempo
        candidati.sort(key=lambda inc: abs((versamento.data - inc.data).days))
        return [candidati[0]]

    def trova_combinazioni(self, versamento, incassi_disponibili):
        """Cerca una combinazione di incassi che corrisponda al versamento."""
        limite_data_inf = versamento.data - timedelta(days=self.giorni_finestra)
        limite_data_sup = versamento.data + timedelta(days=self.giorni_finestra)

        candidati = [
            inc for inc in incassi_disponibili
            if not inc.usato and
               limite_data_inf <= inc.data <= limite_data_sup and
               inc.importo < versamento.importo + self.tolleranza
        ]

        if not candidati:
            return None

        for n in range(2, min(self.max_combinazioni + 1, len(candidati) + 1)):
            num_combinations = math.comb(len(candidati), n)
            if num_combinations > 10000: continue

            for combo in combinations(candidati, n):
                somma = sum(c.importo for c in combo)
                if abs(somma - versamento.importo) <= self.tolleranza:
                    return list(combo)
        
        return None

    def _registra_match(self, versamento, incassi_abbinati):
        """Marca i movimenti come usati e salva il risultato."""
        versamento.usato = True
        somma_incassi = 0
        for incasso in incassi_abbinati:
            incasso.usato = True
            somma_incassi += incasso.importo

        self.risultati.append({
            'versamento_idx': versamento.indice,
            'versamento_data': versamento.data,
            'versamento_importo': versamento.importo,
            'incassi_indices': [inc.indice for inc in incassi_abbinati],
            'incassi_date': [inc.data for inc in incassi_abbinati],
            'incassi_importi': [inc.importo for inc in incassi_abbinati],
            'somma_incassi': somma_incassi,
            'differenza': versamento.importo - somma_incassi,
            'num_elementi': len(incassi_abbinati)
        })

    def _riconcilia_residui_greedy(self):
        """
        Fase 2: Tenta di riconciliare i versamenti rimanenti usando una combinazione
        "ingorda" di piccoli incassi non ancora utilizzati (residui).
        """
        print("\n🧹 Avvio Fase 2: Riconciliazione dei residui (Greedy)...")
        soglia = self.residui_config.get('soglia_importo', 100)
        
        incassi_residui = sorted(
            [inc for inc in self.incassi if not inc.usato and inc.importo < soglia],
            key=lambda x: x.importo, reverse=True
        )
        versamenti_rimanenti = [v for v in self.versamenti if not v.usato]

        if not incassi_residui or not versamenti_rimanenti:
            print("  - Nessun residuo o versamento rimanente da analizzare.")
            return

        for versamento in tqdm(versamenti_rimanenti, desc="  Analisi residui", unit=" versamento"):
            if versamento.usato: continue

            importo_target = versamento.importo
            somma_corrente = 0
            combinazione_corrente = []

            # Scansione "ingorda" degli incassi residui
            for incasso in incassi_residui:
                if not incasso.usato and (somma_corrente + incasso.importo <= importo_target + self.tolleranza):
                    somma_corrente += incasso.importo
                    combinazione_corrente.append(incasso)
            
            # Verifica se la combinazione trovata è valida
            if abs(somma_corrente - importo_target) <= self.tolleranza and combinazione_corrente:
                self._registra_match(versamento, combinazione_corrente)

    def esegui_riconciliazione(self):
        """Orchestra il processo di riconciliazione in due fasi."""
        # --- FASE 1: RICONCILIAZIONE PRINCIPALE ---
        print("🚀 Avvio Fase 1: Riconciliazione principale...")
        for versamento in tqdm(self.versamenti, desc="  Riconciliazione", unit=" versamento"):
            if versamento.usato:
                continue

            # 1. Prova match esatto 1:1
            match = self.trova_match_esatto(versamento, self.incassi)

            # 2. Prova combinazioni multiple
            if not match:
                match = self.trova_combinazioni(versamento, self.incassi)

            if match:
                self._registra_match(versamento, match)

        # --- FASE 2: RICONCILIAZIONE RESIDUI ---
        if self.residui_config.get('attiva', False):
            self._riconcilia_residui_greedy()
        else:
            print("\nℹ️ Fase 2 (Riconciliazione residui) disattivata dalla configurazione.")

    def salva_risultati(self, file_output):
        """Salva tutti i risultati in un file Excel multi-foglio."""
        with pd.ExcelWriter(file_output, engine='openpyxl') as writer:
            # Foglio 1: Abbinamenti
            df_abbinamenti = pd.DataFrame(self.risultati)
            if not df_abbinamenti.empty:
                df_abbinamenti['incassi_indices'] = df_abbinamenti['incassi_indices'].apply(lambda x: ', '.join(map(str, x)))
                df_abbinamenti['incassi_date'] = df_abbinamenti['incassi_date'].apply(lambda x: ', '.join([d.strftime('%d/%m/%y') for d in x]))
                df_abbinamenti['incassi_importi'] = df_abbinamenti['incassi_importi'].apply(lambda x: ', '.join([f'{i:.2f}' for i in x]))
                df_abbinamenti['versamento_data'] = df_abbinamenti['versamento_data'].dt.strftime('%d/%m/%y')
            df_abbinamenti.to_excel(writer, sheet_name='Abbinamenti', index=False)

            # Foglio 2: Versamenti non riconciliati
            versamenti_non_riconc = [{'Indice': v.indice, 'Data': v.data.strftime('%d/%m/%y'), 'Importo': v.importo} for v in self.versamenti if not v.usato]
            pd.DataFrame(versamenti_non_riconc).to_excel(writer, sheet_name='Versamenti non riconciliati', index=False)

            # Foglio 3: Incassi non utilizzati
            incassi_non_util = [{'Indice': i.indice, 'Data': i.data.strftime('%d/%m/%y'), 'Importo': i.importo} for i in self.incassi if not i.usato]
            pd.DataFrame(incassi_non_util).to_excel(writer, sheet_name='Incassi non utilizzati', index=False)

            # Foglio 4: Statistiche
            num_versamenti = len(self.versamenti)
            num_incassi = len(self.incassi)
            versamenti_riconciliati = sum(1 for v in self.versamenti if v.usato)
            incassi_utilizzati = sum(1 for i in self.incassi if i.usato)
            
            stats = {
                'Metrica': ['Totale Versamenti', 'Totale Incassi', 'Versamenti Riconciliati', 'Incassi Utilizzati', '% Versamenti Riconciliati', '% Incassi Utilizzati'],
                'Valore': [
                    num_versamenti, num_incassi,
                    versamenti_riconciliati, incassi_utilizzati,
                    f"{(versamenti_riconciliati / num_versamenti * 100):.1f}%" if num_versamenti > 0 else "0.0%",
                    f"{(incassi_utilizzati / num_incassi * 100):.1f}%" if num_incassi > 0 else "0.0%"
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistiche', index=False)