# Manuale Utente — CashRec

CashRec ti aiuta a **quadrare la cassa**: confronta gli incassi di un punto vendita con i versamenti in banca, li abbina tra loro e ti segnala le differenze che devi verificare.

Questa guida è pensata per chi usa l'applicazione tutti i giorni, senza conoscenze tecniche.

---

## Concetti da conoscere

| Termine | Cosa significa |
|---|---|
| **Incasso (Dare)** | Il denaro entrato in cassa (scontrini, fatture, ricevute). |
| **Versamento (Avere)** | Il denaro versato in banca. |
| **Abbinamento** | L'unione di un versamento con uno o più incassi che lo coprono. |
| **Tolleranza** | La differenza massima (in €) accettata come "uguale": sotto questa cifra il programma considera tutto a posto. |
| **Finestra temporale** | Quanti giorni, intorno alla data del versamento, il programma cerca gli incassi. |
| **Data Valuta** | Il giorno in cui il movimento ha "valore" per la banca. Se presente, è preferita alla data di registrazione. |
| **Saldo Prog.** | Il saldo progressivo di cassa: cassa iniziale + incassi − versamenti. |

## Prima di iniziare

Ti serve un file Excel (o CSV) con i movimenti del punto vendita, che contenga almeno queste colonne:

- **Data** — la data del movimento;
- **Dare** — l'importo dell'incasso;
- **Avere** — l'importo del versamento.

Se il file contiene anche le colonne **Data Valuta** e **Saldo Prog.**, CashRec le usa per un'analisi più precisa.

## Guida passo per passo

1. **Apri** CashRec.
2. **Trascina** il file Excel nell'area di caricamento, oppure clicca per selezionarlo.
3. (Opzionale) Apri le **Impostazioni Avanzate** e controlla che i nomi delle colonne (Data, Dare, Avere, Data Valuta, Codice Negozio) corrispondano a quelli del tuo file.
4. (Opzionale) Modifica i parametri:
   - **Algoritmo**: AUTO è la scelta consigliata.
   - **Direzione ricerca**: *Solo passato* cerca solo incassi precedenti al versamento; *Entrambi* cerca anche quelli successivi.
   - **Tolleranza (€)**: di solito 50 €.
   - **Finestra temporale (giorni)**: di solito 5 giorni.
5. Clicca su **Elabora File**.
6. Al termine, clicca su **Scarica il Report Excel**.

## Il report che ottieni

Il report è un file Excel con più fogli. Ecco come leggerli.

### Summary (Riepilogo)

Mostra in un colpo d'occhio il risultato:

- abbinamenti trovati e importi abbinati;
- incassi **non abbinati** (mai usati) e versamenti **non abbinati** (mai coperti);
- la **differenza** complessiva;
- se è presente la colonna Saldo Prog., anche il **saldo iniziale e finale** di cassa.

### Matches (Abbinamenti)

L'elenco di tutti gli abbinamenti trovati, uno per riga:

- **Transaction ID** (es. `D(2)_A(3)`): indica quali righe del foglio *Original* sono coinvolte. `D(2)` = incasso alla riga 2 del foglio *Original*, `A(3)` = versamento alla riga 3.
- **Debit / Credit**: gli importi coinvolti nell'abbinamento.
- **Difference**: la differenza tra versamento e incassi. `0` significa che è tutto a posto.
- **Uncovered**: quanto del versamento resta scoperto (compare solo quando è maggiore di zero).

### Anomalie

Contiene **solo** gli abbinamenti con una differenza **oltre la tolleranza**: sono i casi da verificare per primi (le righe sono colorate di rosso).

### Original

Tutti i movimenti del file originale, uno per riga, con due colonne aggiuntive:

- **Gruppo**: l'identificativo dell'abbinamento (stesso codice = stesso gruppo);
- **Difference**: la differenza del gruppo.

I colori aiutano a capire la situazione al volo:

- **Verde** = incasso abbinato;
- **Rosso** = incasso non abbinato;
- **Arancione** = versamento abbinato;
- **Blu** = versamento non abbinato;
- **Grigio** = riga di riepilogo del mese.

In fondo a ogni mese c'è una riga **TOTALE MESE**: la somma degli incassi, dei versamenti e la differenza (Dare − Avere) di quel mese.

### Monthly Balance (Quadratura mensile)

Per ogni mese mostra i totali, gli importi usati e la colonna **Monthly Difference (DEBIT - CREDIT)**: se la differenza mensile è vicina a zero, quel mese è quadrato.

### Unused DEBIT / Unreconciled CREDIT

- **Unused DEBIT**: gli incassi che non hanno trovato un versamento;
- **Unreconciled CREDIT**: i versamenti che non sono stati coperti dagli incassi.

## Cosa significa "ANOMALY"

Quando il programma trova un versamento con degli incassi vicini, ma **non riesce a coprirlo del tutto** e la differenza supera la tolleranza, lo segnala come **ANOMALY**.

Esempio:

- versamento in banca di **5.500,00 €**;
- incassi disponibili per **4.694,50 €**;
- scoperto: **805,50 €**, oltre la tolleranza di 50 € → **ANOMALY**.

Possibili cause:

- un incasso registrato in un giorno successivo al versamento (e con la direzione *Solo passato* il programma non può usarlo per coprirlo);
- un movimento dimenticato;
- un errore di registrazione.

**Cosa fare**: controlla il periodo, verifica se manca un incasso o se l'importo versato è corretto, poi correggi la registrazione nel tuo gestionale.

## Consigli pratici

1. Controlla **prima le Anomalie**: sono i punti da chiarire.
2. Usa la **Data Valuta** se il gestionale la gestisce: i versamenti hanno spesso valuta diversa dalla data di registrazione.
3. Se un versamento *del mese precedente* non viene abbinato, può essere normale: i versamenti di fine anno restano scoperti perché gli incassi di dicembre sono stati registrati a gennaio.
4. Con *Solo passato* un incasso successivo al versamento non viene mai usato per coprirlo: se ti serve quella flessibilità, prova *Entrambi*.
5. Se trovi molte differenze, usa il pulsante **Ottimizza Parametri**: l'applicazione calcola da sola i valori migliori per i tuoi dati.

## Glossario minimo

- **Dare / Debit**: incassi (denaro in entrata).
- **Avere / Credit**: versamenti (denaro in banca).
- **Quadrare la cassa**: verificare che gli incassi del punto vendita coincidano con i versamenti in banca.
- **Match / Abbinamento**: un versamento coperto da uno o più incassi.
