# Git Tutorial for the riconcilia_casse project

This quick guide covers the essential Git commands for managing this project.

## 1. Initial Setup (to be done only once)

If you are starting from scratch and want to upload the project to GitHub for the first time.

### 1.1. Initialize a local Git repository
```bash
git init -b main
```

### 1.2. Aggiungi tutti i file al monitoraggio
*(Verranno esclusi i file specificati nel `.gitignore`)*
```bash
git add .
```

### 1.3. Crea la prima "fotografia" del progetto (commit)
```bash
git commit -m "Initial commit: Aggiunto sistema di riconciliazione contabile"
```

### 1.4. Collega il repository locale a quello remoto su GitHub
*Sostituisci `<TUO_USERNAME>` con il tuo nome utente GitHub.*
```bash
git remote add origin https://github.com/<TUO_USERNAME>/riconcilia_casse.git
```

### 1.5. Carica i file su GitHub
```bash
git push -u origin main
```

---

## 2. Flusso di Lavoro Quotidiano

### 2.1. Salvare e caricare le modifiche su GitHub

Dopo aver modificato i file, esegui questi comandi per salvare una nuova "fotografia" e caricarla online.

```bash
# 1. Aggiungi le modifiche al prossimo commit
git add .

# 2. Crea il commit con un messaggio che descrive le modifiche
git commit -m "Descrivi qui le modifiche che hai fatto"

# 3. Carica il commit su GitHub
git push origin main
```

### 2.2. Aggiornare la copia locale con l'ultima versione da GitHub

Quando hai lavorato su un altro PC e vuoi aggiornare la macchina attuale.

#### Metodo A: Aggiornamento Semplice (Uso comune)
Questo comando scarica e integra le modifiche. Usalo se non hai fatto modifiche in locale.
```bash
git pull origin main
```

#### Metodo B: Sostituzione Completa della Copia Locale (Forzare l'aggiornamento)
**Attenzione:** Questo metodo scarta tutte le modifiche locali non salvate su GitHub e rende la tua cartella identica a quella remota.

```bash
# 1. Scarica le informazioni più recenti dal repository remoto
git fetch origin

# 2. Forza la tua copia locale a diventare identica a quella remota
git reset --hard origin/main

# 3. (Opzionale) Rimuovi file e cartelle extra non tracciati da Git
git clean -fd
```