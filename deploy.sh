#!/bin/bash

# Script di Deploy Automatico per Riconciliazione Casse
# 1. Esegue i test unitari
# 2. Costruisce l'immagine Docker
# 3. Avvia il container in background

# Configurazione
IMAGE_NAME="riconcilia-casse"
CONTAINER_NAME="riconcilia-casse-container"
PORT=5000

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   DEPLOY AUTOMATICO RICONCILIAZIONE CASSE   ${NC}"
echo -e "${GREEN}=============================================${NC}"

# --- FASE 1: TEST ---
echo -e "\n${YELLOW}[1/3] Esecuzione Test Unitari...${NC}"

# Assicura che lo script dei test sia eseguibile
if [ -f "./run_tests.sh" ]; then
    chmod +x ./run_tests.sh
    ./run_tests.sh
else
    echo -e "${RED}❌ ERRORE: Script 'run_tests.sh' non trovato.${NC}"
    exit 1
fi

# Verifica esito test
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERRORE: I test sono falliti. Deploy interrotto.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Test superati.${NC}"
fi

# --- FASE 2: BUILD DOCKER ---
echo -e "\n${YELLOW}[2/3] Costruzione Immagine Docker...${NC}"

docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERRORE: Build Docker fallita.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Immagine costruita: $IMAGE_NAME${NC}"
fi

# --- FASE 3: RUN CONTAINER ---
echo -e "\n${YELLOW}[3/3] Avvio Container...${NC}"

# Ferma e rimuove container esistente se presente
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "   - Rimozione container esistente..."
    docker rm -f $CONTAINER_NAME > /dev/null
fi

# Avvia il nuovo container
# Mappa le cartelle output e log per persistenza
echo "   - Avvio nuovo container..."
docker run -d \
  -p $PORT:5000 \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/log:/app/log" \
  $IMAGE_NAME > /dev/null

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ DEPLOY COMPLETATO CON SUCCESSO!${NC}"
    echo -e "   - App attiva su: http://localhost:$PORT"
    echo -e "   - Log container: 'docker logs -f $CONTAINER_NAME'"
else
    echo -e "${RED}❌ ERRORE: Impossibile avviare il container.${NC}"
    exit 1
fi