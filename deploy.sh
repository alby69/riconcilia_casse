#!/bin/bash

# Automated Deployment Script for Accounting Reconciliation
# 1. Runs unit tests
# 2. Builds the Docker image
# 3. Starts the container in the background

# Configuration
IMAGE_NAME="accounting-reconciliation"
CONTAINER_NAME="accounting-reconciliation-app"
PORT=5000

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  AUTOMATED DEPLOYMENT: ACCOUNTING RECONCILIATION  ${NC}"
echo -e "${GREEN}=============================================${NC}"

# --- STEP 1: RUN TESTS ---
echo -e "\n${YELLOW}[1/3] Running Unit Tests...${NC}"

# Ensure the test script is executable
if [ -f "./run_tests.sh" ]; then
    chmod +x ./run_tests.sh
    ./run_tests.sh
else
    echo -e "${RED}❌ ERROR: 'run_tests.sh' script not found.${NC}"
    exit 1
fi

# Check test results
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: Tests failed. Deployment aborted.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Tests passed successfully.${NC}"
fi

# --- STEP 2: BUILD DOCKER IMAGE ---
echo -e "\n${YELLOW}[2/3] Building Docker Image...${NC}"

docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: Docker build failed.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Image built successfully: $IMAGE_NAME${NC}"
fi

# --- STEP 3: RUN CONTAINER ---
echo -e "\n${YELLOW}[3/3] Starting Container...${NC}"

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "   - Removing existing container..."
    docker rm -f $CONTAINER_NAME > /dev/null
fi

# Start the new container
# Map output and log folders for data persistence
echo "   - Starting new container..."
docker run -d \
  -p $PORT:5000 \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/log:/app/log" \
  $IMAGE_NAME > /dev/null

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
    echo -e "   - Application is live at: http://localhost:$PORT"
    echo -e "   - To view logs, run: 'docker logs -f $CONTAINER_NAME'"
else
    echo -e "${RED}❌ ERROR: Failed to start the container.${NC}"
    exit 1
fi