#!/bin/bash
set -e

# Load ENV
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi

REQ_FILE=""
if [ -f "../requirements.txt" ]; then REQ_FILE="../requirements.txt"; fi
if [ -z "$REQ_FILE" ] && [ -f "requirements.txt" ]; then REQ_FILE="requirements.txt"; fi
if [ -n "$REQ_FILE" ]; then python3 -c "import importlib; importlib.import_module('transformers'); importlib.import_module('unsloth')" 2>/dev/null || python3 -m pip install -r "$REQ_FILE"; fi

chmod +x scripts/*.sh

echo "🏭 PHIL AI FACTORY: STARTING SEQUENCE..."
echo "📊 Step 1/4: Preparing data..."
./scripts/01_prepare_data.sh

echo "🧠 Step 2/4: Training Brain..."
./scripts/02_train_brain.sh
python3 -c "import torch; torch.cuda.empty_cache()"

echo "👁️ Step 3/4: Training Vision..."
./scripts/03_train_vision.sh
python3 -c "import torch; torch.cuda.empty_cache()"

echo "👂🗣️ Step 4/4: Training Senses..."
./scripts/04_train_senses.sh

echo "✅ ALL DONE! Training pipeline completed successfully!"
