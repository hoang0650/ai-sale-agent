#!/bin/bash
set -e

# Load ENV
if [ -f .env ]; then export $(grep -v '^#' .env | xargs); fi

chmod +x scripts/*.sh

echo "🏭 PHIL AI FACTORY: STARTING SEQUENCE..."
./scripts/01_prepare_data.sh

echo "🧠 Training Brain..."
./scripts/02_train_brain.sh
python3 -c "import torch; torch.cuda.empty_cache()"

echo "👁️ Training Vision..."
./scripts/03_train_vision.sh
python3 -c "import torch; torch.cuda.empty_cache()"

echo "👂🗣️ Training Senses..."
./scripts/04_train_senses.sh

echo "✅ ALL DONE!"