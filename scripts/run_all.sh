#!/bin/bash

# 1. Setup
pip install -r requirements.txt

# 2. Data Prep
python3 src/data_processing/translator_ultimate.py

# 3. Training Sequences (Tuần tự để không nổ VRAM)
# Train Brain
python3 src/training/train_generic.py --config configs/deepseek_70b.yaml

# Train Vision
chmod +x scripts/run_internvl2.sh
./scripts/run_internvl2.sh

# Train Ear
python3 src/training/train_generic.py --config configs/whisper_large.yaml

# Train Mouth
python3 src/training/train_f5_tts.py

echo "🎉🎉🎉 ALL SYSTEMS GO! YOUR DIGITAL HUMAN IS READY."