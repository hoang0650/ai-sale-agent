#!/bin/bash
# Script train InternVL2-76B trên H200

# Kiểm tra xem đã có dataset chưa
if [ ! -d "data/processed/vision" ]; then
    echo "📁 Creating vision data directory..."
    mkdir -p data/processed/vision
    echo "⚠️  Warning: No vision dataset found. Please prepare your dataset in data/processed/vision"
    echo "📋 Required: dataset_info.json file for LLaMA-Factory format"
fi

echo ">>> 👁️ STARTING INTERNVL2-76B TRAINING..."
echo "🔄 Using vision wrapper for consistent training..."

# Sử dụng vision wrapper để nhất quán với 03_train_vision.sh
python3 src/trainers/vision_wrapper.py configs/vision_internvl2_76b.yaml

echo ">>> ✅ Vision Model Training Completed!"