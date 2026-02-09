#!/bin/bash

# Đường dẫn
FACTORY_DIR="./training_factory/outputs"
INFERENCE_DIR="./phil_inference/models"

mkdir -p $INFERENCE_DIR

echo ">>> 📦 Đang đóng gói Phil AI để triển khai..."

# 1. Copy Brain
if [ -d "$FACTORY_DIR/Phil-70B-Coder-N1" ]; then
    echo "   + Syncing Brain..."
    # Dùng rsync để copy nhanh và check thay đổi
    rsync -av --progress $FACTORY_DIR/Phil-70B-Coder-N1 $INFERENCE_DIR/
fi

# 2. Copy Vision
if [ -d "$FACTORY_DIR/Phil-InternVL2-76B-N1" ]; then
    echo "   + Syncing Eyes..."
    rsync -av --progress $FACTORY_DIR/Phil-InternVL2-76B-N1 $INFERENCE_DIR/
fi

# 3. Copy Audio Models
echo "   + Syncing Ears & Mouth..."
rsync -av $FACTORY_DIR/Phil-Ear-v1 $INFERENCE_DIR/
rsync -av $FACTORY_DIR/Phil-F5-TTS $INFERENCE_DIR/

echo ">>> ✅ Deploy hoàn tất! Chuyển sang thư mục 'phil_inference' và chạy 'docker-compose up'."