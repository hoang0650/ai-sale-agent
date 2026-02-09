#!/bin/bash
# Script train InternVL2-76B trên H200

# Kiểm tra xem đã có dataset chưa
if [ ! -d "data/processed/phil_vision_data" ]; then
    echo "Creating dummy vision data for demo..."
    # Bạn cần code python để tạo dataset ảnh thật ở đây
    # Format LLaMA-Factory yêu cầu file dataset_info.json
fi

echo ">>> 👁️ STARTING INTERNVL2-76B TRAINING..."

# Sử dụng LLaMA Factory CLI (Cực mạnh cho model Vision)
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path OpenGVLab/InternVL2-76B \
    --dataset ocr_vqa_dataset \
    --template internvl2 \
    --finetuning_type lora \
    --lora_target all \
    --output_dir outputs/Phil-InternVL2-76B-N1 \
    --hub_model_id phil-ai/Phil-InternVL2-76B-N1 \
    --push_to_hub True \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --quantization_bit 4 \
    --bf16 True \
    --report_to wandb

echo ">>> ✅ Vision Model Done!"