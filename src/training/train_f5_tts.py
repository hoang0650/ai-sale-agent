import os
import torch
from accelerate import Accelerator
from f5_tts.model import DiT, CFM
from f5_tts.train import Trainer
from dotenv import load_dotenv

load_dotenv()

# CẤU HÌNH CỨNG (Sửa trực tiếp ở đây cho nhanh)
DATASET_NAME = "phil_voice_studio"
OUTPUT_DIR = "outputs/F5-TTS-Phil"
HF_REPO = "phil-ai/Phil-F5-TTS"

def main():
    print(">>> 🎙️ STARTING F5-TTS TRAINING (CLONE YOUR VOICE)...")
    accelerator = Accelerator()
    device = accelerator.device

    # 1. Định nghĩa Model (Flow Matching Transformer)
    model = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4).to(device)
    cfm = CFM(transformer=model, sigma_min=0.0, sigma_max=1.0, ode_method='euler').to(device)

    # 2. Dataset Path (Yêu cầu cấu trúc: wavs/ và metadata.csv)
    dataset_path = os.path.join("data/processed", DATASET_NAME)
    
    # 3. Trainer
    trainer = Trainer(
        cfm,
        args={
            "num_warmup_updates": 200,
            "save_per_updates": 500,
            "checkpoint_path": OUTPUT_DIR,
            "batch_size": 4,  # H200 có thể tăng lên 8 hoặc 16
            "learning_rate": 1e-4,
            "accumulate_grad_batches": 4,
            "epochs": 50 # Train sâu để giọng mượt
        },
        dataset_path=dataset_path,
    )

    trainer.train()
    print(f">>> ✅ Train xong! Checkpoint lưu tại {OUTPUT_DIR}")
    # Lưu ý: F5-TTS hiện tại upload thủ công file .pt lên HF là tốt nhất

if __name__ == "__main__":
    main()