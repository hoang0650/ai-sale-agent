# 🏭 Phil AI Training Factory

> **"Xưởng đúc" Trí tuệ nhân tạo cho Phil - Thực thể số Việt Nam (Vietnam's Sovereign Digital Human).**
> Dự án này chuyên biệt hóa để Fine-tune các mô hình SOTA (State-of-the-Art) hạng nặng trên phần cứng **NVIDIA H200 SXM (141GB VRAM)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-H200_SXM-green.svg)
![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20LLaMA--Factory-red)
![Status](https://img.shields.io/badge/Status-Operational-brightgreen)

---

## 🧠 Kiến Trúc "Tứ Trụ" (The Big Four)

Hệ thống này không tạo ra một chatbot, mà tạo ra 4 thành phần cấu thành một con người kỹ thuật số:

| Thành phần | Vai trò | Model Gốc (Base) | Kỹ thuật Train | Dataset Chính |
| :--- | :--- | :--- | :--- | :--- |
| **1. Brain** | Tư duy, Code, Logic | `DeepSeek-R1-Distill-Llama-70B` | QLoRA 4-bit (Unsloth) | Glaive + Evol + **Vietnamese Translated** |
| **2. Eyes** | Nhìn, OCR, UI/UX | `OpenGVLab/InternVL2-76B` | QLoRA 4-bit (LLaMA-Factory) | OCR-VQA + Tech Screenshots |
| **3. Ears** | Nghe thuật ngữ IT | `OpenAI/Whisper-Large-v3` | LoRA Adapter | Youtube Tech Talks (Vietnamese) |
| **4. Mouth** | Giọng nói định danh | `F5-TTS (E2-TTS)` | Flow Matching | **Phil Studio Voice** (Custom) |

---

## 🛠️ Yêu Cầu Hệ Thống

Dự án này được tối ưu hóa cho **Runpod H200 Pod**. Không chạy được trên GPU dân dụng (RTX 4090) hoặc A100 80GB đơn lẻ (đối với Vision & Brain training).

* **GPU:** 1x NVIDIA H200 SXM (141GB VRAM).
* **Disk:** Tối thiểu 200GB Container Disk / Volume.
* **RAM:** 128GB+.
* **Internet:** Runpod Datacenter Speed (Download Dataset ~10Gbps).

---

## 📂 Cấu Trúc Dự Án

```text
phil-training-factory/
├── configs/                   # Cấu hình Hyperparameters (YAML)
│   ├── deepseek_70b.yaml      # Cấu hình Brain
│   ├── whisper_large.yaml     # Cấu hình Ears
│   └── ...
├── data/                      # Kho dữ liệu
│   ├── raw/                   # Dữ liệu thô
│   └── processed/             # Dữ liệu sạch (JSONL, WAV)
├── scripts/                   # Shell scripts điều khiển
│   ├── run_internvl2.sh       # Script riêng cho Vision
│   └── run_all.sh             # Script "One-Click" chạy tất cả
├── src/                       # Mã nguồn Python
│   ├── data_processing/       # Module dịch thuật & xử lý Audio
│   └── training/              # Module train Core (Unsloth & F5-TTS)
└── requirements.txt           # Dependencies
```

---

## 🚀 Hướng Dẫn Vận Hành (Step-by-Step)

**Bước 1: Khởi tạo Môi trường**
Kết nối SSH vào Runpod và chạy:
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Cấu hình biến môi trường
# Tạo file .env và điền Token HF của bạn vào
echo "HF_TOKEN=hf_write_token_here" > .env
```
**Bước 2: Chuẩn bị "Nguyên liệu" (Data Processing)**
Giai đoạn này dùng vinai/PhoGPT-4B để Việt hóa các bộ dataset Code chất lượng cao.
```bash
python3 src/data_processing/translator_ultimate.py
```
Output: `data/processed/combined_vietnamese_data.jsonl`

**Bước 3: Training**
Bạn có thể chạy từng module hoặc chạy tất cả.

**Cách 1: Chạy tự động (Khuyên dùng)
```bash
chmod +x scripts/*.sh
./scripts/run_all.sh
```
Lưu ý: Quá trình này mất khoảng 5-8 tiếng trên H200.

**Cách 2: Chạy thủ công từng phần**
1. **Train Brain (DeepSeek 70B):**
```bash
python3 src/training/train_generic.py --config configs/deepseek_70b.yaml
```
2. **Train Eyes (InternVL2 76B):**
```bash
./scripts/run_internvl2.sh
```
3. **Train Ears (Whisper):**
```bash
python3 src/training/train_generic.py --config configs/whisper_large.yaml
```
4. **Train Mouth (F5-TTS):**
Yêu cầu: Đã bỏ file giọng mẫu vào `data/processed/phil_voice_studio/`
```bash
python3 src/training/train_f5_tts.py
```
---

## 📦 Output Artifacts (Sản phẩm đầu ra)
Sau khi train xong, các model sẽ được tự động upload lên HuggingFace của bạn với tên:
* phil-ai/Phil-70B-Coder-N1 (Brain)
* phil-ai/Phil-InternVL2-76B-N1 (Vision)
* phil-ai/Phil-Ear-N1 (STT)
* phil-ai/Phil-F5-TTS (TTS Checkpoint)

---

## 🔌 Triển khai Inference (Phil-CLI)
Để sử dụng các model này, hãy chuyển sang project phil-cli và sử dụng cấu hình Docker Compose sau trên máy chủ Inference (Yêu cầu VRAM > 110GB):
```yaml
# Trích đoạn docker-compose.yml
services:
  ai-brain:
    image: vllm/vllm-openai
    command: --model phil-ai/Phil-70B-Coder-N1 --quantization awq ...
  
  ai-vision:
    image: openmmlab/lmdeploy
    command: lmdeploy serve api_server phil-ai/Phil-InternVL2-76B-v1 ...
```