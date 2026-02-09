# 🏭 Phil AI Training & Inference Factory

> **"Xưởng đúc" Trí tuệ nhân tạo cho Phil - Thực thể số Việt Nam (Vietnam's Sovereign Digital Human).**
> Dự án này chuyên biệt hóa để Fine-tune các mô hình SOTA (State-of-the-Art) hạng nặng trên phần cứng **NVIDIA H200 SXM (141GB VRAM)** và cung cấp giải pháp Inference đa nền tảng.

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

## 🚀 Tính năng mới: Đa nền tảng Inference

Hệ thống hiện đã tích hợp các engine inference mạnh mẽ nhất để tối ưu hóa tốc độ và tài nguyên cho việc triển khai AI Sale Agent:

1.  **vLLM**: Tối ưu hóa throughput cho GPU NVIDIA, hỗ trợ PagedAttention.
2.  **Text Generation Inference (TGI)**: Giải pháp từ HuggingFace cho việc triển khai production.
3.  **llama.cpp**: Chạy mô hình trên CPU hoặc GPU với định dạng GGUF, cực kỳ tiết kiệm tài nguyên.
4.  **Transformers**: Backend mặc định cho việc thử nghiệm nhanh.

---

## 🔄 Chiến lược "Cuốn chiếu" (Rolling Strategy)

Để tiết kiệm chi phí thuê ổ cứng trên RunPod, dự án triển khai chiến lược **"Cuốn chiếu"** trong script `run_all.sh`:
- **Quy trình**: Tải Model Gốc -> Huấn luyện (Fine-tune) -> Upload kết quả lên HuggingFace -> **Xóa Model Gốc & Cache** -> Chuyển sang model tiếp theo.
- **Lợi ích**: Giảm yêu cầu dung lượng Disk từ >500GB xuống còn khoảng 200GB, ngay cả khi làm việc với các model khổng lồ như DeepSeek 70B hay InternVL2 76B.

---

## 🛠️ Yêu Cầu Hệ Thống

Dự án này được tối ưu hóa cho **Runpod H200 Pod**.

* **GPU:** 1x NVIDIA H200 SXM (141GB VRAM).
* **Disk:** Tối thiểu 200GB Container Disk / Volume (Nhờ chiến lược Cuốn chiếu).
* **RAM:** 128GB+.
* **Internet:** Runpod Datacenter Speed (Download Dataset ~10Gbps).

---

## 📂 Cấu Trúc Dự Án

```text
ai-sale-agent/
├── 🏭 phil_training_factory/    # Xưởng luyện Model (Chạy 1 lần)
│   ├── configs/                 # Chỉnh tham số train
│   ├── scripts/                 # Script tự động hóa
│   └── outputs/                 # Nơi Model ra lò
│
├── 🚀 phil_inference/           # Server triển khai (Chạy 24/7)
│   ├── config/                  # Chọn backend (vLLM/TGI)
│   ├── src/                     # API Gateway Logic
│   └── docker-compose.yml       # Hạ tầng container
```

---

## 🚀 Hướng Dẫn Vận Hành (Step-by-Step)

### Bước 1: Khởi tạo Môi trường
Yêu cầu: NVIDIA H200 (141GB VRAM).
Kết nối SSH vào Runpod và chạy:
```bash
git clone https://github.com/hoang0650/ai-sale-agent
cd phil_training_factory
pip install -r requirements.txt
### Khai báo nhiều biến môi trường
### Cách 1
cp .env.example .env
### Cách 2
echo "HF_TOKEN=hf_write_token_here" > .env
echo "WANDB_API_KEY=write_wandb_api_key" >> .env
### Cách 3
cat << EOF > .env
HF_TOKEN=hf_write_token_here
WANDB_API_KEY=write_wandb_api_key
EOF
```

### Bước 2: Chạy toàn bộ quy trình (Chiến lược Cuốn chiếu)
Bạn chỉ cần chạy 1 lệnh duy nhất để train toàn bộ 4 model:
```bash
# Script này sẽ tự động:
# 1. Tải và xử lý dữ liệu (Dịch sang tiếng Việt)
# 2. Train Brain (DeepSeek 70B)
# 3. Train Vision (InternVL2 76B)
# 4. Train Audio (Whisper + F5-TTS)
chmod +x scripts/*.sh
./scripts/run_all.sh
```
Sau khi chạy xong, kết quả sẽ nằm trong thư mục `phil_training_factory/outputs/`.

### Bước 3: CHUYỂN ĐỔI & TRIỂN KHAI (PHIL INFERENCE)
Bạn có thể chọn engine thông qua biến môi trường:

**Chuyển Model sang Inference**
Chúng ta cần copy model từ "Xưởng" sang thư mục "Triển khai".
```bash
mkdir -p phil_inference/models
cp -r phil_training_factory/outputs/* phil_inference/models/
```

**Lựa chọn Backend (vLLM vs TGI vs llama.cpp)**
Mở file `phil_inference/config/model_config.yaml` để cấu hình.
**Option A: Dùng vLLM (Khuyên dùng cho H200 - Tốc độ cao nhất)**
```yaml
brain:
  active_backend: "vllm"
```
Ưu điểm: Hỗ trợ PagedAttention, throughput cực cao.
**Option B: Dùng llama.cpp (Nếu muốn chạy tiết kiệm VRAM)**
Trước tiên, cần convert model sang GGUF:
```bash
# Tại thư mục phil_training_factory
python3 convert_hf_to_gguf.py outputs/Phil-70B-Coder-N2 --outfile models/phil-brain.gguf
```
Sau đó sửa config:
```yaml
brain:
  active_backend: "llamacpp"
```
**Khởi động Server**
```bash
cd phil_inference
docker-compose up -d --build
```
Hệ thống sẽ khởi động các container:
* vllm-brain (Port 8000)
* vllm-vision (Port 8001)
* phil-gateway (Port 3000 - API chính)

**SỬ DỤNG (PHIL CLI)**
Trên máy tính cá nhân của bạn:
```bash
cd phil-cli
pip install requests

# Chat với Phil
python phil.py chat "Phil ơi, viết cho anh code Python giải thuật Dijkstra"

# Nhờ Phil nhìn lỗi
python phil.py see ./error_screenshot.png --prompt "Lỗi này sửa sao em?"
```

---

## ☁️ Triển khai lên RunPod

Xem chi tiết trong file [RunPod_Deployment_Guide.docx](https://docs.google.com/document/d/1JeqsSHzRNZQ1dpyWgQYaZKOmHqpz86a5/edit?usp=sharing&ouid=111551674717295623221&rtpof=true&sd=true) để biết cách thiết lập môi trường GPU trên RunPod.

---

## 📦 Output Artifacts (Sản phẩm đầu ra)
Sau khi train xong, các model sẽ được tự động upload lên HuggingFace của bạn.

---

## 📄 Giấy phép
Code dự án tuân thủ MIT/Apache 2.0. Dữ liệu training đã được lọc để đảm bảo quyền thương mại (Commercial Use).
