import json
import os
from datasets import load_dataset
from tqdm import tqdm

IMG_DIR = "data/processed/vision/images"
JSON_FILE = "data/processed/vision/phil_vision.json"
INFO_FILE = "data/processed/vision/dataset_info.json"
HF_DATASET_NAME = "sahil2801/CodeAlpaca-20k"

def create_ide_screenshot(text, filename):
    """
    Hàm này lấy Text Code và 'vẽ' nó lên một bức ảnh nền tối, 
    giả lập giao diện chụp màn hình của Visual Studio Code.
    """
    img_width, img_height = 800, 600
    # Màu nền Dark Mode (#1E1E1E)
    image = Image.new("RGB", (img_width, img_height), color="#1E1E1E")
    draw = ImageDraw.Draw(image)

    # Sử dụng font mặc định của hệ thống để không bị lỗi thiếu file font
    font = ImageFont.load_default()

    # Chèn chữ vào ảnh, tự động xuống dòng nếu code dài
    margin = 20
    offset = 20
    for line in text.split('\n'):
        # Cắt bớt nếu dòng quá 110 ký tự để không tràn ngang
        draw.text((margin, offset), line[:110], font=font, fill="#D4D4D4") 
        offset += 20
        # Dừng vẽ nếu tràn chiều dọc của ảnh
        if offset > img_height - 40: 
            draw.text((margin, offset), "... (code truncated)", font=font, fill="#F44336")
            break

    image.save(filename)

def build():
    print(f">>> 📥 Tải dataset văn bản (MIT License): {HF_DATASET_NAME}...")
    os.makedirs(IMG_DIR, exist_ok=True)
    
    # Tải dataset văn bản
    ds = load_dataset(HF_DATASET_NAME, split="train")
    
    # BỘ LỌC THÔNG MINH: Chỉ lấy những mẫu có chứa từ khóa liên quan đến "sửa lỗi" (bug/fix/error)
    bug_data = [item for item in ds if "fix" in item["instruction"].lower() or "bug" in item["instruction"].lower() or "error" in item["instruction"].lower()]
    
    print(f">>> 🔍 Tìm thấy {len(bug_data)} mẫu sửa lỗi code. Bắt đầu tạo ảnh chụp màn hình giả lập...")

    data_json = []
    
    # Trích xuất 500 mẫu để train (Để test nhanh. Bạn có thể bỏ [:500] để train toàn bộ)
    for i, item in enumerate(tqdm(bug_data[:500], desc="Generating IDE Screenshots")):
        img_filename = f"code_bug_{i}.jpg"
        img_path = os.path.join(IMG_DIR, img_filename)
        
        # 1. Ghép câu lệnh yêu cầu và đoạn code lỗi lại với nhau
        buggy_code = f"// User Request: {item['instruction']}\n\n{item['input']}"
        
        # 2. Vẽ thành ảnh chụp màn hình IDE
        create_ide_screenshot(buggy_code, img_path)
        
        # 3. Lấy câu trả lời (Code đã sửa) từ Dataset
        fixed_code = item['output']
        
        # 4. Format dữ liệu theo chuẩn Vision (ShareGPT) của LLaMA-Factory
        data_json.append({
            "images": [f"images/{img_filename}"],
            "messages": [
                {"role": "user", "content": "<image>\nPhil, hãy xem ảnh chụp màn hình này. Code đang bị lỗi, hãy tìm lỗi và viết lại bản sửa lỗi giúp tôi."},
                {"role": "assistant", "content": f"Dựa vào ảnh chụp màn hình, tôi đã phát hiện ra vấn đề. Dưới đây là đoạn code đã được sửa và tối ưu lại:\n\n```python\n{fixed_code}\n```"}
            ]
        })

    # Lưu file JSON cấu hình
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    # Khai báo với LLaMA-Factory
    info = {
        "phil_vision_custom": {
            "file_name": "phil_vision.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        }
    }
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        
    print(f">>> ✅ Hoàn tất! Đã tự động tạo ra {len(data_json)} ảnh chụp màn hình báo lỗi code từ dataset MIT.")

if __name__ == "__main__":
    build()