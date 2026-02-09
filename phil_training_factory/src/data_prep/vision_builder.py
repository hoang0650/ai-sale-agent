import json
import os
import glob

IMG_DIR = "data/processed/vision/images"
JSON_FILE = "data/processed/vision/phil_vision.json"
INFO_FILE = "data/processed/vision/dataset_info.json"

def build():
    if not os.path.exists(IMG_DIR):
        print(">>> ⚠️ Chưa có ảnh. Hãy copy ảnh vào data/processed/vision/images/")
        return

    images = glob.glob(f"{IMG_DIR}/*.*")
    data = []
    for img in images:
        name = os.path.basename(img)
        data.append({
            "images": [f"images/{name}"],
            "messages": [
                {"role": "user", "content": "Phân tích lỗi trong ảnh này."},
                {"role": "assistant", "content": "TODO: Điền nội dung..."}
            ]
        })

    with open(JSON_FILE, "w") as f: json.dump(data, f, indent=2)

    info = {
        "phil_vision_custom": {
            "file_name": "phil_vision.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
        }
    }
    with open(INFO_FILE, "w") as f: json.dump(info, f, indent=2)
    print(">>> ✅ Vision Dataset Configured.")

if __name__ == "__main__":
    build()