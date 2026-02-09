import json
import os

OUTPUT_FILE = "data/processed/brain/tools_data.jsonl"

def build():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    samples = [
        {"input": "Mở nhạc Sơn Tùng", "tool": "play_music", "params": {"artist": "Sơn Tùng M-TP"}},
        {"input": "Tắt máy tính ngay", "tool": "system_control", "params": {"action": "shutdown"}},
        {"input": "Tra cứu giá Bitcoin", "tool": "web_search", "params": {"query": "Bitcoin price"}}
    ]
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            text = f"User: {s['input']}\nAssistant: <tool_code>{json.dumps({'tool': s['tool'], 'parameters': s['params']})}</tool_code>"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f">>> ✅ Agent Tools Dataset created at {OUTPUT_FILE}")

if __name__ == "__main__":
    build()