import torch
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# CẤU HÌNH
TRANSLATOR_MODEL = "vinai/PhoGPT-4B-Chat"
OUTPUT_FILE = "data/processed/combined_vietnamese_data.jsonl"
SAMPLES_TO_TRANSLATE = 5000  # Số mẫu lấy từ mỗi dataset (Tăng lên nếu có thời gian)

def load_translator():
    print(">>> 🔄 Đang tải PhoGPT...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        TRANSLATOR_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    return model, tokenizer

def translate_text(model, tokenizer, text):
    prompt = f"### Instruction:\nDịch phần giải thích sau sang Tiếng Việt. GIỮ NGUYÊN CODE, TÊN BIẾN, TIẾNG ANH CHUYÊN NGÀNH.\n\nInput:\n{text[:1500]}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, max_new_tokens=512, do_sample=True, temperature=0.6,
            top_p=0.9, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip() if "### Response:" in response else response

def run():
    os.makedirs("data/processed", exist_ok=True)
    model, tokenizer = load_translator()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 1. Glaive Dataset
        ds = load_dataset("glaiveai/glaive-code-assistant-v2", split=f"train[:{SAMPLES_TO_TRANSLATE}]")
        for item in tqdm(ds, desc="Translating Glaive"):
            try:
                vn_instr = translate_text(model, tokenizer, item['question'])
                # Lưu cả bản gốc và bản dịch để train song ngữ
                record = {"instruction": vn_instr, "output": item['answer'], "source": "glaive_vn"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except: continue

        # 2. Evol Dataset
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split=f"train[:{SAMPLES_TO_TRANSLATE}]")
        for item in tqdm(ds, desc="Translating Evol"):
            try:
                vn_instr = translate_text(model, tokenizer, item['instruction'])
                record = {"instruction": vn_instr, "output": item['output'], "source": "evol_vn"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except: continue
            
    print(f">>> ✅ Đã dịch xong! File lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()