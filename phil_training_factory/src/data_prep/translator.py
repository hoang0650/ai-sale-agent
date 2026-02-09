import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

TRANSLATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_FILE = "data/processed/brain/combined_data.jsonl"

def run():
    print(f">>> 🔄 Loading Translator: {TRANSLATOR_MODEL}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TRANSLATOR_MODEL, torch_dtype=torch.float16, device_map="auto")

    ds = load_dataset("glaiveai/glaive-code-assistant-v2", split="train[:2000]") # Demo 2000 mẫu

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="Translating"):
            text = item['question']
            prompt = f"Dịch sang tiếng Việt, giữ nguyên Code và thuật ngữ IT:\n{text}"
            
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
            
            out = model.generate(input_ids, max_new_tokens=512, temperature=0.3)
            trans = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            f.write(json.dumps({"instruction": trans, "output": item['answer']}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run()