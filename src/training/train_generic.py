import yaml
import argparse
import torch
import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio, concatenate_datasets
from peft import LoraConfig, get_peft_model

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def train(config_path):
    with open(config_path, 'r') as f: cfg = yaml.safe_load(f)
    print(f"\n>>> 🚀 STARTING TRAINING: {cfg['new_model_name']}")

    # ==========================================
    # CASE 1: TRAIN LLM (DEEPSEEK / QWEN)
    # ==========================================
    if "DeepSeek" in cfg['base_model'] or "Llama" in cfg['base_model']:
        print(">>> Mode: LLM Fine-Tuning (Unsloth Optimized)")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg['base_model'],
            max_seq_length=cfg['max_seq_length'],
            dtype=None, 
            load_in_4bit=cfg['load_in_4bit']
        )
        
        model = FastLanguageModel.get_peft_model(
            model, r=cfg['lora_rank'], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16, bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
        )

        # Load & Mix Datasets
        ds_vn = load_dataset("json", data_files=cfg['dataset_path'], split="train")
        # Thêm dataset gốc tiếng Anh (Optional)
        ds_en = load_dataset("glaiveai/glaive-code-assistant-v2", split="train[:5000]")
        ds_en = ds_en.rename_column("question", "instruction").rename_column("answer", "output")
        dataset = concatenate_datasets([ds_vn, ds_en]).shuffle(seed=42)
        # 3. Dataset Danh tính (Identity) - MỚI THÊM VÀO
        identity_file = "data/processed/phil_identity.jsonl"
        if os.path.exists(identity_file):
            ds_identity = load_dataset("json", data_files=identity_file, split="train")
            print(f">>> Đã thêm {len(ds_identity)} dòng dữ liệu danh tính Phil AI.")
            # Trộn 3 nguồn lại
            dataset = concatenate_datasets([ds_vn, ds_en, ds_identity]).shuffle(seed=42)
        else:
            print(">>> CẢNH BÁO: Không tìm thấy dữ liệu danh tính. Model có thể không biết tên mình.")
            dataset = concatenate_datasets([ds_vn, ds_en]).shuffle(seed=42)

        # Format DeepSeek R1
        def format_prompts(examples):
            texts = []
            for inst, out in zip(examples["instruction"], examples["output"]):
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
                texts.append(text)
            return {"text": texts}
        dataset = dataset.map(format_prompts, batched=True)

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
            max_seq_length=cfg['max_seq_length'],
            args=TrainingArguments(
                per_device_train_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['grad_accum'],
                max_steps=cfg['steps'], learning_rate=cfg['learning_rate'], fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(), logging_steps=1, output_dir="outputs", optim="paged_adamw_8bit",
                report_to="wandb"
            )
        )
        trainer.train()
        print(">>> Saving & Uploading LLM...")
        model.push_to_hub_merged(f"{cfg['hf_username']}/{cfg['new_model_name']}", tokenizer, save_method="merged_4bit_forced", token=HF_TOKEN)

    # ==========================================
    # CASE 2: TRAIN WHISPER (STT)
    # ==========================================
    elif "whisper" in cfg['base_model']:
        print(">>> Mode: Whisper Fine-Tuning")
        model = WhisperForConditionalGeneration.from_pretrained(cfg['base_model'], load_in_8bit=True, device_map="auto")
        processor = WhisperProcessor.from_pretrained(cfg['base_model'], language=cfg['language'], task="transcribe")
        
        # Load Dataset
        dataset = load_dataset(cfg['dataset_name'], cfg['dataset_subset'], split="train", trust_remote_code=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
            batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
            return batch
        
        dataset = dataset.map(prepare_dataset, num_proc=4)
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], bias="none"))

        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir="outputs_whisper", per_device_train_batch_size=cfg['batch_size'],
                gradient_accumulation_steps=cfg['grad_accum'], max_steps=cfg['steps'], learning_rate=cfg['learning_rate'],
                fp16=True, predict_with_generate=True, report_to="wandb"
            ),
            train_dataset=dataset, tokenizer=processor.feature_extractor,
            data_collator=lambda data: {
                "input_features": torch.stack([torch.tensor(f["input_features"]) for f in data]),
                "labels": torch.tensor([f["labels"] for f in data]).nn.pad_sequence(padding_value=-100)
            }
        )
        trainer.train()
        print(">>> Uploading Whisper...")
        model.push_to_hub(f"{cfg['hf_username']}/{cfg['new_model_name']}", token=HF_TOKEN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)