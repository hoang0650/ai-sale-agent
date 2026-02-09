import yaml, sys, os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets

def train(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    print(f">>> 🧠 Training Brain: {cfg['model']['new_model_name']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg['model']['base_model'],
        max_seq_length=cfg['training']['max_seq_length'],
        load_in_4bit=cfg['training']['load_in_4bit']
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=cfg['training']['lora_rank'], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg['training']['lora_alpha'], use_gradient_checkpointing="unsloth"
    )

    # Load Main Data + Tools Data
    ds1 = load_dataset("json", data_files=cfg['training']['dataset_path'], split="train")
    ds2 = load_dataset("json", data_files="data/processed/brain/tools_data.jsonl", split="train")
    dataset = concatenate_datasets([ds1, ds2]).shuffle(seed=42)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
        max_seq_length=cfg['training']['max_seq_length'],
        args=TrainingArguments(
            per_device_train_batch_size=cfg['training']['batch_size'],
            gradient_accumulation_steps=cfg['training']['grad_accum'],
            learning_rate=cfg['training']['learning_rate'],
            num_train_epochs=cfg['training']['epochs'],
            fp16=True, output_dir=cfg['training']['output_dir'], report_to="wandb"
        )
    )
    trainer.train()
    model.push_to_hub_merged(f"{cfg['model']['hf_username']}/{cfg['model']['new_model_name']}", tokenizer, save_method="merged_4bit_forced", token=os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    train(sys.argv[1])