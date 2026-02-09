import yaml, sys, os, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model

def train(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    print(">>> 👂 Training Whisper...")

    model = WhisperForConditionalGeneration.from_pretrained(cfg['model']['base_model'], load_in_8bit=True, device_map="auto")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    processor = WhisperProcessor.from_pretrained(cfg['model']['base_model'], language=cfg['model']['language'], task="transcribe")
    
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], bias="none"))

    dataset = load_dataset(cfg['training']['dataset_name'], cfg['training']['dataset_subset'], split="train", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir="outputs/whisper",
            per_device_train_batch_size=cfg['training']['batch_size'],
            gradient_accumulation_steps=cfg['training']['grad_accum'],
            max_steps=cfg['training']['steps'],
            learning_rate=cfg['training']['learning_rate'],
            fp16=True, report_to="wandb"
        ),
        train_dataset=dataset,
        data_collator=lambda data: {
            "input_features": torch.stack([torch.tensor(f["input_features"]) for f in data]),
            "labels": torch.tensor([f["labels"] for f in data]).nn.pad_sequence(padding_value=-100)
        }
    )
    trainer.train()
    model.push_to_hub(f"{cfg['model']['hf_username']}/{cfg['model']['new_model_name']}", token=os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    train(sys.argv[1])