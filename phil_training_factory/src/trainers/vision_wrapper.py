import yaml, sys, subprocess

def train(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    print(f">>> 👁️ Training Vision via LLaMA-Factory...")

    cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft", "--do_train",
        "--model_name_or_path", cfg['model']['base_model'],
        "--dataset_dir", cfg['training']['dataset_dir'],
        "--dataset", cfg['training']['dataset_name'],
        "--template", cfg['training']['template'],
        "--finetuning_type", cfg['training']['finetuning_type'],
        "--output_dir", f"outputs/{cfg['model']['new_model_name']}",
        "--quantization_bit", str(cfg['training']['quantization_bit']),
        "--num_train_epochs", str(cfg['training']['epochs']),
        "--per_device_train_batch_size", str(cfg['training']['batch_size']),
        "--gradient_accumulation_steps", str(cfg['training']['grad_accum']),
        "--learning_rate", str(cfg['training']['learning_rate']),
        "--bf16", "True", "--overwrite_output_dir"
    ]
    
    if cfg['model'].get('hf_username'):
        cmd.extend(["--hub_model_id", f"{cfg['model']['hf_username']}/{cfg['model']['new_model_name']}", "--push_to_hub", "True"])

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    train(sys.argv[1])