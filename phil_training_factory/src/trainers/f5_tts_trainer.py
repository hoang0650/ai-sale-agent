import yaml, sys
from accelerate import Accelerator
from f5_tts.model import DiT, CFM
from f5_tts.train import Trainer

def train(config_path):
    with open(config_path) as f: cfg = yaml.safe_load(f)
    print(">>> 🗣️ Training F5-TTS...")
    
    accelerator = Accelerator()
    model = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    cfm = CFM(transformer=model, sigma_min=0.0, sigma_max=1.0, ode_method='euler')
    
    trainer = Trainer(
        cfm,
        args={
            "num_warmup_updates": 200,
            "save_per_updates": cfg['training']['save_step'],
            "checkpoint_path": "outputs/f5_tts",
            "batch_size": cfg['training']['batch_size'],
            "learning_rate": cfg['training']['learning_rate'],
            "epochs": cfg['training']['epochs']
        },
        dataset_path=cfg['training']['dataset_path'],
    )
    trainer.train()

if __name__ == "__main__":
    train(sys.argv[1])