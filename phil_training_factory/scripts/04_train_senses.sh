#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python3 src/trainers/whisper_trainer.py configs/ear_whisper_large.yaml
python3 src/trainers/f5_tts_trainer.py configs/mouth_f5_tts.yaml