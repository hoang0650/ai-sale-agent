#!/bin/bash
echo ">>> Data Prep Phase..."
python3 src/data_prep/translator.py
python3 src/data_prep/agent_builder.py
python3 src/data_prep/vision_builder.py