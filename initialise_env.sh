#!/bin/bash

module load ollama/0.11.8
export OLLAMA_MODULES=/data/datasets/community/ollama
ollama-start
module load mamba/latest
source activate props

python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsp.yaml
python3 main.py --config configs/invertedpendulum/invertedpendulum_propsp.yaml


watch -n 1 -t "myjobs | grep -Ec '^[[:space:]]*[0-9]'"