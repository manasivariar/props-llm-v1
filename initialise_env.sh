#!/bin/bash

module load ollama/0.12.10
export OLLAMA_MODULES=/data/datasets/community/ollama
ollama-start
module load mamba/latest
source activate props
clear

python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_propsp.yaml
python3 main.py --config configs/mountaincar/mountaincar_propsp.yaml
python3 main.py --config configs/invertedpendulum/invertedpendulum_propsp.yaml

python3 main.py --config configs/cartpole/cartpole_propsp_reward.yaml


watch -n 1 -t "myjobs | grep -Ec '^[[:space:]]*[0-9]'"

sinfo -p general --format="%N %G %C %t"

pkill ollama
export OLLAMA_HOST=0.0.0.0
OLLAMA_KEEP_ALIVE=-1 ollama serve

 SDL_AUDIODRIVER=dummy python3 main.py --config configs/cartpole/cartpole_props+.yaml


seff 36850227
sacct -u $USER --starttime=2025-10-29 --format=JobID,JobName%-40,State
scancel 36835405
squeue -u $USER
scontrol update JobId=36861100 TimeLimit=10:00:00
myjobs