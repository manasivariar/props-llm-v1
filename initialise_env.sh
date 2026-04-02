#!/bin/bash
module load mamba/latest
source activate props
clear

kill
python3 main.py --config configs/mountaincar/mountaincar_propsp.yaml
python3 main.py --config configs/invertedpendulum/invertedpendulum_propsp.yaml

python3 main.py --config configs/cartpole/cartpole_propsp.yaml


watch -n 1 -t "myjobs | grep -Ec '^[[:space:]]*[0-9]'"

sinfo -p general --format="%N %G %C %t"


module load ollama/0.12.3
export OLLAMA_MODULES=/data/datasets/community/ollama
OLLAMA_CONTEXT_LENGTH=131072 OLLAMA_KEEP_ALIVE=-1 ollama serve
export OLLAMA_HOST=10.139.126.7:11434

 SDL_AUDIODRIVER=dummy python3 main.py --config configs/cartpole/cartpole_props+.yaml


seff 36850227
sacct -u $USER --starttime=2025-10-29 --format=JobID,JobName%-40,State
scancel 36835405
squeue -u $USER
scontrol update JobId=36861100 TimeLimit=10:00:00
myjobs

The error bind address already in use:
lsof -i :11436
# Kill the Ollama server
kill -9 2579528

sudo killall -s 9 ollama


# Kill the Python script connected to it (optional, but good for a clean slate)
kill -9 2580211

do nvidia-smi to see which process is taking GPU memory and kill it using kill -9 2580211