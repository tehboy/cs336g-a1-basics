#!/bin/bash
# Get the directory of the script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Go to the project root
pushd "$SCRIPT_DIR/.."

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_norms
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.ablate_norms.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_position_embeddings
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.ablate_position_embeddings.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --use_silu --d_ff 2048
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.use_silu.checkpoint

uv run scripts/run_model.py --model_file conf/owt.yaml

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --max_lr=0.01 --min_lr=0.001
uv run scripts/run_model.py --model_file conf/TinyStories.yaml --max_lr=0.005 --min_lr=0.0005
uv run scripts/run_model.py --model_file conf/TinyStories.yaml --max_lr=0.0001 --min_lr=0.00001
uv run scripts/run_model.py --model_file conf/TinyStories.yaml --max_lr=0.05 --min_lr=0.00001

popd