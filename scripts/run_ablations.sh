#!/bin/bash
# Get the directory of the script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# Go to the project root
pushd "$SCRIPT_DIR/.."

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_norms
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.ablate_norms.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_position_embeddings
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.ablate_position_embeddings.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --use_silu --dff 2048
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint ../data/TinyStoriesV2-GPT4-train.txt.use_silu.checkpoint

uv run scripts/run_model.py --model_file conf/owt.yaml

popd