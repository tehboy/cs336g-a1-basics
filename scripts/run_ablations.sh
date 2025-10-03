#!/bin/bash
pushd /home/nathan_boy/cs336g-a1-basics
uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_norms
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint mv ../data/TinyStoriesV2-GPT4-train.txt.ablate_norms.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --ablate_position_embeddings
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint mv ../data/TinyStoriesV2-GPT4-train.txt.ablate_position_embeddings.checkpoint

uv run scripts/run_model.py --model_file conf/TinyStories.yaml --use_silu --dff 2048
mv ../data/TinyStoriesV2-GPT4-train.txt.best.checkpoint mv ../data/TinyStoriesV2-GPT4-train.txt.use_silu.checkpoint

uv run scripts/run_model.py --model_file conf/owt.yaml

popd