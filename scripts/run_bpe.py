import pathlib
import time
import cProfile
from cs336_basics import bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"
TINY_STORIES_VALID = "TinyStoriesV2-GPT4-valid.txt"
TINY_STORIES_FULL =  "TinyStoriesV2-GPT4-train.txt"

def main():
    filename = TINY_STORIES_FULL
    vocab, merge_list = bpe.run_nboy_bpe(
        input_path=DATA_PATH / filename,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    with open(DATA_PATH / (filename + ".vocab"), "w", encoding="utf-8") as vocab_file:
        for token, bytes in vocab.items():
            vocab_file.write(f"{token} {bytes}\n")
    with open(DATA_PATH / (filename + ".merge_list"), "w", encoding="utf-8") as merge_file:
        for merge in merge_list:
            merge_file.write(f"{merge[0] + merge[1]}\n")


if __name__ == "__main__":
    start_time = time.time()
    cProfile.run('main()', sort='cumtime')
    end_time = time.time()
    print(end_time - start_time)
