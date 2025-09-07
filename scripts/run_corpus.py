import pathlib
import time
import cProfile
from cs336_basics import bpe

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"
CORPUS_PATH = FIXTURES_PATH / "corpus.en"

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"
TINY_STORIES_VALID = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
TINY_STORIES_FULL = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"

def main():
    _, _ = bpe.run_nboy_bpe(
        input_path=TINY_STORIES_FULL,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    start_time = time.time()
#    cProfile.run("main()", sort="cumtime")
    main()
    end_time = time.time()
    print(end_time - start_time)
