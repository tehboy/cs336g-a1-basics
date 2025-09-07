import pathlib
import time
import cProfile
from cs336_basics import bpe

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"
input_path = FIXTURES_PATH / "corpus.en"


def main():
    _, _ = bpe.run_sennrich_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    start_time = time.time()
    cProfile.run("main()", sort="cumtime")
    end_time = time.time()
    print(end_time)
