import logging
import pathlib
import time
import cProfile
from cs336_basics import bpe
from cs336_basics.token_utils import save_vocab_and_merges
import argparse
import pickle
import sys


def main():
    logging.basicConfig(
        level=logging.INFO,                # Set log level
        format="%(asctime)s %(levelname)s %(message)s",  # Set log format
        filename="run_bpe.txt",              # Log to a file (optional)
        filemode="w"                       # Overwrite file (optional)
    )
    parser = argparse.ArgumentParser(description="Run BPE on input file.")
    parser.add_argument("input_path", type=str, help="Path to input text file")
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size")
    parser.add_argument(
        "--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="Special tokens"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile the bpe tokenization process."
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Do not save vocab and merges to disk."
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path '{input_path}' does not point to a valid file.")

    vocab, merge_list = bpe.run_nboy_bpe(
        input_path=input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    if not args.dry_run:
        vocab_path = input_path.with_suffix(input_path.suffix + ".vocab")
        merges_path = input_path.with_suffix(input_path.suffix + ".merges")
        save_vocab_and_merges(vocab, merge_list, vocab_path=vocab_path, merges_path=merges_path)


if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)
