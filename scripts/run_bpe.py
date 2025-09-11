import pathlib
import time
import cProfile
from cs336_basics import bpe
import argparse
import pickle
import sys


def main():
    parser = argparse.ArgumentParser(description="Run BPE on input file.")
    parser.add_argument("input_path", type=str, help="Path to input text file")
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocabulary size")
    parser.add_argument(
        "--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="Special tokens"
    )
    parser.add_argument("--debug", action="store_true", help="Write out human readable versions.")
    parser.add_argument(
        "--profile", action="store_true", help="Profile the bpe tokenization process."
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
    vocab_pickle_path = input_path.with_suffix(input_path.suffix + ".vocab")
    merge_list_pickle_path = input_path.with_suffix(input_path.suffix + ".merge_list")
    with open(vocab_pickle_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merge_list_pickle_path, "wb") as f:
        pickle.dump(merge_list, f)
    if args.debug:
        vocab_file_path = input_path.with_suffix(input_path.suffix + ".vocab.txt")
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for token, bytes in vocab.items():
                vocab_file.write(f"{token} {bytes}\n")
        merge_list_file_path = input_path.with_suffix(input_path.suffix + ".merge_list.txt")
        with open(merge_list_file_path, "w", encoding="utf-8") as merge_file:
            for merge in merge_list:
                merge_file.write(f"{merge[0] + merge[1]}\n")


if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)
