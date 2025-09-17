import logging
import pathlib
import time
import cProfile
from cs336_basics.bpe import _find_chunk_boundaries, Tokenizer
from cs336_basics.token_utils import load_vocab_and_merges
import argparse
import sys
import numpy as np


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set log level
        format="%(asctime)s %(levelname)s %(message)s",  # Set log format
        filename="run_tok.txt",  # Log to a file (optional)
        filemode="w",  # Overwrite file (optional)
    )

    parser = argparse.ArgumentParser(description="Run BPE tokenizer on input file.")
    parser.add_argument("--input_path", type=str, help="Path to input text file")
    parser.add_argument("--vocab_path", type=str, help="Path to vocab file")
    parser.add_argument("--merges_path", type=str, help="Path to merges file")
    parser.add_argument(
        "--profile", action="store_true", help="Profile the bpe tokenization process."
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Do not save bpe encoding to disk."
    )
    args = parser.parse_args()

    tok = Tokenizer.from_files(args.vocab_path, args.merges_path)
    with open(args.input_path) as f:
        arr = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
        out_path = args.input_path + ".bpe"
        if not args.dry_run:
            arr.tofile(out_path)
        logging.info(f"Wrote {len(arr)} tokens to {out_path}")



if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)