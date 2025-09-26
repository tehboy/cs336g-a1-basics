import logging
import time
import cProfile
from cs336_basics.bpe import Tokenizer, find_chunk_boundaries, ENDOFTEXT
import argparse
import sys
import numpy as np
import os
import multiprocessing

# 10 MB chunks
CHUNK_SIZE = 10 * 1024 * 1024


def tokenize_chunk(args_tuple):
    """Tokenizes a chunk of a file and returns a numpy array of tokens."""
    input_path, vocab_path, merges_path, start, end = args_tuple
    logging.info(f"Tokenizing chunk from {start} to {end}")
    tok = Tokenizer.from_files(vocab_path, merges_path)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        tokens = tok.encode(chunk_data.decode("utf-8", errors="replace"))
        return np.array(tokens, dtype=np.uint16)


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
    parser.add_argument("--dry_run", action="store_true", help="Do not save bpe encoding to disk.")
    args = parser.parse_args()

    file_size = os.path.getsize(args.input_path)
    num_chunks = max(1, file_size // CHUNK_SIZE)

    with open(args.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, ENDOFTEXT)

    chunk_args = [
        (args.input_path, args.vocab_path, args.merges_path, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(tokenize_chunk, chunk_args)

    arr = np.concatenate(results)
    out_path = args.input_path + ".bpe"
    if not args.dry_run:
        np.save(out_path, arr)
    logging.info(f"Wrote {len(arr)} tokens to {out_path}")


if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)
