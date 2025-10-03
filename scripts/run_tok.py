import logging
import pickle
import time
import cProfile
from cs336_basics.bpe import Tokenizer, find_chunk_boundaries, ENDOFTEXT
import sys
import numpy as np
import os
import multiprocessing
from cs336_basics.utils import ModelArgs


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

    args = ModelArgs()

    def process_file(txt_path, vocab_path, merges_path, bpe_path, bpe_shape_path, dry_run):
        file_size = os.path.getsize(str(txt_path))
        num_chunks = max(1, file_size // CHUNK_SIZE)

        with open(str(txt_path), "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, ENDOFTEXT)

        chunk_args = [
            (txt_path, vocab_path, merges_path, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(tokenize_chunk, chunk_args)

        arr = np.concatenate(results, dtype=np.uint16)
        if not dry_run:
            arr.tofile(str(bpe_path))
            with open(str(bpe_shape_path), "wb") as f:
                pickle.dump(arr.shape, f)
            np.save(str(bpe_path), arr)
        logging.info(f"Wrote {len(arr)} tokens to {bpe_path}")

    process_file(
        args.txt_path,
        args.vocab_path,
        args.merges_path,
        args.bpe_path,
        args.bpe_shape_path,
        args.dry_run,
    )
    process_file(
        args.valid_txt_path,
        args.vocab_path,
        args.merges_path,
        args.valid_bpe_path,
        args.valid_bpe_shape_path,
        args.dry_run,
    )


if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)
