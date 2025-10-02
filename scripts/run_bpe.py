import logging
import pathlib
import time
import cProfile
from cs336_basics import bpe
from cs336_basics.token_utils import save_vocab_and_merges
import sys
from cs336_basics.utils import ModelArgs


def main():
    logging.basicConfig(
        level=logging.INFO,  # Set log level
        format="%(asctime)s %(levelname)s %(message)s",  # Set log format
        filename="run_bpe.txt",  # Log to a file (optional)
        filemode="w",  # Overwrite file (optional)
    )
    args = ModelArgs()

    input_path = pathlib.Path(str(args.txt_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path '{input_path}' does not point to a valid file.")

    vocab, merge_list = bpe.run_nboy_bpe(
        input_path=input_path,
        vocab_size=int(args.vocab_size),
        special_tokens=list(args.special_tokens) if args.special_tokens is not None else [],
    )

    logging.info(
        f"dry_run: {args.dry_run}, save_vocab_and_merges(vocab, merge_list, vocab_path={args.vocab_path}, merges_path={args.merges_path})"
    )
    if not args.dry_run:
        save_vocab_and_merges(
            vocab, merge_list, vocab_path=str(args.vocab_path), merges_path=str(args.merges_path)
        )


if __name__ == "__main__":
    start_time = time.time()
    if "--profile" in sys.argv:
        cProfile.run("main()", sort="cumtime")
    else:
        main()
    end_time = time.time()
    print(end_time - start_time)
