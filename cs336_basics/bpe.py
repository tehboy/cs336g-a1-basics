from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import BinaryIO, Iterable, TypeAlias

import os
import regex

from .common_types import BytePair, ByteSequence, MergeList, Vocab
from itertools import chain

ByteSequenceCounts: TypeAlias = Mapping[ByteSequence, int]

ENDOFTEXT: bytes = "<|endoftext|>".encode("utf-8")


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _initialize_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocabulary = {i: i.to_bytes() for i in range(256)}
    for token in special_tokens:
        vocabulary[len(vocabulary)] = token.encode("utf-8")
    return vocabulary


def _sennrich_compute_next_bp(word_counts: ByteSequenceCounts) -> BytePair | None:
    max_bp: BytePair = (bytes(), bytes())  # always reassigned
    max_bp_count = 0
    bp_counts: dict[BytePair, int] = defaultdict(int)
    for word, count in word_counts.items():
        for bp in zip(word[0:-1], word[1:]):
            bp_count = bp_counts[bp] + count
            bp_counts[bp] = bp_count
            if bp_count > max_bp_count or (bp_count == max_bp_count and bp > max_bp):
                max_bp = bp
                max_bp_count = bp_count
    if max_bp_count == 0:
        return None
    return max_bp


def _update_byte_sequence_with_bp(byte_sequence_counts: ByteSequenceCounts, bp: BytePair) -> ByteSequenceCounts:
    def _replace_bps(byte_seq: ByteSequence, bp: BytePair) -> ByteSequence:
        result = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i + 1]) == bp:
                result.append(byte_seq[i] + byte_seq[i + 1])
                i += 2  # Skip the next one, since it's merged
            else:
                result.append(byte_seq[i])
                i += 1
        return tuple(result)

    return {_replace_bps(k, bp): v for k, v in byte_sequence_counts.items()}


def _pretokenize_words(input: str, special_tokens: list[str]) -> dict[ByteSequence, int]:
    PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def split_on_special_token(inputs: Iterable[str], token: str) -> Iterable[str]:
        return chain.from_iterable(input.split(token) for input in inputs)

    input_splits: Iterable[str] = [input]
    for token in special_tokens:
        input_splits = split_on_special_token(input_splits, token)
    pretoken_counts: dict[ByteSequence, int] = defaultdict(int)
    for input_split in input_splits:
        for match in regex.finditer(PRETOKEN_PATTERN, input_split):
            word = match.captures(0)[0]
            pretoken_counts[tuple(map(int.to_bytes, word.encode("utf-8")))] += 1
    return pretoken_counts


def _merge_byte_sequence_counts(one: ByteSequenceCounts, two: ByteSequenceCounts) -> ByteSequenceCounts:
    merged = defaultdict(int, one)
    for k, v in two.items():
        merged[k] += v
    return dict(merged)


def run_sennrich_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, Sequence[BytePair]]:
    vocab: dict[int, bytes] = _initialize_vocabulary(special_tokens)
    if len(vocab) > vocab_size:
        ValueError("vocab_size is too low.")
    boundaries = []
    with open(input_path, "rb") as input_file:
        boundaries: list[int] = _find_chunk_boundaries(input_file, 16, ENDOFTEXT)
        byte_sequence_counts: ByteSequenceCounts = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            input_file.seek(start)
            chunk_data = input_file.read(end - start)
            byte_sequence_counts = _merge_byte_sequence_counts(
                byte_sequence_counts, _pretokenize_words(chunk_data.decode("utf-8"), special_tokens)
            )
    merge_list = []
    for _ in range(vocab_size - len(vocab)):
        bp = _sennrich_compute_next_bp(byte_sequence_counts)
        if bp is None:
            break
        byte_sequence_counts = _update_byte_sequence_with_bp(byte_sequence_counts, bp)
        merge_list.append(bp)
    for bp in merge_list:
        merged_token = bp[0] + bp[1]
        vocab[len(vocab)] = merged_token
    return (vocab, merge_list)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
