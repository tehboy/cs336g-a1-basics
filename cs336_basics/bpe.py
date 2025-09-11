from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import BinaryIO, Iterable, TypeAlias

import os
import pickle
import regex

from .common_types import BytePair, ByteSequence, MergeList, Vocab
from itertools import chain

import multiprocessing

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


def _sennrich_compute_next_bp(
    word_counts: ByteSequenceCounts,
) -> BytePair | None:
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


def _replace_bps_in_bseq(byte_seq: ByteSequence, bp: BytePair) -> ByteSequence:
    result = []
    i = 0
    byte_seq_len = len(byte_seq)
    while i < byte_seq_len:
        if i < byte_seq_len - 1 and (byte_seq[i], byte_seq[i + 1]) == bp:
            result.append(byte_seq[i] + byte_seq[i + 1])
            i += 2  # Skip the next one, since it's merged
        else:
            result.append(byte_seq[i])
            i += 1
    return tuple(result)


def _update_byte_sequence_with_bp(
    byte_sequence_counts: ByteSequenceCounts, bp: BytePair
) -> ByteSequenceCounts:
    return {_replace_bps_in_bseq(k, bp): v for k, v in byte_sequence_counts.items()}


def _pretokenized_word_iter(
    input: str, special_tokens: set[str], include_special_tokens=False
) -> Iterable[ByteSequence]:
    PRETOKEN_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def split_on_special_token(inputs: Iterable[str], token: str) -> Iterable[str]:
        if include_special_tokens:
            return chain.from_iterable(
                regex.split(f"({regex.escape(token)})", input) for input in inputs
            )
        return chain.from_iterable(input.split(token) for input in inputs)

    input_splits: Iterable[str] = [input]
    for token in special_tokens:
        input_splits = split_on_special_token(input_splits, token)
    for input_split in input_splits:
        if include_special_tokens and input_split in special_tokens:
            yield (input_split.encode("utf-8"),)
        else:
            for match in regex.finditer(PRETOKEN_PATTERN, input_split):
                word = match.captures(0)[0]
                yield tuple(map(int.to_bytes, word.encode("utf-8")))


def _pretokenize_words(input: str, special_tokens: set[str]) -> dict[ByteSequence, int]:
    pretoken_counts: dict[ByteSequence, int] = defaultdict(int)
    for word in _pretokenized_word_iter(input, special_tokens):
        pretoken_counts[word] += 1
    return pretoken_counts


def _merge_byte_sequence_counts(
    one: ByteSequenceCounts, two: ByteSequenceCounts
) -> ByteSequenceCounts:
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
                byte_sequence_counts,
                _pretokenize_words(chunk_data.decode("utf-8"), set(special_tokens)),
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


def _bps_from_bseq(bseq: ByteSequence) -> list[BytePair]:
    return [(first, second) for first, second in zip(bseq[0:-1], bseq[1:])]


def _bps_in_bseq(bseq: ByteSequence) -> set[BytePair]:
    return {(first, second) for first, second in zip(bseq[0:-1], bseq[1:])}


def _count_bps_in_bseq(bseq: ByteSequence) -> dict[BytePair, int]:
    bp_counts = defaultdict(int)
    for bp in _bps_from_bseq(bseq):
        bp_counts[bp] += 1
    return bp_counts


class Word:
    def __init__(self, word: ByteSequence) -> None:
        self.word: ByteSequence = word
        self.bp_counts = _count_bps_in_bseq(word)

    def _bp_at(self, idx: int) -> BytePair:
        return (self.word[idx], self.word[idx + 1])

    def replace_bp(self, bp: BytePair) -> tuple[dict[BytePair, int], set[BytePair], set[BytePair]]:
        """Join any instances of the given BytePair. Return a dictionary
        containing the computed differences, a tuple of removed byte pairs,
        and a tuple of added byte pairs."""
        new_word = _replace_bps_in_bseq(self.word, bp)
        self.word = tuple(new_word)
        bp_diffs = defaultdict(int)
        old_bps = self.bp_counts.keys()
        for bp, count in self.bp_counts.items():
            bp_diffs[bp] -= count
        self.bp_counts = _count_bps_in_bseq(new_word)
        new_bps = self.bp_counts.keys()
        for bp, count in self.bp_counts.items():
            bp_diffs[bp] += count
        return (bp_diffs, old_bps - new_bps, new_bps - old_bps)


class BpeState:
    def __init__(self) -> None:
        self.ids_by_word: dict[ByteSequence, int] = {}
        self.words_by_id: dict[int, Word] = {}
        self.word_counts_by_id: dict[int, int] = defaultdict(int)
        self.next_word_id = 0
        self.word_ids_by_bp: dict[BytePair, set[int]] = defaultdict(set)
        self.bp_counts: dict[BytePair, int] = defaultdict(int)
        self.merge_list = list[BytePair]

    def add_word(self, word: ByteSequence, count: int = 1) -> None:
        word_id = self.ids_by_word.get(word)
        if word_id is None:
            word_id = self.next_word_id
            self.ids_by_word[word] = word_id
            self.words_by_id[word_id] = Word(word)
            self.next_word_id += 1
        self.word_counts_by_id[word_id] += count

    def compute_initial_bp_counts(self) -> None:
        assert len(self.bp_counts) == 0 and len(self.word_ids_by_bp) == 0
        for word_id, word_count in self.word_counts_by_id.items():
            word = self.words_by_id[word_id]
            for bp, bp_count in word.bp_counts.items():
                self.word_ids_by_bp[bp].add(word_id)
                self.bp_counts[bp] += bp_count * word_count

    def compute_next_bp(self) -> BytePair | None:
        if len(self.bp_counts) == 0:
            return None
        next_bp, _ = max(self.bp_counts.items(), key=lambda x: (x[1], x[0]))
        bp_word_ids = list(self.word_ids_by_bp[next_bp])
        for word_id in bp_word_ids:
            word_count = self.word_counts_by_id[word_id]
            word_bp_counts, removed_bps, added_bps = self.words_by_id[word_id].replace_bp(next_bp)
            for bp, bp_count in word_bp_counts.items():
                new_count = self.bp_counts[bp] + bp_count * word_count
                if new_count == 0:
                    del self.bp_counts[bp]
                else:
                    self.bp_counts[bp] = new_count
            for removed_bp in removed_bps:
                self.word_ids_by_bp[removed_bp].remove(word_id)
            for added_bp in added_bps:
                self.word_ids_by_bp[added_bp].add(word_id)
        return next_bp


def pretokenize_chunk(
    input_path: str, start: int, end: int, special_tokens
) -> dict[ByteSequence, int]:
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        chunk_data = input_file.read(end - start)
    return _pretokenize_words(chunk_data.decode("utf-8"), special_tokens)


def run_nboy_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, Sequence[BytePair]]:
    vocab: dict[int, bytes] = _initialize_vocabulary(special_tokens)
    if len(vocab) > vocab_size:
        ValueError("vocab_size is too low.")
    boundaries = []
    bpe_state = BpeState()
    num_cpus = multiprocessing.cpu_count()
    with open(input_path, "rb") as input_file:
        boundaries: list[int] = _find_chunk_boundaries(input_file, num_cpus * 10, ENDOFTEXT)
    with multiprocessing.Pool(num_cpus) as pool:
        for result in pool.starmap(
            pretokenize_chunk,
            [
                (input_path, start, end, special_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ],
        ):
            for word, count in result.items():
                bpe_state.add_word(word, count)
    merge_list = []
    bpe_state.compute_initial_bp_counts()
    for _ in range(vocab_size - len(vocab)):
        bp = bpe_state.compute_next_bp()
        if bp is None:
            break
        merge_list.append(bp)
    for bp in merge_list:
        merged_token = bp[0] + bp[1]
        vocab[len(vocab)] = merged_token
    return (vocab, merge_list)


class Tokenizer:
    def __init__(self, vocab: Vocab, merges: MergeList, special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab: Vocab = vocab
        self.rvocab: dict[bytes, int] = dict((v, k) for (k, v) in vocab.items())
        self.merges: MergeList = merges
        self.mergemap: dict[bytes, set[bytes]] = defaultdict(set)
        for first, second in self.merges:
           self.mergemap[first].add(second)
        if special_tokens is None:
            self.special_tokens: set[str] = set()
        else:
            self.special_tokens: set[str] = set(special_tokens)

    def _encode_byte_sequence(self, tokens: Iterable[ByteSequence]) -> Iterable[int]:
        def apply_mergelist(current_bytes: bytes, remaining_bytes: ByteSequence):
            if remaining_bytes and remaining_bytes[0] in self.mergemap[current_bytes]:
                return apply_mergelist(current_bytes + remaining_bytes[0], remaining_bytes[1:])
            return current_bytes, remaining_bytes
        for token in tokens:
            current_bytes, *remaining_bytes = token
            while True:
                current_bytes, remaining_bytes = apply_mergelist(current_bytes, remaining_bytes)
                yield self.rvocab[current_bytes]
                if not remaining_bytes:
                    break
                current_bytes, *remaining_bytes = remaining_bytes

    def encode(self, text: str) -> list[int]:
        return list(
            self._encode_byte_sequence(
                _pretokenized_word_iter(
                    text, self.special_tokens, include_special_tokens=True)))

    def encode_iter(self, iterable: Iterable[str]) -> Iterable[int]:
        for word in iterable:
            for encoding in self._encode_byte_sequence(
                _pretokenized_word_iter(
                    word, self.special_tokens, include_special_tokens=True)):
                yield encoding

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[i] for i in ids)
        return decoded_bytes.decode("utf-8")

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merge_list_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_path, "rb") as vf:
            vocab: Vocab = pickle.load(vf)
        with open(merge_list_path, "rb") as mlf:
            merge_list: MergeList = pickle.load(mlf)
        return cls(vocab, merge_list, special_tokens or list())
