import io
import itertools
import os

from cs336_basics import bpe
from cs336_basics import common_types

from .common import FIXTURES_PATH


def byte_seq(word: str) -> common_types.ByteSequence:
    return tuple(map(int.to_bytes, word.encode("utf-8")))


def test_find_chunk_boundaries_basic():
    # Create a bytes buffer with special tokens
    data = b"abc<|endoftext|>def<|endoftext|>ghi"
    file = io.BytesIO(data)
    boundaries = bpe._find_chunk_boundaries(file, 2, bpe.ENDOFTEXT)
    # Should find boundaries at the special token positions
    assert boundaries[0] == 0
    assert boundaries[-1] == len(data)
    assert all(b <= len(data) for b in boundaries)


def test_initialize_vocabulary_adds_special_tokens():
    vocab = bpe._initialize_vocabulary(["<|endoftext|>", "<SPECIAL>"])
    # Should contain 256 initial bytes + special tokens
    assert b"<|endoftext|>" in vocab.values()
    assert b"<SPECIAL>" in vocab.values()
    assert len(vocab) >= 258


def test_sennrich_compute_next_bp_returns_none_for_empty():
    result = bpe._sennrich_compute_next_bp({})
    assert result is None


def test_sennrich_compute_next_bp_finds_most_frequent():
    # Two words: ab, ab, bc
    word_counts = {
        (b"a", b"b"): 2,
        (b"b", b"c"): 1,
    }
    bp = bpe._sennrich_compute_next_bp(word_counts)
    assert bp == (b"a", b"b")


def test_update_byte_sequence_with_bp_merges_pairs():
    word_counts = {
        (b"a", b"b", b"c"): 1,
        (b"a", b"b"): 1,
    }
    bp = (b"a", b"b")
    updated = bpe._update_byte_sequence_with_bp(word_counts, bp)
    # Should merge a+b into ab
    assert (b"a" + b"b", b"c") in updated
    assert (b"a" + b"b",) in updated


def test_pretokenize_words_counts_words():
    text = "hello world hello hello"
    counts = bpe._pretokenize_words(text, [])
    # Should count "hello" twice, "world" once
    hello_token = byte_seq("hello")
    hello_token_with_space = byte_seq(" hello")
    world_token = byte_seq(" world")
    assert counts[hello_token] == 1
    assert counts[hello_token_with_space] == 2
    assert counts[world_token] == 1


def test_pretokenize_words_honors_special_tokens():
    text = "hello world <a> foo <a> foo <b> bar <b>"
    counts = bpe._pretokenize_words(text, ["<a>", "<b>"])
    hello_token = byte_seq("hello")
    world_token = byte_seq(" world")
    foo_token = byte_seq(" foo")
    bar_token = byte_seq(" bar")
    space_token = byte_seq(" ")
    assert counts == {hello_token: 1, world_token: 1, foo_token: 2, bar_token: 1, space_token: 4}


def test_merge_byte_sequence_counts_merges_counts():
    one = {("a",): 1, ("b",): 2}
    two = {("a",): 2, ("c",): 3}
    merged = bpe._merge_byte_sequence_counts(one, two)
    assert merged[("a",)] == 3
    assert merged[("b",)] == 2
    assert merged[("c",)] == 3


def test_sennheiser_example():
    expected_vocab = bpe._initialize_vocabulary(["<|endoftext|>"])
    expected_merge_list = [b"st", b"est", b"ow", b"low", b"west", b"ne"]
    for i, bp in zip(itertools.count(len(expected_vocab)), expected_merge_list):
        expected_vocab[i] = bp
    vocab_size = 263

    input_path = FIXTURES_PATH / "sennheiser_example.txt"
    vocab, merge_list = bpe.run_sennrich_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    assert (len(vocab)) == vocab_size
    assert set(expected_vocab.keys()) == set(vocab.keys())
    assert set(expected_vocab.values()) == set(vocab.values())
    assert set([a + b for (a, b) in merge_list]) == set(expected_merge_list)
