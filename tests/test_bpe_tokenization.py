import io
import itertools
import os

import pytest

from cs336_basics import bpe
from cs336_basics import common_types
from cs336_basics.token_utils import save_vocab_and_merges

from .common import FIXTURES_PATH
import pickle


def byte_seq(word: str) -> common_types.ByteSequence:
    return tuple(map(int.to_bytes, word.encode("utf-8")))


def byte_pair(a, b) -> common_types.BytePair:
    return (a.encode("utf-8"), b.encode("utf-8"))


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
    counts = bpe._pretokenize_words(text, {"<a>", "<b>"})
    hello_token = byte_seq("hello")
    world_token = byte_seq(" world")
    foo_token = byte_seq(" foo")
    bar_token = byte_seq(" bar")
    space_token = byte_seq(" ")
    assert counts == {
        hello_token: 1,
        world_token: 1,
        foo_token: 2,
        bar_token: 1,
        space_token: 4,
    }


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


def test_word_initialization_and_bp_counts():
    w = bpe.Word((b"a", b"b", b"c", b"d"))
    # Should count ab, bc, cd
    expected = {
        (b"a", b"b"): 1,
        (b"b", b"c"): 1,
        (b"c", b"d"): 1,
    }
    assert w.bp_counts == expected


def test_word_bp_at_returns_correct_pair():
    w = bpe.Word((b"x", b"y", b"z"))
    assert w._bp_at(0) == (b"x", b"y")
    assert w._bp_at(1) == (b"y", b"z")


def test_word_replace_bp_merges_and_updates_counts():
    w = bpe.Word((b"a", b"b", b"c", b"b", b"c"))
    bp = (b"b", b"c")
    bp_diffs, removed_bps, added_bps = w.replace_bp(bp)
    # After merging, should have (b"a", b"bc", b"bc")
    assert w.word == (b"a", b"bc", b"bc")
    # bp_diffs should reflect removal of (b"b", b"c") and addition of (b"bc", b"bc")
    assert bp_diffs == {
        byte_pair("a", "bc"): 1,
        byte_pair("bc", "bc"): 1,
        byte_pair("a", "b"): -1,
        byte_pair("c", "b"): -1,
        byte_pair("b", "c"): -2,
    }
    assert removed_bps == {byte_pair("a", "b"), byte_pair("b", "c"), byte_pair("c", "b")}
    assert added_bps == {byte_pair("a", "bc"), byte_pair("bc", "bc")}


def test_word_replace_bp_no_merge_if_not_present():
    w = bpe.Word((b"a", b"b", b"c"))
    bp = (b"x", b"y")
    bp_diffs, removed_bps, added_bps = w.replace_bp(bp)
    # No merge should happen
    assert w.word == (b"a", b"b", b"c")
    # bp_diffs should be zero for all pairs
    assert all(v == 0 for v in bp_diffs.values())
    assert removed_bps == set()
    assert added_bps == set()


def test_bpe_state_add_word_and_counts():
    state = bpe.BpeState()
    word1 = (b"a", b"b", b"c")
    word2 = (b"d", b"e")
    state.add_word(word1, count=2)
    state.add_word(word2, count=1)
    # Should assign unique IDs and count correctly
    assert state.ids_by_word[word1] != state.ids_by_word[word2]
    assert state.word_counts_by_id[state.ids_by_word[word1]] == 2
    assert state.word_counts_by_id[state.ids_by_word[word2]] == 1
    assert isinstance(state.words_by_id[state.ids_by_word[word1]], bpe.Word)


def test_bpe_state_compute_initial_bp_counts():
    state = bpe.BpeState()
    word1 = (b"a", b"b", b"c")
    word2 = (b"b", b"c", b"d")
    state.add_word(word1, count=1)
    state.add_word(word2, count=1)
    state.compute_initial_bp_counts()
    # Should count all byte pairs in both words
    expected_bps = {(b"a", b"b"), (b"b", b"c"), (b"c", b"d")}
    assert set(state.bp_counts.keys()) == expected_bps
    assert state.bp_counts[(b"b", b"c")].count == 2
    assert state.bp_counts[(b"a", b"b")].count == 1
    assert state.bp_counts[(b"c", b"d")].count == 1
    # Should map bps to word IDs
    for bp in expected_bps:
        assert isinstance(state.word_ids_by_bp[bp], set)
        assert len(state.word_ids_by_bp[bp]) >= 1


def test_bpe_state_compute_next_bp_merges():
    state = bpe.BpeState()
    word1 = (b"a", b"b", b"b")
    word2 = (b"a", b"b")
    state.add_word(word1, count=1)
    state.add_word(word2, count=1)
    state.compute_initial_bp_counts()
    # The most frequent bp is (b"a", b"b")
    next_bp = state.compute_next_bp()
    assert next_bp == (b"a", b"b")
    # After merge, words should be updated
    assert {w.word for w in state.words_by_id.values()} == {(b"ab", b"b"), (b"ab",)}


def test_bpe_state_multiple_merges():
    state = bpe.BpeState()
    word = (b"x", b"y", b"z", b"y", b"z")
    state.add_word(word, count=1)
    state.compute_initial_bp_counts()
    # First merge: (b"y", b"z")
    bp1 = state.compute_next_bp()
    assert bp1 == (b"y", b"z")
    # After merge, word should contain b"yz"
    merged_word = state.words_by_id[state.ids_by_word[word]].word
    assert b"y" + b"z" in merged_word
    # Second merge: (b"yz", b"yz")
    bp2 = state.compute_next_bp()
    assert bp2 == (b"y" + b"z", b"y" + b"z")
    merged_word2 = state.words_by_id[state.ids_by_word[word]].word
    assert merged_word2 == (b"x", b"y" + b"z" + b"y" + b"z")


def test_bpe_state_no_merge_when_no_pairs():
    state = bpe.BpeState()
    word = (b"a", b"b")
    state.add_word(word, count=1)
    state.compute_initial_bp_counts()
    assert state.compute_next_bp() == byte_pair("a", "b")
    # No byte pairs to merge
    # compute_next_bp will return None because bp_counts is empty
    assert state.compute_next_bp() is None


def test_nboy_example():
    expected_vocab = bpe._initialize_vocabulary(["<|endoftext|>"])
    expected_merge_list = [b"st", b"est", b"ow", b"low", b"west", b"ne"]
    for i, bp in zip(itertools.count(len(expected_vocab)), expected_merge_list):
        expected_vocab[i] = bp
    vocab_size = 263

    input_path = FIXTURES_PATH / "sennheiser_example.txt"
    vocab, merge_list = bpe.run_nboy_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    assert (len(vocab)) == vocab_size
    assert set(expected_vocab.keys()) == set(vocab.keys())
    assert set(expected_vocab.values()) == set(vocab.values())
    assert set([a + b for (a, b) in merge_list]) == set(expected_merge_list)


def test_pretokenized_word_iter_with_special_tokens():
    text = "foo<SPECIAL> bar<SPECIAL> baz(SPECIAL2) quux"
    expected = [
        byte_seq("foo"),
        byte_seq(" bar"),
        byte_seq(" baz"),
        byte_seq(" quux"),
    ]
    expected_with_tokens = [
        byte_seq("foo"),
        (b"<SPECIAL>",),
        byte_seq(" bar"),
        (b"<SPECIAL>",),
        byte_seq(" baz"),
        (b"(SPECIAL2)",),
        byte_seq(" quux"),
    ]
    assert list(bpe._pretokenized_word_iter(text, {"<SPECIAL>", "(SPECIAL2)"})) == expected
    assert (
        list(bpe._pretokenized_word_iter_with_special_tokens(text, {"<SPECIAL>", "(SPECIAL2)"}))
        == expected_with_tokens
    )


def test_tokenizer_encode_and_decode_basic():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"c",
        3: b"ab",
        4: b"bc",
    }
    merges = [(b"a", b"b"), (b"b", b"c")]
    tokenizer = bpe.Tokenizer(vocab, merges)
    text = "abc"
    encoded = tokenizer.encode(text)
    assert encoded == [3, 2]
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_tokenizer_with_special_tokens():
    vocab = {
        0: b"a",
        1: b"b",
        2: b"<SPECIAL>",
    }
    merges = []
    tokenizer = bpe.Tokenizer(vocab, merges, special_tokens=["<SPECIAL>"])
    text = "ab<SPECIAL>b"
    encoded = tokenizer.encode(text)
    assert encoded == [0, 1, 2, 1]
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_tokenizer_from_files(tmp_path):
    vocab = {0: b"a", 1: b"b", 2: b"ab"}
    merges = [(b"a", b"b")]
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"

    save_vocab_and_merges(vocab, merges, vocab_path=vocab_path, merges_path=merges_path)
    tokenizer = bpe.Tokenizer.from_files(vocab_path, merges_path)
    assert isinstance(tokenizer, bpe.Tokenizer)
    assert tokenizer.vocab == vocab
    assert tokenizer.merges == merges


def test_tokenizer_encode_empty_string():
    vocab = {0: b"a"}
    merges = []
    tokenizer = bpe.Tokenizer(vocab, merges)
    encoded = tokenizer.encode("")
    assert encoded == []
    decoded = tokenizer.decode(encoded)
    assert decoded == ""


def test_tokenizer_decode_unknown_id_raises():
    vocab = {0: b"a"}
    merges = []
    tokenizer = bpe.Tokenizer(vocab, merges)
    with pytest.raises(KeyError):
        tokenizer.decode([999])
