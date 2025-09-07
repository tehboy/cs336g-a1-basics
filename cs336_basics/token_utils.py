"""Common utilities for working with tokens."""

import functools
import json
import os
from collections.abc import Mapping

from cs336_basics.common_types import MergeList, Vocab


@functools.lru_cache
def gpt2_bytes_to_unicode() -> Mapping[int, str]:
    """Returns a mapping between every possible byte (an integer from 0 to 255)
    to a printable unicode string character representation. This function is
    taken from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns
    `Ā`. The bytes that are visually printable keep their original string
    representation [1]. For example, `chr(33)` returns `!`, and so accordingly
    `d[33]` returns `!`. Note in particular that the space character `chr(32)`
    becomes `d[32]`, which returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer
    representing the Unicode code point of that character (returned by the
    Python `ord`) function and shifts it by 256. For example, `ord(" ")`
    returns `32`, so the the space character ' ' is shifted to `256 + 32`.
    Since `chr(256 + 32)` returns `Ġ`, we use that as the string representation
    of the space.

    This function can simplify the BPE implementation and makes it slightly
    easier to manually inspect the generated merges after they're serialized to
    a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or
    # control characters. See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need
    # shifting each will get mapped chr(256 + n), where n will grow from 0...67
    # in the loop Get printable representations of the remaining integers 68
    # integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters, strict=True))
    return d


def save_vocab_and_merges(
    vocab: Vocab,
    merges: MergeList,
    *,
    vocab_path: str,
    merges_path: str,
) -> None:
    """
    Saves the vocabulary and merges to disk in a human-readable format.

    Args:
        vocab: A dictionary mapping token IDs to byte sequences.
        merges: A list of merged byte pairs.
        vocab_path: The file path to save the vocabulary JSON file.
        merges_path: The file path to save the merges text file.
    """
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Convert the byte tokens in the vocab to unicode strings for serialization.
    # The vocab is saved as {string_token: id}.
    string_vocab = {
        "".join([byte_to_unicode[b] for b in byte_token]): k for k, byte_token in vocab.items()
    }

    # Convert the byte pairs in the merges list to space-separated strings.
    string_merges = [
        f"{''.join([byte_to_unicode[b] for b in merge[0]])} {''.join([byte_to_unicode[b] for b in merge[1]])}"
        for merge in merges
    ]

    # Save the vocabulary to a JSON file.
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(string_vocab, f, ensure_ascii=False, indent=2)

    # Save the merges to a text file, one merge per line.
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("\n".join(string_merges))


def load_vocab_and_merges(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[Vocab, MergeList]:
    """Loads a vocabulary and merge list from files.

    Args:
        vocab_path: The file path to the vocabulary JSON file.
        merges_path: The file path to the merges text file.

    Returns:
        A tuple containing the vocabulary and the merge list, with all tokens
        parsed back into bytes.
    """
    # Create the reverse mapping from unicode characters back to bytes.
    unicode_to_byte = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    # Load the string-based vocabulary from the JSON file.
    with open(vocab_path, encoding="utf-8") as vocab_f:
        string_vocab = json.load(vocab_f)

    # Convert the string tokens back into byte sequences.
    # The vocabulary is loaded as {id: byte_token}.
    vocab = {
        index: bytes([unicode_to_byte[char] for char in token])
        for token, index in string_vocab.items()
    }

    # Load the string-based merges from the text file.
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.strip()
            if cleaned_line:
                p1_str, p2_str = cleaned_line.split(" ")
                p1_bytes = bytes([unicode_to_byte[char] for char in p1_str])
                p2_bytes = bytes([unicode_to_byte[char] for char in p2_str])
                merges.append((p1_bytes, p2_bytes))

    return vocab, merges
