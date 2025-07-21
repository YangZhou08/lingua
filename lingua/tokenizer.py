# Copyright (c) Meta Platforms, Inc. and affiliates.

import abc
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import os

from sentencepiece import SentencePieceProcessor
import tiktoken
from tiktoken.load import load_tiktoken_bpe 
from typing import Dict, Collection, Union, Literal, AbstractSet, Sequence, Iterator, cast 

logger = logging.getLogger(__name__)


@dataclass
class TokenizerArgs:
    name: str = "bytes"
    path: Optional[str] = None


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, tokens, add_bos, add_eos):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class MockTokenizer(Tokenizer):
    n_words: int = 256

    def encode(self, tokens, add_bos, add_eos):
        return tokens


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens: List[int]):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path: str) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: List[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets


DEFAULT_TIKTOKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DEFAULT_TIKTOKEN_SPECIAL_TOKENS = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|fim_prefix|>": 2,
    "<|fim_middle|>": 3,
    "<|fim_end_fill|>": 253,
    "<|fim_pad|>": 254,
    "<|fim_suffix|>": 255,
}
TIKTOKEN_MAX_ENCODE_CHARS = 400_000


class TikTokenTokenizer(Tokenizer):

    def __init__(self, model_path: str) -> None:
        mergeable_ranks = load_tiktoken_bpe(model_path)
        all_special_tokens_with_ids = copy(DEFAULT_TIKTOKEN_SPECIAL_TOKENS)
        missing_ids = set(range(256)) - set(all_special_tokens_with_ids.values())
        for id in missing_ids:
            all_special_tokens_with_ids[f"<|reserved_special_token_{id}|>"] = id
        for name in all_special_tokens_with_ids:
            all_special_tokens_with_ids[name] += len(mergeable_ranks)

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=DEFAULT_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=all_special_tokens_with_ids,
        )

        self.bos_id: int = self.tkt_model.encode_single_token("<|begin_of_text|>")
        self.eos_id: int = self.tkt_model.encode_single_token("<|end_of_text|>")

        self.n_words: int = self.tkt_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert isinstance(s, str)

        subs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(s[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return (
            [self.bos_id] * add_bos
            + sum(self.tkt_model.encode_ordinary_batch(subs), start=[])
            + [self.eos_id] * add_eos
        )

    def decode(self, tokens: List[int]):
        return self.tkt_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is not None:
            token_bytes = self.tkt_model.decode_tokens_bytes(tokens)
        else:
            token_bytes = self.tkt_model.decode_tokens_bytes(
                self.tkt_model.encode(text, allowed_special="all")
            )

        text_len, offsets = 0, []
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        substrs = [text[s:e] for s, e in zip(offsets, offsets[1:] + [None])]
        return substrs, offsets

class Llama3Tokenizer(Tokenizer): 
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:] 
    
    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        raise NotImplementedError("get_token_offsets is not implemented for Llama3Tokenizer") 

def build_tokenizer(name: str, path: Optional[str] = None) -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    elif name == "mock":
        return MockTokenizer()
    elif name == "sp":
        return SentencePieceTokenizer(path)
    elif name == "tiktoken":
        return TikTokenTokenizer(path) 
    elif name == "llama3": 
        return Llama3Tokenizer(path) 
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
