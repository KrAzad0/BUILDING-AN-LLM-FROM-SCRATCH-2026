class ByteTokenizer:
    """A minimal reversible UTF-8 byte tokenizer.

    Every byte is a token, so the vocabulary size is always 256 and no
    tokenizer training step is required.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(int(i) % 256 for i in token_ids)
        return data.decode("utf-8", errors="replace")
