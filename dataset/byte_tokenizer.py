from __future__ import annotations


class ByteTokenizer:
    """A deterministic byte-level tokenizer for tutorial smoke tests.

    IDs:
    - 0: pad
    - 1: bos
    - 2: eos
    - 3..258: raw byte value + 3
    - 259: unk, kept for API symmetry
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 259
    vocab_size = 260

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = [b + 3 for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 3 <= idx <= 258:
                raw.append(idx - 3)
        return raw.decode("utf-8", errors="replace")
