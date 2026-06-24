"""Configuration types for TurboQuant KV-cache quantization."""

from __future__ import annotations

from dataclasses import dataclass

# Default number of tokens quantized together as one paged KV block.
DEFAULT_PAGE_SIZE = 16

# Supported kv_cache_dtype string prefix; the trailing digit is the bit width.
DTYPE_PREFIX = "turboquant"


@dataclass(frozen=True)
class KVConfig:
    """Per-head KV-cache quantization parameters.

    Attributes:
        d_head: Per-attention-head dimension.
        b_key: Bits per key coordinate.
        b_val: Bits per value coordinate.
        page_size: Tokens per paged KV block.
    """

    d_head: int
    b_key: int
    b_val: int
    page_size: int = DEFAULT_PAGE_SIZE

    def __post_init__(self) -> None:
        if self.d_head < 1:
            raise ValueError(f"d_head must be >= 1, got {self.d_head}")
        for name, b in (("b_key", self.b_key), ("b_val", self.b_val)):
            if not 1 <= b <= 8:
                raise ValueError(f"{name} must be in 1..8, got {b}")
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")


def parse_dtype(dtype: str, d_head: int, page_size: int = DEFAULT_PAGE_SIZE) -> KVConfig:
    """Parse a vLLM ``kv_cache_dtype`` string such as ``"turboquant3"`` into a
    :class:`KVConfig`. The trailing integer sets both key and value bit widths.

    Raises:
        ValueError: if the string is not a recognized TurboQuant dtype.
    """
    if not dtype.startswith(DTYPE_PREFIX):
        raise ValueError(
            f"{dtype!r} is not a TurboQuant kv_cache_dtype (expected '{DTYPE_PREFIX}<bits>')"
        )
    suffix = dtype[len(DTYPE_PREFIX):]
    try:
        bits = int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid bit width in dtype {dtype!r}") from exc
    return KVConfig(d_head=d_head, b_key=bits, b_val=bits, page_size=page_size)


def is_turboquant_dtype(dtype: str) -> bool:
    """Return True if ``dtype`` selects a TurboQuant KV-cache backend."""
    if not dtype.startswith(DTYPE_PREFIX):
        return False
    suffix = dtype[len(DTYPE_PREFIX):]
    return suffix.isdigit() and 1 <= int(suffix) <= 8
