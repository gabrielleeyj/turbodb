"""Tests for config parsing and package import (no GPU required)."""

import pytest

import turboquant_kv
from turboquant_kv import KVConfig, is_turboquant_dtype, parse_dtype
from turboquant_kv.config import DEFAULT_PAGE_SIZE


def test_package_imports_without_cuda():
    # The package must import cleanly in a CUDA-less venv.
    assert turboquant_kv.__version__ == "0.1.0"
    # is_available() is False here but must not raise.
    assert isinstance(turboquant_kv.is_available(), bool)


def test_parse_dtype():
    cfg = parse_dtype("turboquant3", d_head=128)
    assert cfg == KVConfig(d_head=128, b_key=3, b_val=3, page_size=DEFAULT_PAGE_SIZE)


def test_parse_dtype_custom_page():
    cfg = parse_dtype("turboquant4", d_head=64, page_size=32)
    assert cfg.page_size == 32
    assert cfg.b_key == 4


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("turboquant3", True),
        ("turboquant4", True),
        ("turboquant8", True),
        ("turboquant0", False),
        ("turboquant9", False),
        ("fp8", False),
        ("turboquant", False),
        ("turboquantX", False),
    ],
)
def test_is_turboquant_dtype(dtype, expected):
    assert is_turboquant_dtype(dtype) is expected


def test_parse_dtype_rejects_non_turboquant():
    with pytest.raises(ValueError):
        parse_dtype("fp8", d_head=128)


def test_kvconfig_validation():
    with pytest.raises(ValueError):
        KVConfig(d_head=0, b_key=4, b_val=4)
    with pytest.raises(ValueError):
        KVConfig(d_head=128, b_key=0, b_val=4)
    with pytest.raises(ValueError):
        KVConfig(d_head=128, b_key=4, b_val=9)
    with pytest.raises(ValueError):
        KVConfig(d_head=128, b_key=4, b_val=4, page_size=0)
