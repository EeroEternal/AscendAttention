"""The decode kernel's per-plan key must move with every field that changes the generated code.

Scope is deliberately narrow, because the mechanism is narrower than "a global kernel cache": the key
guards **one plan's** ``backend_state`` (does the built kernel still describe this call?).  Overclaiming
here would be worse than not testing: a future reader would believe cross-plan reuse is safe.

The interesting assertions are therefore (a) an identical description reuses, (b) each code-generating
field forces a rebuild, and (c) the capacity is quantised, so a pool that grows inside a bucket does not.
"""

from __future__ import annotations

import pytest

from ascend_attention.abi import declared_pool_cap
from ascend_attention.backends.tilelang_ascend950 import decode_spec_key


def _key(**overrides):
    base = dict(
        batch_size=8,
        num_kv_heads=2,
        gqa_group_size=4,
        head_dim=128,
        page_size=128,
        pool_cap=declared_pool_cap(1024),
    )
    base.update(overrides)
    return decode_spec_key(**base)


def test_identical_descriptions_share_a_key():
    assert _key() == _key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 16),
        ("num_kv_heads", 4),
        ("gqa_group_size", 8),
        ("head_dim", 256),
        ("page_size", 64),
        ("pool_cap", declared_pool_cap(4096)),
    ],
)
def test_every_field_moves_the_key(field, value):
    assert _key() != _key(**{field: value}), field


def test_capacity_is_quantised_so_a_growing_pool_does_not_recompile():
    assert declared_pool_cap(1000) == declared_pool_cap(1024)
    assert declared_pool_cap(512) < declared_pool_cap(1000)
