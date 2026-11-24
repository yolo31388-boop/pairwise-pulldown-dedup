import pytest
import media.pulldown_dedup as mod

CASES = [
    ("adaptive_period", (1,)),
    ("tolerant_match", (1,)),
    ("field_order_meta", (1,)),
    ("timestamp_actual_rate", (1,)),
    ("continue_at_splice", (1,)),
    ("validate_output", (1,)),
    ("load", (1,)),
    ("detect", (1,)),
    ("dedup", (1,)),
    ("export", (1,)),
    ("validate_pulldown", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
