import math

import pytest

import mixer

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return mixer.Mixer()


A = [0.5, 0.5, 0.5]
B = [0.25, 0.25, 0.25]

CASES = [
        ("single", lambda: _m().mix([A], [1.0]) == A),
        ("two", lambda: _m().mix([A, B], [1.0, 1.0]) == [0.75, 0.75, 0.75]),
        ("half_gain", lambda: _m().mix([A, B], [0.5, 0.5]) == [0.375] * 3),
        ("clip", lambda: max(abs(v) for v in _m().mix([A, A, A], [1.0, 1.0, 1.0])) <= 1.0),
        ("pad", lambda: len(_m().mix([A, [0.1]], [1.0, 1.0])) == 3),
        ("pad_value", lambda: _m().mix([A, [0.1]], [1.0, 1.0])[2] == 0.5),
        ("peak", lambda: (lambda m: (m.mix([A], [1.0]), abs(m.peak() - 0.5) < 1e-9)[1])(_m())),
        ("normalize", lambda: max(abs(v) for v in _m().normalize([0.25, -0.5])) == 1.0),
        ("normalize_zero", lambda: _raises(lambda: _m().normalize([0.0, 0.0]))),
        ("bad_gain", lambda: _raises(lambda: _m().mix([A], [2.0]))),
        ("gain_len", lambda: _raises(lambda: _m().mix([A, B], [1.0]))),
        ("empty", lambda: _m().mix([], []) == []),
        ("range", lambda: all(-1.0 <= v <= 1.0 for v in _m().mix([A, A], [1.0, 1.0]))),
        ("deterministic", lambda: _m().mix([A, B], [1.0, 1.0]) == _m().mix([A, B], [1.0, 1.0])),
        ("type", lambda: isinstance(_m().mix([A], [1.0]), list)),
    ]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
