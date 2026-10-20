"""Pins the binning conventions the production SQL depends on."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lcscore.binning import CategoricalBinner, NumericBinner  # noqa: E402


def _data(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(600, 800, n).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp((x - 640) / 25))).astype(float)
    x[:300] = np.nan
    return pd.Series(x), pd.Series(y)


def test_right_closed_edges():
    x, y = _data()
    b = NumericBinner("x").fit(x, y)
    edge = b.edges[0]
    idx = b.bin_index(pd.Series([edge, np.nextafter(edge, np.inf)]))
    assert idx[0] == 0 and idx[1] == 1          # lo < x <= hi


def test_monotone_woe_and_min_share():
    x, y = _data()
    b = NumericBinner("x", min_share=0.05).fit(x, y)
    rng = b.table[b.table.type == "range"]
    assert (np.diff(rng.woe) > 0).all()          # risk falls as x rises
    assert (rng["count"] / len(x) >= 0.05).all()


def test_missing_and_unseen():
    x, y = _data()
    b = NumericBinner("x").fit(x, y)
    assert b.bin_index(pd.Series([np.nan]))[0] == len(b.table) - 1
    c = CategoricalBinner("c", min_share=0.05).fit(
        pd.Series(["a"] * 500 + ["b"] * 500 + [None] * 100), pd.Series([0, 1] * 550))
    worst = int(np.argmin(c.table.woe))
    assert c.bin_index(pd.Series(["never_seen"]))[0] == worst
