"""Monotonic weight-of-evidence binning, written out explicitly so every rule
can be re-implemented in SQL from the exported spec.

Conventions (shared with production/score_scorecard.sql)
  * WoE = ln(%good / %bad); higher WoE = lower risk. Counts get +0.5 smoothing.
  * Numeric bins are left-open, right-closed: lo < x <= hi; lo/hi None = -inf/+inf.
  * Numeric missing values get their own bin when they are at least
    `min_missing_share` of the sample; otherwise they take the riskiest bin's WoE.
  * Categorical nulls are coerced to the string "MISSING". Levels under
    `rare_share` pool into one group; unseen levels at scoring time ("__ELSE__")
    fall into the riskiest group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH = 0.5


def _woe(goods, bads, tot_g, tot_b):
    g = (np.asarray(goods, float) + SMOOTH) / (tot_g + SMOOTH)
    b = (np.asarray(bads, float) + SMOOTH) / (tot_b + SMOOTH)
    return np.log(g / b), (g - b) * np.log(g / b)


def _round_sig(x, sig=6):
    return float(f"{x:.{sig}g}")


class NumericBinner:
    def __init__(self, name, fine_bins=20, max_bins=6, min_share=0.05, min_missing_share=0.005):
        self.name, self.fine_bins, self.max_bins = name, fine_bins, max_bins
        self.min_share, self.min_missing_share = min_share, min_missing_share

    # edges = interior cut points c1 < c2 < ...; bins (-inf,c1], (c1,c2], ..., (ck, inf)
    @staticmethod
    def _assign(x, edges):
        return np.searchsorted(np.asarray(edges, float), x, side="left")

    def _stats(self, x, y, edges):
        idx = self._assign(x, edges)
        k = len(edges) + 1
        n = np.bincount(idx, minlength=k)
        b = np.bincount(idx, weights=y, minlength=k)
        return n, b

    def fit(self, x: pd.Series, y: pd.Series):
        x = x.to_numpy(float)
        y = y.to_numpy(float)
        miss = np.isnan(x)
        xv, yv = x[~miss], y[~miss]
        tot_b, tot_g = y.sum(), len(y) - y.sum()

        uniq = np.unique(xv)
        if len(uniq) <= self.fine_bins:
            edges = list(uniq[:-1])
        else:
            qs = np.quantile(xv, np.linspace(0, 1, self.fine_bins + 1)[1:-1], method="lower")
            edges = sorted(set(_round_sig(q) for q in qs))
            edges = [e for e in edges if e < uniq[-1]]

        # monotone direction from the rank correlation
        direction = np.sign(pd.Series(xv).corr(pd.Series(yv), method="spearman") or 1.0)

        def merge_once(edges, n, b):
            # 1. monotonicity violations: bad rate must move with `direction`
            rate = b / np.maximum(n, 1)
            for i in range(len(rate) - 1):
                if direction * (rate[i + 1] - rate[i]) < 0:
                    return edges[:i] + edges[i + 1:]
            # 2. undersized bins: merge with the neighbour of closer bad rate
            share = n / len(x)
            small = np.where(share < self.min_share)[0]
            if len(small) and len(edges):
                i = small[np.argmin(share[small])]
                if i == 0:
                    j = 0
                elif i == len(n) - 1:
                    j = i - 1
                else:
                    j = i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i
                return edges[:j] + edges[j + 1:]
            # 3. too many bins: merge the adjacent pair with closest bad rates
            if len(n) > self.max_bins:
                j = int(np.argmin(np.abs(np.diff(rate))))
                return edges[:j] + edges[j + 1:]
            return None

        while True:
            n, b = self._stats(xv, yv, edges)
            new = merge_once(edges, n, b)
            if new is None:
                break
            edges = new

        n, b = self._stats(xv, yv, edges)
        woe, iv = _woe(n - b, b, tot_g, tot_b)
        bounds = [None] + list(edges) + [None]
        rows = [{"feature": self.name, "type": "range", "lo": bounds[i], "hi": bounds[i + 1],
                 "category": None, "count": int(n[i]), "bads": int(b[i]),
                 "woe": float(woe[i]), "iv": float(iv[i])} for i in range(len(n))]
        n_miss, b_miss = int(miss.sum()), float(y[miss].sum())
        if n_miss / len(x) >= self.min_missing_share:
            mw, mi = _woe([n_miss - b_miss], [b_miss], tot_g, tot_b)
            rows.append({"feature": self.name, "type": "missing", "lo": None, "hi": None,
                         "category": None, "count": n_miss, "bads": int(b_miss),
                         "woe": float(mw[0]), "iv": float(mi[0])})
        else:
            worst = min(rows, key=lambda r: r["woe"])
            rows.append({"feature": self.name, "type": "missing", "lo": None, "hi": None,
                         "category": None, "count": n_miss, "bads": int(b_miss),
                         "woe": worst["woe"], "iv": 0.0, "note": "assigned riskiest bin"})
        self.edges, self.table = edges, pd.DataFrame(rows)
        self.direction = int(direction)
        return self

    def transform(self, x: pd.Series) -> np.ndarray:
        x = x.to_numpy(float)
        rng = self.table[self.table.type == "range"]
        out = rng.woe.to_numpy()[self._assign(np.nan_to_num(x, nan=0.0), self.edges)]
        miss_woe = self.table[self.table.type == "missing"].woe.iloc[0]
        return np.where(np.isnan(x), miss_woe, out)

    def bin_index(self, x: pd.Series) -> np.ndarray:
        """Row position in self.table for each value (missing -> last row)."""
        x = x.to_numpy(float)
        idx = self._assign(np.nan_to_num(x, nan=0.0), self.edges)
        return np.where(np.isnan(x), len(self.table) - 1, idx)


class CategoricalBinner:
    def __init__(self, name, max_bins=6, min_share=0.05, rare_share=0.01):
        self.name, self.max_bins, self.min_share, self.rare_share = name, max_bins, min_share, rare_share

    @staticmethod
    def clean(x: pd.Series) -> pd.Series:
        return x.astype("object").where(x.notna(), "MISSING").astype(str).str.strip()

    def fit(self, x: pd.Series, y: pd.Series):
        x = self.clean(x)
        tot_b, tot_g = y.sum(), len(y) - y.sum()
        stats = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["count", "sum"])
        rare = stats.index[stats["count"] / len(x) < self.rare_share]
        groups = [[c] for c in stats.index if c not in rare]
        if len(rare):
            groups.append(list(rare))

        def gstats(gs):
            n = np.array([stats.loc[g, "count"].sum() for g in gs], float)
            b = np.array([stats.loc[g, "sum"].sum() for g in gs], float)
            return n, b

        while True:
            n, b = gstats(groups)
            order = np.argsort(b / n)
            groups = [groups[i] for i in order]
            n, b = n[order], b[order]
            rate = b / n
            share = n / len(x)
            if len(groups) > 1 and share.min() < self.min_share:
                i = int(np.argmin(share))
                j = i + 1 if i == 0 else i - 1 if i == len(groups) - 1 else (
                    i - 1 if rate[i] - rate[i - 1] <= rate[i + 1] - rate[i] else i + 1)
            elif len(groups) > self.max_bins:
                i = int(np.argmin(np.diff(rate)))
                j = i + 1
            else:
                break
            lo, hi = sorted((i, j))
            groups = groups[:lo] + [groups[lo] + groups[hi]] + groups[hi + 1:]

        woe, iv = _woe(n - b, b, tot_g, tot_b)
        self.groups = groups
        self.mapping = {c: gi for gi, g in enumerate(groups) for c in g}
        self.else_group = int(np.argmin(woe))
        self.table = pd.DataFrame([{
            "feature": self.name, "type": "category", "lo": None, "hi": None,
            "category": "|".join(sorted(groups[i])), "count": int(n[i]), "bads": int(b[i]),
            "woe": float(woe[i]), "iv": float(iv[i])} for i in range(len(groups))])
        return self

    def bin_index(self, x: pd.Series) -> np.ndarray:
        return self.clean(x).map(self.mapping).fillna(self.else_group).astype(int).to_numpy()

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.table.woe.to_numpy()[self.bin_index(x)]
