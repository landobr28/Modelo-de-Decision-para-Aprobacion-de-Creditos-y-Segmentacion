# -*- coding: utf-8 -*-
"""
Scorecard de puntos (Weight of Evidence + PDO)
==============================================

Transforma el modelo logístico en un scorecard de crédito clásico:

* **Fine classing**: agrupa cada variable en tramos/categorías.
* **Weight of Evidence (WoE)** e **Information Value (IV)** por tramo.
* Regresión logística sobre los WoE (1 coeficiente por variable).
* Conversión a **puntos** con la convención PDO (points to double odds):
  cada +20 puntos duplica las probabilidades de ser Bueno.

El score final se define como

    score = offset - factor * ln(odds_Malo)

de modo que un score MÁS ALTO implica MENOR riesgo.
"""

import numpy as np
import pandas as pd

from . import config


# ----------------------------------------------------------------------
# Binning
# ----------------------------------------------------------------------
def quantile_bins(x, n_bins=10, min_samples=30):
    """Divide un vector continuo en n_bins tramos de igual población."""
    x = pd.Series(x)
    x_clean = x.dropna()
    if x_clean.nunique() <= 1:
        return [0.0]
    if x_clean.nunique() <= n_bins:
        return sorted(x_clean.unique())[:-1]
    edges = x_clean.quantile(np.linspace(0, 1, n_bins + 1)).unique()
    return list(edges[1:-1])  # cortes interiores


def woe_iv(bad, x, bins):
    """WoE e IV por tramo de una variable binarizada en `bins`. (docstring)"""
    edges = [-np.inf] + list(bins) + [np.inf]
    df = pd.DataFrame({"bad": bad, "x": x})
    df["tramo"] = pd.cut(df["x"], bins=edges, right=True, include_lowest=True)
    return _woe_frame(df)


def scorecard_table(X, y, features, n_bins=8):
    """Construye la tabla WoE/IV para todas las variables del scorecard."""
    rows = []
    var_iv = {}
    for f in features:
        if X[f].dtype.kind == "O" or X[f].nunique() <= 8:
            tbl = _woe_cat(y, X[f], sorted(X[f].dropna().unique()))
        else:
            bins = quantile_bins(X[f], n_bins=n_bins)
            tbl = _woe_num(y, X[f], bins)
        iv = float(tbl["iv"].sum())
        var_iv[f] = iv
        for _, r in tbl.iterrows():
            rows.append({
                "variable": f,
                "tramo": r["tramo"],
                "n": int(r["n"]),
                "n_bad": int(r["n_bad"]),
                "pct_bad": r["pct_bad"],
                "pct_good": r["pct_good"],
                "woe": r["woe"],
                "iv_componente": r["iv"],
            })
    table = pd.DataFrame(rows)
    table["IV_total"] = table["variable"].map(var_iv)
    return table.reset_index(drop=True)


def _woe_num(y, x, bins):
    df = pd.DataFrame({"bad": y, "x": x})
    edges = [-np.inf] + list(bins) + [np.inf]
    cats = pd.cut(df["x"], bins=edges, right=True, include_lowest=True)
    df["tramo"] = cats.cat.rename_categories(
        lambda c: f"[{c.left:.2g} , {c.right:.2g})")
    return _woe_frame(df)


def _woe_cat(y, x, cats):
    df = pd.DataFrame({"bad": y, "x": x})
    df["tramo"] = df["x"].fillna("MISSING").astype(str)
    return _woe_frame(df)


def _woe_frame(df):
    tot_bad = df["bad"].sum()
    tot_good = len(df) - tot_bad
    tbl = df.groupby("tramo", observed=True).agg(
        n=("bad", "size"), n_bad=("bad", "sum")
    ).reset_index()
    tbl["pct_bad"] = tbl["n_bad"] / tot_bad
    tbl["pct_good"] = (tbl["n"] - tbl["n_bad"]) / tot_good
    tbl["woe"] = np.log((tbl["pct_bad"] + 1e-9) / (tbl["pct_good"] + 1e-9))
    tbl["iv"] = (tbl["pct_bad"] - tbl["pct_good"]) * tbl["woe"]
    return tbl


# ----------------------------------------------------------------------
# Scorecard
# ----------------------------------------------------------------------
class Scorecard:
    """Scorecard listo para puntos.

    Parámetros (definidos en config):
      * PDO: puntos para duplicar odds
      * odds_at_offset: odds en el score base
      * score_offset: score de referencia
    """

    def __init__(self, table, betas, intercept, features):
        self.table = table            # tabla WoE/IV
        self.betas = betas            # dict variable -> beta
        self.intercept = intercept
        self.features = features
        self._offset = config.SCORE_OFFSET
        self._odds0 = config.ODDS_AT_OFFSET
        self._pdo = config.PDO
        self.factor = self._pdo / np.log(2)

    def points_table(self):
        """Tabla de puntos por tramo de cada variable."""
        rows = []
        for _, r in self.table.iterrows():
            var = r["variable"]
            beta = self.betas.get(var, 0.0)
            pt = -self.factor * beta * r["woe"]
            base = self._offset - self.factor * self.intercept
            rows.append({
                "variable": var,
                "tramo": r["tramo"],
                "woe": r["woe"],
                "puntos": round(pt, 1),
                "puntos_acumulado": round(base + pt, 1),
            })
        pts = pd.DataFrame(rows)
        base_points = self._offset - self.factor * self.intercept
        pts["puntos_base"] = base_points
        return pts.sort_values(["variable", "tramo"])

    def score_from_woe(self, woe_matrix):
        """score = offset - factor*(intercept + sum(beta*WoE))"""
        linear = self.intercept + sum(
            self.betas[f] * woe_matrix[f] for f in self.features
        )
        return self._offset - self.factor * linear
