# -*- coding: utf-8 -*-
"""
Métricas de negocio para modelos de riesgo de crédito
======================================================

Conjunto de métricas estándar usadas en la industria actuarial y bancaria
para evaluar scorecards y modelos de otorgamiento de crédito:

* KS (Kolmogorov-Smirnov)
* AUROC (área bajo la curva ROC)
* Gini (normalizado / coeficiente de concentración)
* CAP y Accuracy Ratio (AR)
* Ganancia / Lift acumulada
* LogLoss y Brier (calibración)
* Matriz de confusión y métricas de corte (precisión, sensibilidad)

Todas las funciones tienen la misma convención:

    y      -> vector de 0/1 (1 = malo / incumple)
    score  -> puntaje del modelo, a MAYOR score MENOR riesgo
            (es decir, score = probabilidad de ser "bueno")
    p      -> probabilidad de clase MALO (1) si se provee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, log_loss, confusion_matrix
)

from . import config


# ----------------------------------------------------------------------
# Métricas de discriminación
# ----------------------------------------------------------------------
def ks(y, score):
    """Distancia de Kolmogorov-Smirnov entre buenos y malos.

    Máxima separación acumulada entre la distribución de los buenos y la
    de los malos ordenados por score. Un KS >= 0.30 es razonable para
    scorecards de otorgamiento.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    df = pd.DataFrame({"score": score, "y": y}).sort_values("score", ascending=False)
    df["cum_bad"] = df["y"].cumsum()
    df["cum_good"] = (1 - df["y"]).cumsum()
    total_bad = df["y"].sum()
    total_good = (1 - df["y"]).sum()
    if total_bad == 0 or total_good == 0:
        return 0.0
    df["ecdf_bad"] = df["cum_bad"] / total_bad
    df["ecdf_good"] = df["cum_good"] / total_good
    diff = (df["ecdf_bad"] - df["ecdf_good"]).abs()
    ks_val = float(diff.max())
    best_scores = df.loc[diff.idxmax(), "score"]
    return {"ks": ks_val, "threshold": float(best_scores)}


def auroc(y, score):
    """Área bajo la curva ROC.

    Convención del proyecto: score ALTO = buen pagador (clase 0).
    `roc_auc_score` con los "buenos" como positivos devuelve
    P(score_bueno > score_malo), correcta y >= 0.5.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    good = (1 - y).astype(int)  # buenos como clase positiva
    return float(roc_auc_score(good, score))


def gini(y, score, method="normalized"):
    """Coeficiente de Gini del score.

    * method='normalized': Gini = 2*AUROC - 1  (independiente del tamaño)
    * method='ratio':      Gini = (AUROC - 0.5)/0.5 (equivale al normalizado)
    """
    auc = auroc(y, score)
    return round(2 * auc - 1.0, 6)


def roc_curve_data(y, score):
    """Devuelve FPR/TPR para graficar la curva ROC (buenos = positivos)."""
    y = np.asarray(y).astype(int)
    good = (1 - y).astype(int)
    fpr, tpr, _ = roc_curve(good, score)  # score alto -> bueno
    return fpr, tpr


def cap_lorenz(y, score, n_points=1000):
    """Coordenadas de la curva CAP (curva de concentración de malos).

    CAP: si ordenamos de mejor a peor score (mayor a menor riesgo), qué
    fracción de los MALOS se concentra en X% de la población.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)

    # orden ascendente de score -> primitive los de mayor riesgo
    df = pd.DataFrame({"score": score, "y": y}).sort_values("score")
    df["cum_y"] = df["y"].cumsum()
    total_bad = df["y"].sum() or 1
    population = np.linspace(0, 1, n_points)
    cum_pop = np.linspace(0, len(df), n_points + 1)[1:] / len(df)
    cum_bad_frac = np.interp(np.arange(1, len(df) + 1), np.arange(1, len(df) + 1),
                             df["cum_y"].values / total_bad)
    x = np.linspace(0, 1, len(df))
    return x, cum_bad_frac


def accuracy_ratio(y, score):
    """Razón de Precisión (AR) = área entre curva CAP y la diagonal.

    AR = (area_model - area_random) / (area_perfect - area_random)
    Es una medida de ganancia de concentración (también llamada
    'coeficiente de Gini' por algunas instituciones).
    """
    x, cap = cap_lorenz(y, score)
    area_model = float(np.trapz(cap, x))
    area_random = 0.5
    area_perfect = 1.0  # se aproxima: toda la masa de malos al inicio
    # área perfecta exacta si hay frac de malos r: 1 - r/2
    r = np.asarray(y).mean()
    area_perfect = 1.0 - r / 2.0
    ar = (area_model - area_random) / (area_perfect - area_random)
    return float(np.clip(ar, -5, 5))


def lift_df(y, score, n_deciles=10):
    """Tabla de deciles con concentración de malos y ganancia (lift)."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    df = pd.DataFrame({"score": score, "y": y}).sort_values("score", ascending=False)
    df["decile"] = pd.qcut(df["score"].rank(method="first"), n_deciles,
                           labels=False) + 1
    # decil 1 = menor riesgo (mejores)
    table = df.groupby("decile").agg(
        n=("y", "size"),
        n_bad=("y", "sum"),
    ).reset_index()
    table["bad_rate"] = table["n_bad"] / table["n"]
    table["cum_bad"] = table["n_bad"].cumsum()
    table["cum_bad_pct"] = table["cum_bad"] / table["n_bad"].sum()
    table["cum_pop"] = table["n"].cumsum() / table["n"].sum()
    table["lift"] = table["bad_rate"] / (table["n_bad"].sum() / table["n"].sum())
    return table


# ----------------------------------------------------------------------
# Calibración
# ----------------------------------------------------------------------
def brier_score(y, p_bad):
    """Brier: error cuadrático medio entre la probabilidad pronosticada
    y la clase observada. Más bajo es mejor (0 = perfecto)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p_bad, dtype=float), 0, 1)
    return float(np.mean((p - y) ** 2))


def log_loss_value(y, p_bad):
    """LogLoss (devianza binomial). Menor es mejor."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p_bad, dtype=float), 1e-9, 1 - 1e-9)
    return float(log_loss(y, p))


# ----------------------------------------------------------------------
# Métricas de corte / decisión
# ----------------------------------------------------------------------
def cutoff_metrics(y, score, max_bad_rate=0.50):
    """Elige el cut-off paramétrico que cumple una tasa de mora objetivo
    y regresa el resto de métricas de la matriz de confusión.

    Regresa un DataFrame con filas por cada candidato de score y columnas:
    aprobados, tasa de mora esperada, precisión, sensibilidad.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    df = pd.DataFrame({"score": score, "y": y})
    # score alto = bueno -> apruebo si score >= s
    candidates = np.quantile(score, np.linspace(0.01, 0.99, 100))
    rows = []
    for s in candidates:
        pred = (df["score"] >= s).astype(int)
        approved = pred.sum()
        if approved == 0:
            continue
        tp = ((pred == 1) & (df["y"] == 0)).sum()  # aprobado y bueno
        fp = ((pred == 1) & (df["y"] == 1)).sum()  # aprobado y malo
        fn = ((pred == 0) & (df["y"] == 1)).sum()  # rechazado y malo
        rows.append({
            "cutoff": float(s),
            "aproved": int(approved),
            "bad_rate_approved": float(fp / approved),
            "precision": float(tp / approved),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Utilidades de graficación
# ----------------------------------------------------------------------
def plot_roc(y, score, ax=None, label="Modelo", color="#1f77b4"):
    fpr, tpr = roc_curve_data(y, score)
    auc = auroc(y, score)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    return ax


def plot_cap(y, score, ax=None, label="Modelo", color="#d62728"):
    x, cap = cap_lorenz(y, score)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, cap, color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.plot([0, 0, 1], [0, 1, 1], "b--", lw=1, label="Perfecto")
    ax.set_xlabel("Fracción de la población")
    ax.set_ylabel("Fracción acumulada de malos")
    ax.set_title("Curva CAP")
    ax.legend(loc="upper left")
    return ax


def plot_ks(y, score, ax=None, label="Modelo"):
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    df = pd.DataFrame({"score": score, "y": y}).sort_values("score")
    df["cum_bad"] = (df["y"]).cumsum()
    df["cum_good"] = (1 - df["y"]).cumsum()
    df["ecdf_bad"] = df["cum_bad"] / df["y"].sum()
    df["ecdf_good"] = df["cum_good"] / (1 - df["y"]).sum()
    diff = np.abs(df["ecdf_bad"] - df["ecdf_good"])
    ks_val = diff.max()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df["ecdf_good"], color="#1f77b4", label="Buenos")
    ax.plot(df["ecdf_bad"], color="#d62728", label="Malos")
    idx = diff.idxmax()
    ax.vlines(idx, df.loc[idx, "ecdf_good"], df.loc[idx, "ecdf_bad"],
              color="k", ls="--", lw=1.5,
              label=f"KS = {ks_val:.3f}")
    ax.set_xlabel("Población ordenada por score")
    ax.set_ylabel("CDF acumulada")
    ax.set_title("Gráfica KS")
    ax.legend(loc="center right")
    return ax


def plot_lift(lift_tbl, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift_tbl["decile"], lift_tbl["bad_rate"],
           color="#2ca02c", edgecolor="k")
    ax.axhline(lift_tbl["bad_rate"].mean(), color="r", ls="--",
               label="Mora promedio")
    ax.set_xlabel("Decil (1 = menor riesgo)")
    ax.set_ylabel("Tasa de mora")
    ax.set_title("Tasa de mora por decil de score")
    ax.legend()
    return ax
