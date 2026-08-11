"""Métricas de desempeño crediticio: AUROC, GINI, KS y métricas de negocio.

Se utilizan las métricas estándar de la industria de riesgo de crédito
(Basilea II, práctica de banca/seguros) para evaluar la capacidad
discriminatoria del score y el valor económico de la política de aprobación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def gini_from_auc(auc_value: float) -> float:
    """Coeficiente de GINI derivado del AUROC: GINI = 2*AUC - 1."""
    return 2.0 * auc_value - 1.0


def ks_statistic(y_true: np.ndarray, score_risk: np.ndarray) -> float:
    """Estadístico de Kolmogorov-Smirnov: máxima separación acumulada
    entre buenos y malos ordenando por score de riesgo (mayor = peor)."""
    frame = pd.DataFrame({"y": y_true, "s": score_risk})
    ordered = frame.sort_values("s")
    n_good = (ordered["y"] == 0).sum()
    n_bad = (ordered["y"] == 1).sum()
    if n_good == 0 or n_bad == 0:
        return 0.0
    cum_good = (ordered["y"] == 0).cumsum() / n_good
    cum_bad = (ordered["y"] == 1).cumsum() / n_bad
    return float((cum_good - cum_bad).abs().max())


def discrimination_metrics(y_true: np.ndarray, score_risk: np.ndarray) -> dict:
    """AUROC, GINI y KS a partir de un score de *riesgo* (mayor = peor)."""
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "gini": np.nan, "ks": np.nan}
    auc_value = roc_auc_score(y_true, score_risk)
    return {
        "auc": float(auc_value),
        "gini": float(gini_from_auc(auc_value)),
        "ks": float(ks_statistic(y_true, score_risk)),
    }


def roc_arrays(y_true: np.ndarray, score_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, score_risk)
    return fpr, tpr, auc(fpr, tpr)


def approval_metrics(df: pd.DataFrame, target_col: str = "bad_loan") -> dict:
    """Métricas de negocio de la política de aprobación aplicada a `df`.

    Requiere que `df` contenga la columna `decision` (Aprobado/Rechazado) y
    el segmento ajustado (Bueno/Malo/Rechazado) en la columna `segmento`.
    """
    total = len(df)
    y = df[target_col].astype(int)

    aprobado = df["decision"] == "Aprobado"
    n_aprobado = int(aprobado.sum())
    tasa_aprobacion = n_aprobado / total

    malos_totales = int(y.sum())
    malos_aprobados = int(y[aprobado].sum())
    malos_rechazados = malos_totales - malos_aprobados

    # Tasa de malos (incumplimiento observado) por grupo
    rate = lambda mask: float(y[mask].mean()) if mask.sum() > 0 else np.nan  # noqa: E731

    # Captura de malos: % de los malos totales que quedan fuera de la cartera
    captura_malos = malos_rechazados / malos_totales if malos_totales > 0 else np.nan

    # Distribución de segmentos dentro de aprobados
    seg = df.loc[aprobado, "segmento"]
    n_bueno = int((seg == "Bueno").sum())
    n_malo = int((seg == "Malo").sum())

    # Capacidad de pago esperada en la cartera aprobada: 1 - tasa_malos(aprobados)
    calidad_cartera = 1.0 - rate(aprobado) if n_aprobado > 0 else np.nan

    return {
        "total": total,
        "aprobados": n_aprobado,
        "rechazados": total - n_aprobado,
        "tasa_aprobacion": tasa_aprobacion,
        "malos_totales": malos_totales,
        "tasa_malos_global": malos_totales / total,
        "malos_aprobados": malos_aprobados,
        "malos_rechazados": malos_rechazados,
        "tasa_malos_aprobados": rate(aprobado),
        "tasa_malos_rechazados": rate(~aprobado),
        "tasa_malos_segmento_bueno": rate(aprobado & (seg == "Bueno")),
        "tasa_malos_segmento_malo": rate(aprobado & (seg == "Malo")),
        "n_segmento_bueno": n_bueno,
        "n_segmento_malo": n_malo,
        "captura_malos": captura_malos,
        "calidad_cartera": calidad_cartera,
        "confusion_matrix": confusion_matrix(y, aprobado.astype(int)).tolist(),
    }


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Convierte un dict de métricas en una tabla tipo informe."""
    rows = [
        ("Solicitudes evaluadas", metrics["total"]),
        ("Solicitudes aprobadas", metrics["aprobados"]),
        ("Solicitudes rechazadas", metrics["rechazados"]),
        ("Tasa de aprobación", f"{metrics['tasa_aprobacion']:.2%}"),
        ("Préstamos malos (observados)", metrics["malos_totales"]),
        ("Tasa de malos global", f"{metrics['tasa_malos_global']:.2%}"),
        ("Tasa de malos en aprobados", f"{metrics['tasa_malos_aprobados']:.2%}"),
        ("Tasa de malos en rechazados", f"{metrics['tasa_malos_rechazados']:.2%}"),
        ("Tasa de malos - segmento 'Bueno'", f"{metrics['tasa_malos_segmento_bueno']:.2%}"),
        ("Tasa de malos - segmento 'Malo'", f"{metrics['tasa_malos_segmento_malo']:.2%}"),
        ("Captura de malos (excluidos)", f"{metrics['captura_malos']:.2%}"),
        ("Calidad de la cartera aprobada", f"{metrics['calidad_cartera']:.2%}"),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Valor"])