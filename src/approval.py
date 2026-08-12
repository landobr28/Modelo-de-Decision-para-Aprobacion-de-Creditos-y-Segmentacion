# -*- coding: utf-8 -*-
"""
Regla de aprobación (25% del crédito solicitado)
================================================

Simula la política de originación del banco:

    1. Se ordenan TODAS las solicitudes por su score (mayor score =
       menor riesgo).
    2. Se aprueba la parte del portafolio que agota el presupuesto
       asignado:
       * RATIONING_BY = 'amount' -> presupuesto = 25% del MONTO TOTAL
         solicitado (se aceptan los mejores hasta llenar el monto).
       * RATIONING_BY = 'count'  -> se aprueba el 25% del NÚMERO de
         solicitudes con mejor score.
    3. Los aprobados se clasifican entre Buenos y Malos con el desenlace
       real (para préstamos resueltos) y se reportan métricas.
"""

import numpy as np
import pandas as pd

from . import config


def apply_rationing(ids, amounts, score, y, cutoff=None, rate=None,
                    by="amount"):
    """Asigna aprobación por racionamiento.

    Parámetros
    ----------
    ids, amounts, score, y : arrays alineados (y: 1=malo, NaN=censurado)
    rate : fracción a aprobar (por defecto config.APPROVAL_RATE)
    by   : 'amount' -> presupuesto = rate * total_monto_solicitado
           'count'  -> aprobar rate * n_solicitudes

    Regresa
    -------
    DataFrame con columnas id, amount, score, y, approved, y razones.
    """
    rate = rate or config.APPROVAL_RATE
    df = pd.DataFrame({
        "id": ids, "amount": amounts, "score": score, "y": y,
    }).sort_values("score", ascending=False).reset_index(drop=True)
    df["approved"] = False

    if by == "amount":
        budget = float(rate * df["amount"].sum())
        used = 0.0
        for i in range(len(df)):
            if used + df.loc[i, "amount"] <= budget:
                df.loc[i, "approved"] = True
                used += df.loc[i, "amount"]
            else:
                # racionamiento por tramo: se aprueba la última de forma parcial no
                # representable; la marcamos como rechazada para conservar el monto
                continue
        df["budget"] = budget
        df["used"] = used
    else:  # by == 'count'
        n_appr = int(round(rate * len(df)))
        df.loc[:n_appr - 1, "approved"] = True

    return df


def approval_report(decisions, label="", rate=None, by="amount"):
    """Resumen ejecutivo de la política aplicada."""
    rate = rate or config.APPROVAL_RATE
    appr = decisions[decisions["approved"]]
    known = appr[appr["y"].notna()]
    total_amt = decisions["amount"].sum()
    appr_amt = appr["amount"].sum()

    report = {
        "politica": label,
        "solicitudes_total": int(len(decisions)),
        "solicitudes_aprobadas": int(len(appr)),
        "monto_solicitado_total": float(total_amt),
        "monto_aprobado": float(appr_amt),
        "tasa_aprobacion_conteo": round(len(appr) / len(decisions), 4),
        "tasa_aprobacion_monto": round(appr_amt / total_amt, 4),
    }
    if len(known) > 0:
        report.update({
            "aprobados_resueltos": int(len(known)),
            "aprobados_buenos": int((known["y"] == 0).sum()),
            "aprobados_malos": int((known["y"] == 1).sum()),
            "bad_rate_aprobados": round(known["y"].mean(), 4),
        })
    return report
