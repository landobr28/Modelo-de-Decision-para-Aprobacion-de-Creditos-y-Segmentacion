# -*- coding: utf-8 -*-
"""
Ingeniería de características
=============================

Construye la matriz de modelado usando únicamente información DISPONIBLE
EN EL MOMENTO DE LA SOLICITUD (originación). Las variables de desempeño
(out_prncp, total_pymnt, recoveries, etc.) sirven sólo para definir la
variable objetivo y se excluyen de las características para no introducir
*look-ahead bias*.

También deriva la variable objetivo (Bueno/Malo) y el estatus de censura.
"""

import numpy as np
import pandas as pd

from . import config


# Características disponibles al originar (aplicación + buró)
ORIGINATION_FEATURES = [
    "loan_amnt", "term_months", "int_rate", "installment",
    "grade", "emp_length_yrs", "home_ownership", "annual_inc",
    "verification_status", "purpose", "dti",
    "delinq_2yrs", "open_acc", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "mths_since_last_delinq",
    "mths_since_last_major_derog", "application_type", "acc_now_delinq",
    "credit_history_yrs", "income_per_loan",
]

CENSOR_FEATURES = [
    "out_prncp", "recoveries", "collection_recovery_fee", "total_pymnt",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "funded_amnt_inv",
]

GRADE_ORDER = {g: i for i, g in enumerate(["A", "B", "C", "D", "E", "F", "G"])}


def build_features(df):
    """Devuelve un DataFrame con las características de originación."""
    out = pd.DataFrame(index=df.index)
    out["id"] = df["id"]
    out["issue_d"] = df["issue_d"]

    # antigüedad del historial crediticio (años) en el momento de originar
    credit_age = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    out["credit_history_yrs"] = credit_age.clip(lower=0)

    out["loan_amnt"] = df["loan_amnt"]
    out["term_months"] = df["term_months"]
    out["int_rate"] = df["int_rate"]
    out["installment"] = df["installment"]
    out["grade"] = df["grade"].map(GRADE_ORDER)
    out["emp_length_yrs"] = df["emp_length_yrs"]
    out["home_ownership"] = df["home_ownership"]
    out["annual_inc"] = df["annual_inc"]
    out["verification_status"] = df["verification_status"]
    out["purpose"] = df["purpose"]
    out["dti"] = df["dti"]
    out["delinq_2yrs"] = df["delinq_2yrs"]
    out["open_acc"] = df["open_acc"]
    out["revol_bal"] = df["revol_bal"]
    out["revol_util"] = df["revol_util"]
    out["total_acc"] = df["total_acc"]
    out["collections_12_mths_ex_med"] = df["collections_12_mths_ex_med"]
    out["mths_since_last_delinq"] = df["mths_since_last_delinq"]
    out["mths_since_last_major_derog"] = df["mths_since_last_major_derog"]
    out["application_type"] = df["application_type"]
    out["acc_now_delinq"] = df["acc_now_delinq"]

    # Cociente ingreso/importe (normalización de capacidad de pago)
    out["income_per_loan"] = df["annual_inc"] / df["loan_amnt"].replace(0, np.nan)

    # Flujo de cancelación de censura (por si se necesita)
    for c in CENSOR_FEATURES:
        out[c] = df[c] if c in df.columns else np.nan
    return out


def build_target(df):
    """Construye la variable objetivo Bueno/Malo y estatus de censura.

    Reglas:
      * resolved : out_prncp == 0  (crédito liquidado o castigado)
      * bad      : resolved & (recoveries > 0 | collection_recovery_fee > 0)
      * good     : resolved & ~bad
      * censored : ~resolved  (aún activo -> se excluye de desarrollo)
    """
    resolved = df["out_prncp"].astype(float).fillna(0) == 0
    bad = resolved & (
        df["recoveries"].fillna(0) > 0
    ) | (
        df["collection_recovery_fee"].fillna(0) > 0
    )
    good = resolved & ~bad
    censored = ~resolved

    return pd.DataFrame({
        "id": df["id"],
        "issue_d": df["issue_d"],
        "resolved": resolved,
        "bad": bad.astype(int),
        "good": good.astype(int),
        "censored": censored.astype(int),
    })
