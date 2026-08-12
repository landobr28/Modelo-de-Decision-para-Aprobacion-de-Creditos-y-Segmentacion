# -*- coding: utf-8 -*-
"""
Carga de datos
==============

Lee el archivo `loan_ejercicio.xlsx` (datos de originación y desempeño de
préstamos personales Lending Club, 2007-2015) y lo deja listo para el
proceso de modelado.

- Normaliza tipos de fecha.
- Guarda copia procesada en parquet para iterar rápido.
"""

import datetime

import numpy as np
import pandas as pd

from . import config


def _parse_month_year(value):
    """Convierte fechas tipo 'Dec-2011' / datetime / NaT a Timestamp."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    if isinstance(value, (datetime.datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    s = str(value).strip()
    for fmt in ("%b-%Y", "%Y-%m-%d", "%B-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s))
    except (ValueError, TypeError):
        return pd.NaT


def _to_years(emp_length):
    """Convierte '10+ years', '< 1 year', '5 years', NaN a años numéricos."""
    if emp_length is None:
        return np.nan
    s = str(emp_length).strip().lower()
    if s.startswith("<"):
        return 0.0
    if s.startswith("10+"):
        return 10.0
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return np.nan


def _term_months(term):
    m = str(term).strip().lower().replace("months", "").strip()
    try:
        return float(m)
    except ValueError:
        return np.nan


def load_raw(force=False):
    """Carga los datos originales desde el xlsx (o la caché parquet)."""
    if config.PROCESSED_LOANS.exists() and not force:
        df = pd.read_pickle(config.PROCESSED_LOANS)
        return df

    df = pd.read_excel(config.RAW_LOANS, sheet_name="loan_ejercicio")

    # --- fechas ----------------------------------------------------
    for col in ("issue_d", "last_pymnt_d", "next_pymnt_d",
                "last_credit_pull_d", "earliest_cr_line"):
        df[col] = df[col].apply(_parse_month_year)
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])

    # --- numéricos derivados de texto -----------------------------
    df["term_months"] = df["term"].apply(_term_months)
    df["emp_length_yrs"] = df["emp_length"].apply(_to_years)

    # --- orden y persistencia -------------------------------------
    df = df.sort_values("issue_d").reset_index(drop=True)
    config.PROCESSED_LOANS.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.PROCESSED_LOANS)
    return df
