"""Preparación de datos: carga, limpieza, ingeniería de variables y target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Columnas del modelo: únicamente información disponible AL MOMENTO DE LA
# SOLICITUD (sin "leakage": se excluyen variables de resultado del préstamo
# como pagos totales, recuperaciones, principal pendiente, etc.)
# ---------------------------------------------------------------------------
APPLICATION_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "mths_since_last_delinq",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "acc_now_delinq",
    "application_type",
    "earliest_cr_line",
    "issue_d",
]

OUTCOME_COLUMNS = [
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "out_prncp",
]

_EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

GRADE_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def load_raw_data() -> pd.DataFrame:
    """Carga el archivo crudo de Lending Club."""
    return pd.read_excel(config.RAW_DATA_FILE)


_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


def _parse_lc_date(value: object) -> tuple[float, float]:
    """Parsea fechas 'Mmm-YYYY' o datetime de Excel, sin depender del locale."""
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (pd.Timestamp,)):
        return float(value.year), float(value.month)
    parts = str(value).strip().split("-")
    if len(parts) == 3:  # datetime serializado como string
        try:
            dt = pd.to_datetime(value)
            return float(dt.year), float(dt.month)
        except (ValueError, TypeError):
            return np.nan, np.nan
    if len(parts) != 2:
        return np.nan, np.nan
    month = _MONTHS.get(parts[0][:3].lower(), np.nan)
    year = pd.to_numeric(parts[1], errors="coerce")
    return year, month


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha 'mmm-yyyy' y deriva variables útiles."""
    for col in ("issue_d", "earliest_cr_line"):
        parsed = df[col].map(_parse_lc_date)
        df[f"{col}_year"] = parsed.map(lambda p: p[0])
        df[f"{col}_month"] = parsed.map(lambda p: p[1])
    # Antigüedad del historial crediticio al momento de la solicitud (meses)
    df["credit_history_months"] = (
        (df["issue_d_year"] - df["earliest_cr_line_year"]) * 12
        + (df["issue_d_month"] - df["earliest_cr_line_month"])
    )
    return df


def _clean_scalar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Valores fuera de rango y tipos numéricos."""

    def _clip(col: str, lo: float, hi: float) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    df["loan_amnt"] = _clip("loan_amnt", 0, None)
    df["funded_amnt"] = _clip("funded_amnt", 0, None)
    df["funded_amnt_inv"] = _clip("funded_amnt_inv", 0, None)
    df["installment"] = _clip("installment", 0, None)
    df["int_rate"] = _clip("int_rate", 0, None) / 100.0  # formato porcentual
    df["dti"] = _clip("dti", 0, 100)
    df["annual_inc"] = _clip("annual_inc", 0, None)
    df["revol_util"] = _clip("revol_util", 0, 100)
    df["revol_bal"] = _clip("revol_bal", 0, None)
    df["open_acc"] = _clip("open_acc", 0, None).astype(float)
    df["total_acc"] = _clip("total_acc", 0, None).astype(float)
    df["acc_now_delinq"] = _clip("acc_now_delinq", 0, None).astype(float)
    df["collections_12_mths_ex_med"] = _clip(
        "collections_12_mths_ex_med", 0, None
    ).astype(float)
    df["delinq_2yrs"] = _clip("delinq_2yrs", 0, None).astype(float)
    df["mths_since_last_delinq"] = pd.to_numeric(
        df["mths_since_last_delinq"], errors="coerce"
    ).clip(0, None)
    df["mths_since_last_major_derog"] = pd.to_numeric(
        df["mths_since_last_major_derog"], errors="coerce"
    ).clip(0, None)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Define la variable objetivo Bad_Loan.

    Un préstamo se considera *malo* (Bad_Loan = 1) si la recuperación de
    principal recibida es inferior al 70% del monto financiado; en caso
    contrario, se considera *bueno* (Bad_Loan = 0).
    """
    valid = df["funded_amnt"] > 0
    df["recovery_pct"] = np.where(
        valid, df["total_rec_prncp"] / df["funded_amnt"], np.nan
    )
    # Montos financiados nulos o cero implican recuperación nula -> malo
    df["recovery_pct"] = df["recovery_pct"].fillna(0.0)
    df["bad_loan"] = (df["recovery_pct"] < config.RECOVERY_THRESHOLD).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica ingeniería de variables y normaliza tipos para el modelado."""
    df = df.copy()
    df["emp_length_years"] = df["emp_length"].map(_EMP_LENGTH_MAP).astype(float)
    df["grade_num"] = df["grade"].map(GRADE_ORDER).astype(float)
    df["term_months"] = df["term"].str.replace(" months", "").astype(float)
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])
    df["log_revol_bal"] = np.log1p(df["revol_bal"])
    df["is_joint"] = (df["application_type"] == "JOINT").astype(float)

    # Indicadores de dato faltante (información que el originador no registró)
    df["mths_since_last_delinq_missing"] = df["mths_since_last_delinq"].isna().astype(float)
    df["mths_since_last_major_derog_missing"] = (
        df["mths_since_last_major_derog"].isna().astype(float)
    )
    df["revol_util_missing"] = df["revol_util"].isna().astype(float)
    return df


def get_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de variables del modelo (solo datos de solicitud)."""
    feats = [
        "grade_num",
        "term_months",
        "int_rate",
        "installment",
        "emp_length_years",
        "dti",
        "delinq_2yrs",
        "mths_since_last_delinq",
        "mths_since_last_delinq_missing",
        "mths_since_last_major_derog",
        "mths_since_last_major_derog_missing",
        "open_acc",
        "total_acc",
        "collections_12_mths_ex_med",
        "acc_now_delinq",
        "revol_util",
        "revol_util_missing",
        "credit_history_months",
        "log_annual_inc",
        "log_loan_amnt",
        "log_revol_bal",
        "is_joint",
    ]
    categoricals = ["home_ownership", "verification_status", "purpose"]
    cat = pd.get_dummies(
        df[categoricals].astype("category"), prefix_sep="_", dtype=float
    )
    numeric = df[feats]
    return pd.concat([numeric, cat], axis=1)


def preprocess_raw() -> pd.DataFrame:
    """Pipeline completo: raw -> dataset de trabajo con target y features."""
    df = load_raw_data()
    df = _parse_dates(df)
    df = _clean_scalar_features(df)
    df = build_target(df)
    df = engineer_features(df)
    return df