"""01_eda.py - Análisis exploratorio de datos (EDA).

Genera las figuras descriptivas de la cartera y una tabla resumen
con perfiles de buenos/malos préstamos en outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.plots import (
    eda_bad_rate_by_grade,
    eda_correlation,
    eda_delinq_profile,
    eda_income_dti,
)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil comparativo buenos vs malos préstamos."""
    grupo = df.groupby("bad_loan")
    def media(col):
        return grupo[col].mean()

    rows = {
        "Observaciones": grupo.size(),
        "Monto solicitado (USD)": media("loan_amnt"),
        "Tasa de interés (%)": media("int_rate") * 100,
        "Ingreso anual (USD)": media("annual_inc"),
        "DTI (%)": media("dti"),
        "Morosidades en 2 años": media("delinq_2yrs"),
        "Meses desde última morosidad": media("mths_since_last_delinq"),
        "Antigüedad historial (meses)": media("credit_history_months"),
        "Utilización de líneas (%)": media("revol_util"),
        "Líneas abiertas": media("open_acc"),
        "Recuperación de principal (%)": media("recovery_pct") * 100,
    }
    tabla = pd.DataFrame(rows).T
    tabla.columns = ["Buen préstamo (0)", "Mal préstamo (1)"]
    tabla.index.name = "Variable"
    return tabla.reset_index()


def main() -> None:
    print("=" * 70)
    print("01_eda | Análisis exploratorio de datos")
    print("=" * 70)

    df = pd.read_csv(config.PROCESSED_CSV_GZ, compression="gzip", low_memory=False)

    paths = [
        eda_bad_rate_by_grade(df),
        eda_delinq_profile(df),
        eda_income_dti(df),
        eda_correlation(df),
    ]
    for p in paths:
        print(f"Figura generada: {p}")

    tabla = summary_table(df)
    out = config.TABLES / "perfil_segmentos.csv"
    tabla.to_csv(out, index=False)
    print(f"\nTabla de perfil guardada: {out}")
    print(tabla.head(12).to_string(index=False))


if __name__ == "__main__":
    main()