"""00_preprocess.py - Preprocesamiento de datos crudos.

Lee data/raw/loan_ejercicio.xlsx (Lending Club) y genera el dataset de
trabajo con: variables de solicitud limpias, variables derivadas,
variable objetivo (bad_loan) y guarda los resultados en data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data_prep import preprocess_raw


def main() -> None:
    print("=" * 70)
    print("00_preprocess | Preparación de datos")
    print("=" * 70)

    df = preprocess_raw()

    print(f"\nFilas: {len(df):,}")
    print(f"Columnas: {df.shape[1]}")
    print(f"\nDistribución de la variable objetivo (bad_loan):")
    print(df["bad_loan"].value_counts().to_string())
    tasa_malos = df["bad_loan"].mean()
    print(f"Tasa global de mal préstamo: {tasa_malos:.2%}")

    print("\nValores nulos por columna (top 10):")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(10).to_string())

    df.to_excel(config.PROCESSED_DATA_FILE, index=False)
    # Versión comprimida de carga rápida para las etapas siguientes
    df.to_csv(config.PROCESSED_CSV_GZ, index=False, compression="gzip")
    print(f"\nGuardado: {config.PROCESSED_DATA_FILE}")
    print(f"Guardado: {config.PROCESSED_CSV_GZ}")


if __name__ == "__main__":
    main()