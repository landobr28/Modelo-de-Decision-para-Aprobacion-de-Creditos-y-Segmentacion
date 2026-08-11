"""03_export.py - Exportación de resultados finales.

Genera:
  - outputs/loan_data_clasificado_final.xlsx : base completa con score,
    decisión (Aprobado/Rechazado) y segmento (Bueno/Malo).
  - outputs/tables/tabla_comparativa_metodos.csv : resumen comparativo
    del modelo estadístico vs. el modelo de reglas del ejercicio original.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config


def build_comparison_table() -> pd.DataFrame:
    """Construye la tabla comparativa de métodos con métricas publicadas."""
    m = (
        pd.read_csv(config.TABLES / "metricas_modelo.csv")
        .set_index("metrica")["valor"]
        .to_dict()
    )
    rows = [
        ("Tasa de aprobación (objetivo)", "25.00%", "25.00%", "25.00%"),
        (
            "Préstamos malos en la cartera aprobada",
            f"{m['tasa_malos_aprobados_test']:.2%}",
            f"{m['malos_aprobados_reglas_originales']:.2%}",
            f"{m['malos_aprobados_reglas_25']:.2%}",
        ),
        (
            "Préstamos malos en solicitudes rechazadas",
            f"{m['tasa_malos_rechazados_test']:.2%}",
            "—",
            "—",
        ),
        ("AUROC", f"{m['auc_test']:.4f}", f"{m['auc_reglas_originales']:.4f}", f"{m['auc_reglas_sin_leakage']:.4f}"),
        ("GINI", f"{m['gini_test']:.4f}", "—", "—"),
        (
            "Captura de malos excluidos de la cartera",
            f"{m['captura_malos_test']:.2%}",
            "—",
            "—",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Métrica",
            "Regresión logística (test)",
            "Reglas del ejercicio (con recuperación)",
            "Reglas del ejercicio (sin recuperación)",
        ],
    )


def main() -> None:
    print("=" * 70)
    print("03_export | Exportación de resultados")
    print("=" * 70)

    df = pd.read_csv(config.PROCESSED_CSV_GZ, compression="gzip", low_memory=False)
    pred = pd.read_csv(config.OUTPUTS / "predicciones_completas.csv.gz", compression="gzip")

    df_out = df.copy()
    df_out["score"] = pred["score"]
    df_out["decision"] = pred["decision"]
    df_out["segmento"] = pred["segmento"]

    cols_deseadas = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
        "installment", "grade", "emp_length", "emp_length_years",
        "home_ownership", "annual_inc", "log_annual_inc", "verification_status",
        "purpose", "dti", "delinq_2yrs", "mths_since_last_delinq",
        "open_acc", "revol_bal", "revol_util", "total_acc",
        "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "acc_now_delinq", "application_type", "credit_history_months",
        "recovery_pct", "bad_loan", "score", "decision", "segmento",
    ]
    df_out = df_out[[c for c in cols_deseadas if c in df_out.columns]]

    with pd.ExcelWriter(config.OUTPUT_CLASSIFIED, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Clasificación", index=False)

    print(f"Clasificación final guardada: {config.OUTPUT_CLASSIFIED}")
    resumen = (
        df_out.groupby(["decision", "segmento"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    print(resumen.to_string(index=False))

    tabla = build_comparison_table()
    tabla.to_csv(config.TABLES / "tabla_comparativa_metodos.csv", index=False)
    print(f"\nTabla comparativa guardada: {config.TABLES / 'tabla_comparativa_metodos.csv'}")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()