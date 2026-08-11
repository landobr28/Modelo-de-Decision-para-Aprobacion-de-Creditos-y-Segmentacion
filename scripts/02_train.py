"""02_train.py - Entrenamiento y evaluación del modelo de decisión.

1. Divide la muestra (70/30, estratificada).
2. Entrena regresión logística sobre variables de solicitud (sin leakage).
3. Construye el scorecard y fija los umbrales de aprobación (25%) y de
   sub-segmentación (Bueno/Malo) con la muestra de entrenamiento.
4. Evalúa en ambas muestras: AUROC, GINI, KS y métricas de negocio.
5. Compara contra el modelo de reglas del ejercicio original.
6. Guarda modelo, umbrales, métricas y figuras en outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import get_model_matrix
from src.metrics import approval_metrics, discrimination_metrics, roc_arrays
from src.plots import (
    plot_approval_rates_by_segment,
    plot_confusion_matrix,
    plot_lift_chart,
    plot_roc_curves,
    plot_score_distribution,
)
from src.scoring import (
    apply_decision_rules,
    fit_desicion_rules,
    fit_model,
    train_test_split_stratified,
)

MODEL_PATH = config.OUTPUTS / "modelo_logistico.joblib"
RULES_PATH = config.OUTPUTS / "reglas_decision.json"


# ---------------------------------------------------------------------------
# Baseline: modelo de reglas del ejercicio original
# ---------------------------------------------------------------------------
def rule_based_baselines(df: pd.DataFrame) -> dict:
    """Replica el modelo de reglas del ejercicio de clase.

    VP: mths_since_last_delinq >= 30 o sin registro.
    VS: (1) dti <= 22, (2) grade = 'A', (3) delinq_2yrs = 0,
        (4) recuperacion >= 70%  <- usa la variable objetivo (leakage).

    Se reporta también la variante *sin* la regla (4), que es la única
    implementable en la práctica con información de solicitud.
    """
    df = df.copy()
    df["vp_ok"] = (df["mths_since_last_delinq"].fillna(np.inf) >= 30).astype(int)
    df["vs_1"] = (df["dti"] <= 22).astype(int)
    df["vs_2"] = (df["grade"] == "A").astype(int)
    df["vs_3"] = (df["delinq_2yrs"].fillna(-1) == 0).astype(int)
    df["vs_4"] = (df["recovery_pct"] >= 0.70).astype(int)

    res = {}
    count_4 = df[["vs_1", "vs_2", "vs_3", "vs_4"]].sum(axis=1)
    res["reglas_originales"] = {
        "aprobado": (df["vp_ok"] == 1) & (count_4 >= 3),
        "score_riesgo": 4 - count_4,
    }
    count_3 = df[["vs_1", "vs_2", "vs_3"]].sum(axis=1)
    res["reglas_sin_leakage"] = {
        "aprobado": (df["vp_ok"] == 1) & (count_3 >= 3),
        "score_riesgo": 3 - count_3,
    }
    return res


def evaluate_baseline(aprobado: np.ndarray, score_riesgo: np.ndarray, y: np.ndarray) -> dict:
    """Métricas de la política de reglas a su umbral natural."""
    frame = pd.DataFrame(
        {"decision": np.where(aprobado, "Aprobado", "Rechazado"), "bad_loan": y}
    )
    frame["segmento"] = frame["decision"]
    m = approval_metrics(frame)
    disc = discrimination_metrics(y, score_riesgo)
    return {**disc, **m}


def evaluate_baseline_at_25(score_riesgo: np.ndarray, y: np.ndarray) -> dict:
    """Evalúa el baseline cuando aprueba exactamente el 25% con menor riesgo."""
    cut = np.quantile(score_riesgo, 0.25)
    frame = pd.DataFrame(
        {"decision": np.where(score_riesgo <= cut, "Aprobado", "Rechazado"), "bad_loan": y}
    )
    frame["segmento"] = frame["decision"]
    return approval_metrics(frame)


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("02_train | Entrenamiento y evaluación del modelo")
    print("=" * 70)

    df = pd.read_csv(config.PROCESSED_CSV_GZ, compression="gzip", low_memory=False)
    X = get_model_matrix(df)
    y = df["bad_loan"].astype(int).to_numpy()
    print(f"Matriz de modelado: {X.shape[0]:,} filas x {X.shape[1]} variables")

    # Split estratificado por índices (70/30)
    train_idx, test_idx = train_test_split_stratified(len(df), y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 1. Entrenamiento
    model = fit_model(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado: {MODEL_PATH}")

    # 2. Probabilidades y score
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    rules = fit_desicion_rules(prob_train)
    train_pred = apply_decision_rules(prob_train, rules)
    test_pred = apply_decision_rules(prob_test, rules)

    ev_train = pd.DataFrame({"bad_loan": y_train}).join(train_pred)
    ev_test = pd.DataFrame({"bad_loan": y_test}).join(test_pred)

    # 3. Métricas de discriminación (score de riesgo = -score de bondad)
    risk_train = -train_pred["score"].to_numpy()
    risk_test = -test_pred["score"].to_numpy()
    disc_train = discrimination_metrics(y_train, risk_train)
    disc_test = discrimination_metrics(y_test, risk_test)
    print(
        "\nDiscriminación (train) -> "
        f"AUC={disc_train['auc']:.4f} | GINI={disc_train['gini']:.4f} | KS={disc_train['ks']:.4f}"
    )
    print(
        f"Discriminación (test)  -> "
        f"AUC={disc_test['auc']:.4f} | GINI={disc_test['gini']:.4f} | KS={disc_test['ks']:.4f}"
    )

    # 4. Métricas de negocio
    biz_train = approval_metrics(ev_train)
    biz_test = approval_metrics(ev_test)
    print("\n--- Política de aprobación (test) ---")
    print(f"Tasa de aprobación: {biz_test['tasa_aprobacion']:.2%} (objetivo 25%)")
    print(
        f"Tasa malos aprobados: {biz_test['tasa_malos_aprobados']:.2%} | "
        f"tasa malos rechazados: {biz_test['tasa_malos_rechazados']:.2%} | "
        f"tasa global: {biz_test['tasa_malos_global']:.2%}"
    )
    print(
        f"Segmento Bueno: n={biz_test['n_segmento_bueno']:,} "
        f"(tasa malos {biz_test['tasa_malos_segmento_bueno']:.2%})"
    )
    print(
        f"Segmento Malo : n={biz_test['n_segmento_malo']:,} "
        f"(tasa malos {biz_test['tasa_malos_segmento_malo']:.2%})"
    )
    print(f"Captura de malos (excluidos): {biz_test['captura_malos']:.2%}")

    # 5. Baseline de reglas (muestra completa)
    baselines = rule_based_baselines(df)
    y_all = df["bad_loan"].astype(int).to_numpy()
    ev_original = evaluate_baseline(
        baselines["reglas_originales"]["aprobado"].to_numpy(),
        baselines["reglas_originales"]["score_riesgo"].to_numpy(),
        y_all,
    )
    ev_sin_leak = evaluate_baseline(
        baselines["reglas_sin_leakage"]["aprobado"].to_numpy(),
        baselines["reglas_sin_leakage"]["score_riesgo"].to_numpy(),
        y_all,
    )
    ev_reglas_25 = evaluate_baseline_at_25(
        baselines["reglas_sin_leakage"]["score_riesgo"].to_numpy(), y_all
    )
    print("\n--- Baseline de reglas (muestra completa) ---")
    print(
        f"Reglas originales  : aprobación {ev_original['tasa_aprobacion']:.2%} | "
        f"malos en aprobados {ev_original['tasa_malos_aprobados']:.2%} | "
        f"AUC {ev_original['auc']:.4f}"
    )
    print(
        f"Reglas sin leakage : aprobación {ev_sin_leak['tasa_aprobacion']:.2%} | "
        f"malos en aprobados {ev_sin_leak['tasa_malos_aprobados']:.2%} | "
        f"AUC {ev_sin_leak['auc']:.4f}"
    )
    print(f"Reglas al 25% exacto: malos en aprobados {ev_reglas_25['tasa_malos_aprobados']:.2%}")

    # 6. Figuras
    fpr_tr, tpr_tr, auc_tr = roc_arrays(y_train, risk_train)
    fpr_te, tpr_te, auc_te = roc_arrays(y_test, risk_test)
    fpr_ru, tpr_ru, auc_ru = roc_arrays(
        y_all, baselines["reglas_originales"]["score_riesgo"].to_numpy()
    )
    fpr_r3, tpr_r3, auc_r3 = roc_arrays(
        y_all, baselines["reglas_sin_leakage"]["score_riesgo"].to_numpy()
    )
    fig1 = plot_roc_curves(
        {
            "Regresión logística (train)": (fpr_tr, tpr_tr, auc_tr),
            "Regresión logística (test)": (fpr_te, tpr_te, auc_te),
            "Reglas originales": (fpr_ru, tpr_ru, auc_ru),
            "Reglas sin leakage": (fpr_r3, tpr_r3, auc_r3),
        }
    )
    print(f"\nFigura generada: {fig1}")

    buenos = ev_train.loc[ev_train["bad_loan"] == 0, "score"].to_numpy()
    malos = ev_train.loc[ev_train["bad_loan"] == 1, "score"].to_numpy()
    fig2 = plot_score_distribution(
        buenos, malos, rules["score_cutoff"], rules["score_interno"]
    )
    print(f"Figura generada: {fig2}")

    fig3 = plot_confusion_matrix(
        y_test, (test_pred["decision"] == "Aprobado").to_numpy().astype(int)
    )
    print(f"Figura generada: {fig3}")

    fig4 = plot_lift_chart(ev_test)
    print(f"Figura generada: {fig4}")

    fig5 = plot_approval_rates_by_segment(ev_test)
    print(f"Figura generada: {fig5}")

    # 7. Guardar métricas y reglas
    tabla = pd.DataFrame(
        {
            "metrica": [
                "auc_train", "gini_train", "ks_train",
                "auc_test", "gini_test", "ks_test",
                "tasa_aprobacion_test", "tasa_malos_global",
                "tasa_malos_aprobados_test", "tasa_malos_rechazados_test",
                "tasa_malos_bueno_test", "tasa_malos_malo_test",
                "captura_malos_test", "calidad_cartera_test",
                "auc_reglas_originales", "auc_reglas_sin_leakage",
                "malos_aprobados_reglas_originales",
                "malos_aprobados_reglas_sin_leakage",
                "malos_aprobados_reglas_25",
            ],
            "valor": [
                disc_train["auc"], disc_train["gini"], disc_train["ks"],
                disc_test["auc"], disc_test["gini"], disc_test["ks"],
                biz_test["tasa_aprobacion"], biz_test["tasa_malos_global"],
                biz_test["tasa_malos_aprobados"], biz_test["tasa_malos_rechazados"],
                biz_test["tasa_malos_segmento_bueno"], biz_test["tasa_malos_segmento_malo"],
                biz_test["captura_malos"], biz_test["calidad_cartera"],
                ev_original["auc"], ev_sin_leak["auc"],
                ev_original["tasa_malos_aprobados"],
                ev_sin_leak["tasa_malos_aprobados"],
                ev_reglas_25["tasa_malos_aprobados"],
            ],
        }
    )
    tabla.to_csv(config.TABLES / "metricas_modelo.csv", index=False)

    reglas_exportable = {
        "nombre": "Modelo de decisión crediticia v1.0",
        "objetivo_aprobacion": config.APPROVAL_RATE,
        "definicion_malo": f"recuperación < {config.RECOVERY_THRESHOLD:.0%}",
        "modelo": "Regresión logística (estandarizada) sobre variables de solicitud",
        "p_cutoff": float(rules["p_cutoff"]),
        "score_cutoff": float(rules["score_cutoff"]),
        "score_interno": float(rules["score_interno"]),
        "factor_scorecard": float(rules["scorecard"]["factor"]),
        "offset_scorecard": float(rules["scorecard"]["offset"]),
    }
    with open(RULES_PATH, "w", encoding="utf-8") as fh:
        json.dump(reglas_exportable, fh, indent=2, ensure_ascii=False)
    print(f"\nReglas de decisión guardadas: {RULES_PATH}")

    # 8. Predicciones de la muestra completa (para exportación)
    prob_all = model.predict_proba(X)[:, 1]
    pred_all = apply_decision_rules(prob_all, rules)
    mask_train = np.zeros(len(df), dtype=bool)
    mask_train[train_idx] = True
    ev_full = pd.DataFrame(
        {"bad_loan": y_all, "split": np.where(mask_train, "train", "test")}
    ).join(pred_all)
    ev_full.to_csv(
        config.OUTPUTS / "predicciones_completas.csv.gz", index=False, compression="gzip"
    )
    print("Predicciones completas guardadas: outputs/predicciones_completas.csv.gz")

    with open(config.TABLES / "split_info.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "tasa_malos_train": float(y_train.mean()),
                "tasa_malos_test": float(y_test.mean()),
            },
            fh,
            indent=2,
        )
    print("\nPipeline 02_train finalizado.")


if __name__ == "__main__":
    main()