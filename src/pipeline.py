# -*- coding: utf-8 -*-
"""
Pipeline maestro
================

Orquesta todo el flujo de trabajo:

  1. Carga y preparación de datos
  2. Construcción de características y variable objetivo
  3. Split temporal (desarrollo / out-of-time)
  4. Comparación de modelos (Logit, RF, XGBoost) con validación cruzada
     y validación out-of-time, usando métricas de negocio
  5. Construcción del scorecard de puntos (WoE/IV/PDO) con variables
     seleccionadas por Information Value
  6. Simulación de la regla de aprobación del 25% y clasificación de
     aprobados en Buenos/Malos, comparada con políticas de referencia
  7. Exportación de tablas, figuras y métricas a `reports/`
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from . import (config, load_data, features, metrics, models,
               scorecard, approval)


def _save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _prepare_sets(raw):
    """Construye features, target y los subconjuntos dev/OOT."""
    feat = features.build_features(raw)
    tgt = features.build_target(raw)
    merged = feat.merge(tgt[tgt["resolved"]], on=["id", "issue_d"])

    dev = merged[merged["issue_d"] < pd.Timestamp(config.MID_CUTOFF)].copy()
    oot = merged[merged["issue_d"] >= pd.Timestamp(config.MID_CUTOFF)].copy()
    oot = oot[oot["issue_d"] < pd.Timestamp(config.OOT_CUTOFF)].copy()

    X_tr = dev[features.ORIGINATION_FEATURES].copy()
    X_oot = oot[features.ORIGINATION_FEATURES].copy()
    y_tr = dev["bad"].astype(int).values
    y_oot = oot["bad"].astype(int).values
    return feat, tgt, dev, oot, X_tr, X_oot, y_tr, y_oot


def _scorecard_features(X_tr, y_tr):
    """Selecciona variables del scorecard por Information Value y
    depura pares colineales (conserva la de mayor IV)."""
    ivs = {}
    for f in features.ORIGINATION_FEATURES:
        s = X_tr[f]
        if s.dtype.kind == "O" or s.nunique() <= 8:
            tbl = scorecard._woe_cat(y_tr, s, sorted(s.dropna().unique()))
        else:
            bins = scorecard.quantile_bins(s, n_bins=8)
            tbl = scorecard._woe_num(y_tr, s, bins)
        ivs[f] = float(tbl["iv"].sum())

    ranked = sorted(ivs.items(), key=lambda t: -t[1])
    sel = []
    # depura colinealidad: no aceptar una variable con |r|>0.9 con otra elegida
    for f, v in ranked:
        if v < config.SCORECARD_MIN_IV:
            continue
        if any(
            abs(X_tr[[f, g]].dropna().corr().iloc[0, 1]) > 0.90
            for g in sel if f in X_tr.columns and g in X_tr.columns
        ):
            continue
        sel.append(f)
    iv_df = pd.DataFrame([{"variable": k, "IV": v} for k, v in ivs.items()])
    iv_df["seleccionada"] = iv_df["variable"].isin(sel)
    return sel, iv_df.sort_values("IV", ascending=False).reset_index(drop=True)


def _build_woe_matrix(X, y, features_, n_bins=8):
    """Matriz de WoE (una columna por variable) con los tramos."""
    woe_map = {}
    for f in features_:
        if X[f].dtype.kind == "O" or X[f].nunique() <= 8:
            tbl = scorecard._woe_cat(y, X[f], sorted(X[f].dropna().unique()))
            mapa = dict(zip(tbl["tramo"].astype(str), tbl["woe"]))
            woe_map[f] = X[f].astype(str).map(mapa)
        else:
            bins = scorecard.quantile_bins(X[f], n_bins=n_bins)
            tbl = scorecard._woe_num(y, X[f], bins)
            edges = [-np.inf] + list(bins) + [np.inf]
            cats = pd.cut(X[f], bins=edges, right=True, include_lowest=True)
            cats_str = cats.cat.rename_categories(
                lambda c: f"[{c.left:.2g} , {c.right:.2g})")
            mapa = dict(zip(cats_str.astype(str), tbl["woe"]))
            woe_map[f] = cats_str.astype(str).map(mapa)
    return pd.DataFrame(woe_map, index=X.index).fillna(0.0)


def run(force_reload=False):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    config.TABLES.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) Datos ----------------
    raw = load_data.load_raw(force=force_reload)
    feat, tgt, dev, oot, X_tr, X_oot, y_tr, y_oot = _prepare_sets(raw)

    # ---------------- 2) Comparativa de modelos ----------------
    model_dict = models.build_models()

    # CV dentro de desarrollo
    cv_df = models.train_evaluate_cv(model_dict, X_tr, y_tr)
    cv_df.to_csv(config.TABLES / "comparativa_cv.csv", index=False)

    # OOT
    oot_df, bundles = models.train_evaluate_oot(
        model_dict, X_tr, y_tr, X_oot, y_oot)
    oot_df.to_csv(config.TABLES / "comparativa_oot.csv", index=False)

    # figura ROC comparativa OOT
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ["Logistica", "RandomForest", "XGBoost"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for (name, _p, _pr, score), c in zip(bundles, colors):
        ax = metrics.plot_roc(y_oot, score, ax=ax, label=name, color=c)
    _save_fig(fig, "roc_oot_comparativa.png")

    # ---------------- 3) Scorecard ----------------
    # 3a. selección por IV
    sc_feats, iv_df = _scorecard_features(X_tr, y_tr)

    # 3b. tabla WoE/IV y matriz de WoE
    sc_table = scorecard.scorecard_table(
        X_tr, y_tr, sc_feats, n_bins=8)
    W_tr = _build_woe_matrix(X_tr, y_tr, sc_feats)
    W_oot = _build_woe_matrix(X_oot, y_oot, sc_feats)

    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    W_tr_f = imp.fit_transform(W_tr)
    W_oot_f = imp.transform(W_oot)

    # 3c. regresión logística sobre WoE (1 beta por variable)
    beta_mod = LogisticRegression(max_iter=3000, C=1.0)
    beta_mod.fit(W_tr_f, y_tr)
    betas = dict(zip(sc_feats, beta_mod.coef_[0]))

    sc = scorecard.Scorecard(sc_table, betas, beta_mod.intercept_[0], sc_feats)

    # puntajes OOT
    score_oot = sc.score_from_woe(pd.DataFrame(W_oot_f, columns=sc_feats))
    linear_oot = (config.SCORE_OFFSET - score_oot) / sc.factor
    p_oot = 1.0 / (1.0 + np.exp(-linear_oot))  # P(malo) desde el score

    sc_metrics = {
        "KS": metrics.ks(y_oot, score_oot)["ks"],
        "AUROC": metrics.auroc(y_oot, score_oot),
        "Gini": metrics.gini(y_oot, score_oot),
        "AR": metrics.accuracy_ratio(y_oot, score_oot),
    }

    # tabla de puntos
    pts = sc.points_table()
    pts.to_csv(config.TABLES / "scorecard_puntos.csv", index=False)

    # tabla de IV
    iv_df.to_csv(config.TABLES / "iv_variables.csv", index=False)

    # figuras del scorecard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    metrics.plot_roc(y_oot, score_oot, ax=ax1, label="Scorecard (OOT)")
    metrics.plot_cap(y_oot, score_oot, ax=ax2, label="Scorecard (OOT)")
    _save_fig(fig, "scorecard_roc_cap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics.plot_ks(y_oot, score_oot, ax=ax, label="Scorecard")
    _save_fig(fig, "scorecard_ks.png")

    # ---------------- 4) Regla 25% ----------------
    # Simulación honesta: decidir sobre los préstamos resueltos del OOT
    # (desenlace conocido); la política aprueba el 25% del monto con mejor
    # score.
    decisions = approval.apply_rationing(
        ids=oot["id"].values,
        amounts=oot["loan_amnt"].values,
        score=score_oot,
        y=oot["bad"].values,
        by=config.RATIONING_BY,
    )
    rep_model = approval.approval_report(decisions, label="Scorecard")
    decisions.to_csv(config.TABLES / "decisiones_aprobacion.csv", index=False)

    # referencias: aleatorio y 'aprobar todos'
    np.random.seed(config.SEED)
    rnd = approval.apply_rationing(
        oot["id"].values, oot["loan_amnt"].values,
        np.random.random(len(oot)), oot["bad"].values, by=config.RATIONING_BY)
    rep_rand = approval.approval_report(rnd, label="Aleatorio")

    all_dec = approval.apply_rationing(
        oot["id"].values, oot["loan_amnt"].values,
        np.full(len(oot), 1e9), oot["bad"].values, by="amount", rate=1.0)
    rep_all = approval.approval_report(all_dec, label="Aprobar todos",
                                       rate=1.0)

    approval_tbl = pd.DataFrame([rep_model, rep_rand, rep_all]).set_index("politica")
    approval_tbl.to_csv(config.TABLES / "regla_25_aprobacion.csv")

    # deciles de mora del scorecard OOT
    lift = metrics.lift_df(y_oot, score_oot, n_deciles=10)
    lift.to_csv(config.TABLES / "deciles_scorecard_oot.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics.plot_lift(lift, ax=ax)
    _save_fig(fig, "deciles_mora.png")

    # ---------------- 5) Resumen JSON ----------------
    summary = {
        "n_total": int(len(raw)),
        "n_resueltos": int(tgt["resolved"].sum()),
        "n_desarrollo": int(len(dev)),
        "n_oot": int(len(oot)),
        "tasa_mora_desarrollo": float(y_tr.mean()),
        "tasa_mora_oot": float(y_oot.mean()),
        "comparativa_cv": cv_df.set_index("modelo").to_dict(orient="index"),
        "comparativa_oot": oot_df.set_index("modelo").to_dict(orient="index"),
        "scorecard": sc_metrics,
        "scorecard_variables": sc_feats,
        "aprobacion": dict(rep_model),
        "aprobacion_aleatorio": dict(rep_rand),
        "aprobacion_todos": dict(rep_all),
    }
    with open(config.TABLES / "resumen_metricas.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    return summary


def _logit_from_score(score, sc):
    """Recupera log-odds desde score: ln(odds) = (offset - score)/factor."""
    return (sc._offset - score) / sc.factor


if __name__ == "__main__":
    run()
