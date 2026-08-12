# -*- coding: utf-8 -*-
"""
Modelos de comparación
======================

Entrena una triada de modelos sobre la muestra de desarrollo (préstamos
resueltos) y evalúa con:

* Validación cruzada k-fold (robustez dentro de la muestra)
* Validación *out-of-time* (vintages más recientes no usados en el ajuste)

Modelos:
  1. Regresión Logística (base, scorecard lineal)
  2. Random Forest
  3. XGBoost

Preprocesamiento: winsorización de colas, transformación logarítmica de
variables muy sesgadas, imputación y estandarización.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier

from . import config, metrics, features


LOG_FEATURES = {
    "annual_inc",
    "income_per_loan",
    "revol_bal",
    "loan_amnt",
    "installment",
}


def _winsorize(features, limits=(0.005, 0.995)):
    """Recorta colas extremas (robusto a outliers)."""
    def _clip(X):
        X = X.copy()
        for col in features:
            if col in X.columns:
                lo, hi = np.nanpercentile(X[col], [limits[0] * 100, limits[1] * 100])
                X[col] = np.clip(X[col], lo, hi)
        return X
    return FunctionTransformer(_clip, validate=False)


class _LogTransformer:
    """log1p sobre variables muy sesgadas (conserva columnas)."""
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = np.log1p(np.clip(X[c], 0, None))
        return X

    def fit(self, X, y=None):
        return self


def _log_one():
    return FunctionTransformer(
        lambda X: _apply_logs(X), validate=False)


def _apply_logs(X):
    X = X.copy()
    for c in LOG_FEATURES:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))
    return X


# ----------------------------------------------------------------------
# Preprocesamiento
# ----------------------------------------------------------------------
def _features_by_type():
    numeric_cols = [c for c in features.ORIGINATION_FEATURES
                    if c not in ("purpose", "home_ownership",
                                 "verification_status", "application_type")]
    cat_cols = ["purpose", "home_ownership",
                "verification_status", "application_type"]
    return numeric_cols, cat_cols


def _preprocessor(logs=True):
    """Pipeline: winsoriza -> log -> imputa -> estandariza (num) / onehot."""
    num_cols, cat_cols = _features_by_type()
    numeric_pipe = Pipeline([
        ("winsor", _winsorize(num_cols)),
        ("log", FunctionTransformer(_apply_logs, validate=False) if logs
               else "passthrough"),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _minimal_onehot()),
    ])
    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre


def _minimal_onehot():
    from sklearn.preprocessing import OneHotEncoder
    return OneHotEncoder(handle_unknown="ignore")


# ----------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------
def build_models():
    """Define la triada de modelos."""
    return {
        "Logistica": LogisticRegression(max_iter=3000, class_weight="balanced",
                                        C=0.7, solver="liblinear"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=10, n_jobs=-1,
            random_state=config.SEED, min_samples_leaf=40,
            class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.8,
            min_child_weight=30, eval_metric="logloss",
            random_state=config.SEED, verbosity=0),
    }


def _fit_and_predict(models, X_tr, y_tr, X_va, y_va):
    """Entrena cada modelo (con su propio preprocesador) y devuelve
    (name, pipe, p_bad_va, score_va)."""
    results = []
    for name, est in models.items():
        pre = _preprocessor(logs=(name != "RandomForest"))
        if name == "RandomForest":
            # RF se beneficia más de imputación que de escalado; aún así
            # se usa el mismo pipeline (winsor+imputa) para consistencia.
            pre = _preprocessor(logs=False)
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]  # P(malo)
        score_va = -p_va                      # score alto = menor riesgo
        results.append((name, pipe, p_va, score_va))
    return results


def train_evaluate_cv(models, X_tr, y_tr, folds=None):
    """Validación cruzada K-fold: métricas medias de KS/AUROC/Gini."""
    folds = folds or config.CV_FOLDS
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    rows = []
    for name, est in models.items():
        pre = _preprocessor(logs=(name != "RandomForest"))
        if name == "RandomForest":
            pre = _preprocessor(logs=False)
        pipe = Pipeline([("pre", pre), ("model", est)])
        p = cross_val_predict(pipe, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
        score = -p
        rows.append({
            "modelo": name,
            "KS_cv": metrics.ks(y_tr, score)["ks"],
            "AUROC_cv": metrics.auroc(y_tr, score),
            "Gini_cv": metrics.gini(y_tr, score),
            "AR_cv": metrics.accuracy_ratio(y_tr, score),
            "LogLoss_cv": metrics.log_loss_value(y_tr, p),
            "Brier_cv": metrics.brier_score(y_tr, p),
        })
    return pd.DataFrame(rows)


def train_evaluate_oot(models, X_tr, y_tr, X_oot, y_oot):
    """Entrena cada modelo y evalúa sobre el OOT (métricas de negocio)."""
    results = []
    bundles = []
    for name, _pipe, p_va, score_va in _fit_and_predict(
            models, X_tr, y_tr, X_oot, y_oot):
        results.append({
            "modelo": name,
            "KS": metrics.ks(y_oot, score_va)["ks"],
            "AUROC": metrics.auroc(y_oot, score_va),
            "Gini": metrics.gini(y_oot, score_va),
            "AccuracyRatio": metrics.accuracy_ratio(y_oot, score_va),
            "LogLoss": metrics.log_loss_value(y_oot, p_va),
            "Brier": metrics.brier_score(y_oot, p_va),
        })
        bundles.append((name, _pipe, p_va, score_va))
    return pd.DataFrame(results), bundles
