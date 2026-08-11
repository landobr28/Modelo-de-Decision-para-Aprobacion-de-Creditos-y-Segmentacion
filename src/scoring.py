"""Modelado: score de riesgo, scorecard crediticio y decisiones de aprobación.

El pipeline modela la probabilidad de *mal préstamo* mediante regresión
logística y la transforma a un scorecard en escala de puntos (tipo FICO),
sobre el cual se fijan los dos umbrales de decisión del negocio:

1. Corte de aprobación   : aprueba el 25% de las solicitudes con menor riesgo.
2. Sub-segmentación       : divide los aprobados en *Buenos* (premium) y
                            *Malos* (riesgo controlado) mediante la mediana
                            del score dentro del grupo aprobado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# División de la muestra
# ---------------------------------------------------------------------------
def train_test_split_stratified(n: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split estratificado (70/30) por la variable objetivo.

    Retorna los índices de entrenamiento y test sobre la población completa.
    """
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return idx_train, idx_test


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Pipeline de regresión logística: imputación por mediana, z-score y modelo.

    La imputación se ajusta únicamente con la muestra de entrenamiento para
    evitar fuga de información hacia la muestra de test.
    """
    from sklearn.impute import SimpleImputer

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=2_000, C=1.0, random_state=config.RANDOM_STATE
                ),
            ),
        ]
    )


def fit_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entrena el modelo de probabilidad de 'mal préstamo'."""
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Scorecard (escala de puntos)
# ---------------------------------------------------------------------------
def build_scorecard(p_cutoff: float) -> dict:
    """Convierte el modelo a scorecard: score = offset - factor * ln(odds).

    - factor  = 20 / ln(2)  (20 puntos por cada duplicación del odds).
    - offset  = SCORE_REFERENCE + factor * ln(odds(p_cutoff)), de modo que
      el punto de corte de aprobación coincida con SCORE_REFERENCE (600).
    - Mayor puntaje = menor riesgo.
    """
    factor = config.SCORE_FACTOR
    odds_cutoff = p_cutoff / (1.0 - p_cutoff)
    offset = config.SCORE_REFERENCE + factor * np.log(odds_cutoff)
    return {"factor": factor, "offset": offset, "p_cutoff": p_cutoff}


def probability_to_score(prob: np.ndarray | pd.Series, scorecard: dict) -> np.ndarray:
    """Convierte probabilidades a puntajes del scorecard."""
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    odds = prob / (1.0 - prob)
    return scorecard["offset"] - scorecard["factor"] * np.log(odds)


def fit_desicion_rules(prob_train: np.ndarray) -> dict:
    """Fija los umbrales operativos sobre la muestra de entrenamiento.

    Retorna el scorecard y los cortes (aprobación y sub-segmento) con los
    que se evaluará cualquier muestra nueva.
    """
    p_cutoff = np.quantile(prob_train, config.APPROVAL_RATE)
    scorecard = build_scorecard(p_cutoff)
    score_train = probability_to_score(prob_train, scorecard)
    score_cutoff = config.SCORE_REFERENCE  # : score >= 600 -> aprobado

    aprobados_mask = score_train >= score_cutoff
    score_interno = np.quantile(score_train[aprobados_mask], 0.50)

    return {
        "p_cutoff": p_cutoff,
        "score_cutoff": score_cutoff,
        "score_interno": score_interno,
        "scorecard": scorecard,
    }


def apply_decision_rules(prob: np.ndarray, rules: dict) -> pd.DataFrame:
    """Clasifica con el scorecard: Aprobado/Rechazado y segmento Bueno/Malo."""
    score = probability_to_score(prob, rules["scorecard"])
    aprobado = score >= rules["score_cutoff"]
    bueno = aprobado & (score >= rules["score_interno"])
    segmento = np.select([bueno, aprobado], ["Bueno", "Malo"], default="Rechazado")
    return pd.DataFrame(
        {"score": score, "decision": np.where(aprobado, "Aprobado", "Rechazado"), "segmento": segmento}
    )