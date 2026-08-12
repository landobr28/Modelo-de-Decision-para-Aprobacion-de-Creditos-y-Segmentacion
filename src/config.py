# -*- coding: utf-8 -*-
"""
Configuración central del proyecto
==================================

Parámetros globales: rutas, umbrales de negocio, semilla de aleatoriedad
y definición del fenómeno de riesgo (proxies de "malo").

Proyecto de Administración Actuarial - FES Acatlán, UNAM
Semestre 2026-1 · Equipo: Bárcena, López, García, Carvajal, Flores
"""

from pathlib import Path

# ------------------------------------------------------------------
# Rutas del repositorio
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

RAW_LOANS = DATA_RAW / "loan_ejercicio.xlsx"
DICT_XLSX = DATA_RAW / "LCDataDictionary.xlsx"
PROCESSED_LOANS = DATA_PROCESSED / "loans_clean.pkl"

# ------------------------------------------------------------------
# Algoritmo / entorno
# ------------------------------------------------------------------
SEED = 2026
TEST_SIZE = 0.30          # fracción out-of-time (vintages más recientes)
MAX_ROWS_FOR_GRID = 60_000  # tope para optimización de hiperparámetros
CV_FOLDS = 5

# ------------------------------------------------------------------
# Regla de negocio de originación
# ------------------------------------------------------------------
APPROVAL_RATE = 0.25            # 25% del crédito total solicitado se aprueba
RATIONING_BY = "amount"         # "amount" -> presupuesto = 25% del monto total
                                # "count"  -> se aprueba el 25% del número de solicitudes

# ------------------------------------------------------------------
# Definición del evento de incumplimiento (proxy)
# ------------------------------------------------------------------
# Lending Club registra el default cuando el crédito se carga a pérdida
# (charge-off). En esa situación el saldo principal se escribe a cero
# (out_prncp == 0) y `recoveries` / `collection_recovery_fee` capturan
# posteriormente lo recuperado. Por tanto:
#   * Préstamos RESUELTOS  : out_prncp == 0  (pagado o cargo a pérdida)
#   * Préstamos ACTIVOS    : out_prncp > 0   (censurados -> se excluyen
#     de la muestra de desarrollo para no sesgar las tasas de mora)
#   * MALO (incumplido)    : recuperación > 0 ó cobranza > 0
#   * BUENO (cumplido)     : out_prncp == 0 y sin señales de cobranza
GOOD_BAD_METHOD = "charge_off"  # documenta el proxy aplicado

# Vintages: corte para garantizar ventana de observación suficiente.
# Los plazos son de 36/60 meses y el corte de datos es 2016-01; los
# créditos emitidos antes de MID_CUTOFF con plazo de 36 meses ya
# maduraron o se castigaron.
MID_CUTOFF = "2013-06-01"   # tren: issue_d < cutoff | OOT: >= cutoff
OOT_CUTOFF = "2015-01-01"   # el OOT considera vintages con >= 12m de ventana

# Scorecard: selección de variables por Information Value
SCORECARD_MIN_IV = 0.05     # conserva variables con IV total >= este umbral

# ------------------------------------------------------------------
# Scorecard (puntos)
# ------------------------------------------------------------------
# PDO (<-> "points to double the odds"): estándar de la industria.
# Cada +20 puntos duplica la probabilidad de ser BUENO.
SCORE_OFFSET = 600          # puntaje base de referencia
PDO = 20                    # duplicación de odds cada 20 puntos
ODDS_AT_OFFSET = 50         # odds (bueno/malo) en el puntaje base
MIN_BOOST_BINS = 5          # bins mínimos por variable discreta
