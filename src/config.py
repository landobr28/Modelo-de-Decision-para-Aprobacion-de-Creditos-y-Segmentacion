"""Configuración central del proyecto.

Define rutas, parámetros del modelo y umbrales operativos en un único
lugar para garantizar reproducibilidad del pipeline completo.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES_EDA = OUTPUTS / "figures" / "eda"
FIGURES_MODEL = OUTPUTS / "figures" / "model"
TABLES = OUTPUTS / "tables"
REPORTS = ROOT / "reports"

RAW_DATA_FILE = DATA_RAW / "loan_ejercicio.xlsx"
RAW_DICTIONARY_FILE = DATA_RAW / "LCDataDictionary.xlsx"
PROCESSED_DATA_FILE = DATA_PROCESSED / "loan_data_features.xlsx"
PROCESSED_CSV_GZ = DATA_PROCESSED / "loan_data_features.csv.gz"
OUTPUT_CLASSIFIED = OUTPUTS / "loan_data_clasificado_final.xlsx"

for _dir in (DATA_PROCESSED, FIGURES_EDA, FIGURES_MODEL, TABLES, OUTPUTS):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parámetros del modelo y del negocio
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Objetivo operativo: aprobar únicamente el 25% de las solicitudes recibidas
APPROVAL_RATE = 0.25

# Definición de "mal préstamo" (proxy de incumplimiento):
# préstamo cuya recuperación de principal es inferior al 70% del monto financiado
RECOVERY_THRESHOLD = 0.70

# Parámetros del scorecard (escala crediticia tipo FICO).
# Factor: 20 puntos por cada duplicación del odds (estándar de la industria).
SCORE_FACTOR = 20.0 / 0.6931471805599453  # 20 / ln(2) ~= 28.85
SCORE_REFERENCE = 600  # puntaje asignado al punto de corte de aprobación (percentil 25 de riesgo)