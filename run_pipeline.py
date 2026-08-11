"""run_pipeline.py - Orquesta el pipeline completo del proyecto.

Uso:
    python run_pipeline.py            # ejecuta todas las etapas en orden
    python run_pipeline.py --etapa 2  # ejecuta solo una etapa (0-3)

Etapas:
    0. Preprocesamiento de datos crudos
    1. Análisis exploratorio (EDA)
    2. Entrenamiento y evaluación del modelo
    3. Exportación de resultados
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    0: "scripts/00_preprocess.py",
    1: "scripts/01_eda.py",
    2: "scripts/02_train.py",
    3: "scripts/03_export.py",
}


def run(script: str) -> None:
    print(f"\n>>> Ejecutando: {script}")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] La etapa {script} falló con código {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de modelado crediticio")
    parser.add_argument(
        "--etapa", type=int, default=None, help="Ejecutar únicamente una etapa (0-3)"
    )
    args = parser.parse_args()

    if args.etapa is not None:
        if args.etapa not in SCRIPTS:
            sys.exit(f"Etapa inválida. Opciones: {sorted(SCRIPTS)}")
        run(SCRIPTS[args.etapa])
        return

    for etapa in sorted(SCRIPTS):
        run(SCRIPTS[etapa])

    print("\n" + "=" * 70)
    print("Pipeline completado. Resultados en la carpeta outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()