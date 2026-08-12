# -*- coding: utf-8 -*-
"""Punto de entrada: ejecuta el pipeline completo y exporta resultados."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Pipeline de otorgamiento de crédito")
    ap.add_argument("--reload", action="store_true",
                    help="Recarga los datos desde el xlsx (ignora la caché)")
    args = ap.parse_args()
    run(force_reload=args.reload)
