# Modelo de Otorgamiento de Crédito con Política de Aprobación al 25%

**Proyecto de Administración Actuarial — FES Acatlán, UNAM · Semestre 2026-1**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/Licencia-MIT-green.svg)](LICENSE)
[![sklearn](https://img.shields.io/badge/scikit--learn-1.7-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-darkred)](https://xgboost.ai/)

Scorecard de riesgo de crédito construido para decidir **a qué solicitudes
otorgar crédito cuando sólo puede aprobarse el 25% del monto total
solicitado**, y clasificar los créditos aprobados en **Buenos** y **Malos**.

---

## Contexto

Una institución de crédito personal recibe **292,849 solicitudes** y tiene un
**presupuesto fijo de originación: 25% del monto total solicitado**.
El reto:

1. **¿A qué solicitudes otorgar el crédito?** (selección bajo presupuesto).
2. De los aprobados, **¿cuáles serán Buenos (pagaron) y cuáles Malos
   (incumplieron)?**

La base (Lending Club, 2007–2015) no reporta el estatus final del préstamo,
por lo que el fenómeno se construyó con un **proxy de cargo a pérdida
(charge-off)**: un crédito es *Malo* si `out_prncp = 0` (saldo liquidado o
castigado) y `recoveries > 0` o `collection_recovery_fee > 0`. Los créditos
activos (censurados) se excluyen de la muestra de desarrollo.

## Resultados principales

| Modelo (validación *out-of-time*) | KS | AUROC | Gini | LogLoss | Brier |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0.294 | **0.698** | **0.396** | 0.703 | 0.251 |
| Random Forest | 0.294 | 0.697 | 0.393 | 0.639 | 0.224 |
| XGBoost | 0.282 | 0.690 | 0.380 | **0.298** | **0.084** |
| **Scorecard de puntos** (modelo de gobierno) | 0.265 | 0.674 | 0.349 | — | — |

**Política de aprobación al 25%** (presupuesto US$ 133.7M; solicitado
US$ 534.7M):

| Política | Aprobados | Buenos | Malos | Morosidad |
|---|---:|---:|---:|---:|
| **Scorecard** | 9,447 | 9,107 | **340** | **3.6%** |
| Selección aleatoria | 9,402 | 8,574 | 828 | 8.8% |
| Aprobar todos | 37,450 | 34,069 | 3,381 | 9.0% |

> El scorecard genera **2.5 veces menos créditos incobrables** (340 vs 828)
> con el mismo presupuesto.

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Datos originales (xlsx) y diccionario de datos
│   └── processed/      # Datos limpios en caché (generados)
├── src/                # Código productivo (paquete Python)
│   ├── config.py       # Parámetros y reglas de negocio
│   ├── load_data.py    # Carga y normalización
│   ├── features.py     # Ingeniería de características y target
│   ├── metrics.py      # KS, AUROC, Gini, CAP/AR, Lift, LogLoss, Brier
│   ├── models.py       # Comparativa Logit / RF / XGBoost (CV + OOT)
│   ├── scorecard.py    # Scorecard WoE + IV + PDO
│   ├── approval.py     # Simulación de la política del 25%
│   └── pipeline.py     # Orquestación completa
├── notebooks/          # EDA y modelado interactivo
├── reports/            # Informe LaTeX, presentación Beamer, figuras y tablas
├── scripts/            # Punto de entrada
└── requirements.txt
```

## Reproducibilidad

```bash
pip install -r requirements.txt

# 1. Coloca los datos en data/raw/
#    (loan_ejercicio.xlsx y LCDataDictionary.xlsx)

# 2. Ejecuta el pipeline completo
python scripts/run_pipeline.py

# 3. (Opcional) Informe y presentación LaTeX
cd reports
pdflatex informe.tex && pdflatex informe.tex
pdflatex presentacion.tex && pdflatex presentacion.tex
```

El pipeline genera en `reports/`:

- **Figuras**: curvas ROC/CAP/KS, tasas de mora por decil.
- **Tablas**: comparativa de modelos (CV y OOT), scorecard de puntos,
  Information Value por variable, política de aprobación.
- **Métricas consolidadas**: `reports/tables/resumen_metricas.json`.

## Metodología

1. **Variable objetivo**: proxy de charge-off sobre préstamos resueltos;
   exclusión de censurados (evita subestimar la mora).
2. **División temporal**: desarrollo (`issue_d` < 2013-06) y validación
   *out-of-time* (2013-06 a 2015-01) con vintages nunca vistos.
3. **Comparativa**: regresión logística, random forest y XGBoost evaluados
   con métricas de negocio de riesgo de crédito en CV 5-fold y OOT.
4. **Modelo de gobierno**: scorecard de puntos (WoE, selección por
   Information Value sin colinealidad, PDO = 20) con 4 variables
   interpretables: `grade`, plazo, ingreso/crédito y utilización rotativa.
5. **Regla de negocio**: racionamiento de capital al 25% del monto
   solicitado, aprobando de menor a mayor riesgo.

## Integrantes

- Bárcena Romero, Orlando
- López Vélez, Ricardo Alejandro
- García Mercado, Karina
- Carvajal Aguilar, César Emir
- Flores, Frida

## Licencia

MIT — ver [LICENSE](LICENSE).

*Los datos provienen del ejercicio académico de la asignatura; el diccionario
de variables está documentado en `data/raw/LCDataDictionary.xlsx`.*