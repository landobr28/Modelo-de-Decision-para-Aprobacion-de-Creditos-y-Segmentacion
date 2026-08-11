# Credit Scoring Decision Model

Modelo estadístico de decisión de crédito que **aprueba únicamente el 25% de las
solicitudes recibidas** y clasifica a los aceptados en dos segmentos de riesgo
(**Buenos** y **Malos**), construido sobre datos públicos de préstamos de
**Lending Club**. Proyecto desarrollado en la asignatura de **Administración
Actuarial** de la Licenciatura en Actuaría.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Estado-Terminado-brightgreen)

**Autor:** Orlando Barcena Romero — Licenciatura en Actuaría, Facultad de
Estudios Superiores Acatlán, UNAM.

---

## Objetivo del negocio

Una institución financiera recibe más solicitudes de crédito de las que puede
aprobar con seguridad. La política operativa exige:

| Regla | Valor |
|-------|-------|
| Tasa de aprobación máxima | **25%** de las solicitudes |
| Criterio de calidad | Aprobar las solicitudes de **menor riesgo** |
| Seguimiento de cartera | Clasificar a los aprobados en **Buenos** y **Malos** |

## Solución

1. **Variable objetivo:** un préstamo es *malo* si la recuperación de principal
   observada es inferior al 70% del monto financiado (proxy de incumplimiento).
2. **Modelo:** regresión logística calibrada sobre una matriz de **44 variables
   de solicitud** —información disponible al momento de la solicitud, sin fuga
   de información (*leakage*)— con validación en muestra de test (70/30).
3. **Scorecard:** el modelo se transforma a una escala de puntos tipo FICO
   (factor de 20 puntos por duplicación de odds, corte en **600**):
   - `score >= 600` → **Aprobado** (25% con menor riesgo);
   - dentro de aprobados, `score >= 613` → **Bueno**, si no → **Malo**.
4. **Comparación:** se evalúa contra el modelo de reglas del ejercicio original
   de la asignatura, evidenciando su sesgo (*leakage*) y la superioridad del
   enfoque estadístico.

## Resultados principales

| Métrica (muestra de test) | Valor |
|---|---|
| Tasa de aprobación | **24.88%** (objetivo 25%) |
| AUROC | **0.7462** |
| GINI | **0.4924** |
| KS | **0.3700** |
| Préstamos malos en la cartera aprobada | **45.8%** |
| Préstamos malos en solicitudes rechazadas | **80.8%** |
| Captura de malos (excluidos de la cartera) | **84.2%** |
| Tasa de malos en segmento *Bueno* | **37.5%** |
| Tasa de malos en segmento *Malo* | **54.3%** |

**Comparación con el modelo de reglas del ejercicio (al 25% de aprobación):**

| Método | AUROC | Malos en la cartera aprobada |
|---|---|---|
| Regresión logística (este proyecto) | **0.7462** | **45.8%** |
| Reglas del ejercicio (sin leakage) | 0.5704 | 67.8% |
| Reglas del ejercicio (con recuperación) | 0.8538 | 28.6% |

> La versión "con recuperación" usa la variable de resultado en la regla de
> decisión, lo que infla artificialmente sus métricas (leakage). La comparación
> honesta es contra la columna "sin leakage", donde el modelo estadístico
> reduce **un tercio** la proporción de malos en la cartera.

## Estructura del repositorio

```
credit-scoring-decision-model/
├── data/
│   ├── raw/                  # Datos públicos de Lending Club + diccionario
│   └── processed/            # Dataset de trabajo generado por el pipeline
├── src/
│   ├── config.py             # Parámetros y rutas centralizadas
│   ├── data_prep.py          # Limpieza, ingeniería de variables y target
│   ├── scoring.py            # Modelo, scorecard y reglas de decisión
│   ├── metrics.py            # AUROC, GINI, KS y métricas de negocio
│   └── plots.py              # Figuras con estilo unificado
├── scripts/
│   ├── 00_preprocess.py      # Datos crudos -> dataset de trabajo
│   ├── 01_eda.py             # Análisis exploratorio + figuras
│   ├── 02_train.py           # Entrenamiento, validación y comparación
│   └── 03_export.py          # Exportación de resultados finales
├── outputs/
│   ├── loan_data_clasificado_final.xlsx
│   ├── figures/              # Figuras del informe
│   ├── tables/               # Tablas de métricas
│   ├── modelo_logistico.joblib
│   └── reglas_decision.json  # Umbrales de decisión exportables
├── reports/
│   ├── informe_latex/        # Informe completo (LaTeX/PDF)
│   └── presentacion_latex/   # Presentación (Beamer/PDF)
├── notebooks/
│   └── 01_exploracion.ipynb  # Cuaderno de exploración
├── run_pipeline.py           # Orquestador del pipeline
└── requirements.txt
```

## Requisitos e instalación

- Python 3.11 o superior.

```bash
git clone https://github.com/<tu-usuario>/credit-scoring-decision-model.git
cd credit-scoring-decision-model
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Uso

```bash
python run_pipeline.py                # Ejecuta las 4 etapas en orden
python run_pipeline.py --etapa 0      # Solo preprocesamiento
python run_pipeline.py --etapa 2      # Solo entrenamiento y evaluación
```

Salidas principales en `outputs/`:

- `loan_data_clasificado_final.xlsx` — base completa con `score`, `decision`
  (Aprobado/Rechazado) y `segmento` (Bueno/Malo).
- `reglas_decision.json` — scorecard y umbrales listos para integrarse a un
  sistema de originación.
- `tables/` y `figures/` — métricas y gráficas utilizadas en el informe.

## Documentos

- [Informe completo (LaTeX/PDF)](reports/informe_latex/) — metodología, marco
  teórico, resultados y conclusiones.
- [Presentación (Beamer/PDF)](reports/presentacion_latex/) — resumen ejecutivo
  en diapositivas.

## Datos

- Fuente: [Lending Club](https://www.lendingclub.com) (datos públicos de
  préstamos personales).
- Definiciones de variables: `LCDataDictionary.xlsx`.
- Los datos de Lending Club se distribuyen públicamente; ver términos del
  originador para uso académico.

## Licencia

MIT — ver `LICENSE`. Los datos son propiedad de Lending Club y se utilizan
exclusivamente con fines académicos.