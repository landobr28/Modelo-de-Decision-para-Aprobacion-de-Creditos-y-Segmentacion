# Modelo de Decisión para la Aprobación de Créditos y Segmentación de Riesgo

Modelo de reglas de decisión (*scorecard* interpretable) para la aprobación de créditos y la segmentación de la cartera crediticia. Proyecto desarrollado en la asignatura **Administración Actuarial** de la Licenciatura en Actuaría, **FES Acatlán · UNAM**, bajo la dirección del **Prof. Tapia Martínez Omar Alejandro**.

---

## Estructura de ramas

Este repositorio organiza la evolución del proyecto en tres ramas:

| Rama | Descripción |
|:---|:---|
| **`master`** | Versión original: modelo de reglas de decisión (scorecard interpretable). |
| **`mejora`** | Mejora de `master`: pipeline MLOps, scorecard estadístico (regresión logística), informe y presentación. |
| **`proyecto`** | Rehecho desde cero sobre la base del modelo original: pipeline modular (`src/`), scorecard con validación CV/OOT, informe LaTeX. |

```bash
git branch -a          # listar ramas
git switch mejora      # cambiar a la versión mejorada
git switch proyecto    # cambiar a la versión rehecha
```

---

## Tabla de contenido

1. [Resumen ejecutivo](#resumen-ejecutivo)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Datos](#datos)
4. [Metodología](#metodolog%C3%ADa)
5. [Resultados](#resultados)
6. [Métricas de desempeño](#m%C3%A9tricas-de-desempe%C3%B1o)
7. [Validación y control de calidad](#validaci%C3%B3n-y-control-de-calidad)
8. [Cómo reproducir](#c%C3%B3mo-reproducir)
9. [Conclusiones](#conclusiones)

---

## Resumen ejecutivo

Se desarrolló un modelo binario de decisión que aprueba o rechaza solicitudes de crédito mediante una **condición principal** (historial de morosidad) combinada con **al menos 3 de 4 condiciones secundarias** de solvencia financiera.

- **Tasa de aprobación:** 24.95% (objetivo operativo: 25%).
- **Aprobados (Bueno):** 73,070 clientes.
- **Rechazados (Malo):** 219,779 clientes.
- **Sub-segmento BMB (Buenos más Buenos):** 69,463 clientes (95.06% de los aprobados).
- **Sub-segmento MMM (Malos más Malos):** 3,607 clientes (4.94% de los aprobados).

El modelo separa de forma efectiva el riesgo: los rechazados concentran una tasa de incumplimiento de **86.5%**, mientras que los aprobados de **28.5%**; además, la recuperación media de los aprobados (76.7%) más que duplica a la de los rechazados (30.7%).

---

## Estructura del repositorio

```
.
├── README.md                                # Documentación del proyecto
├── requirements.txt                         # Dependencias
├── .gitignore
├── src/                                     # Código Python
│   ├── analyze_loan_data.py                 # Clasificación Bueno/Malo y sub-segmentación BMB/MMM
│   ├── metricas_desempeño.py                # Métricas de desempeño (AUROC, GINI, KS)
│   ├── datos_graficas.py                    # Genera datos auxiliares para las gráficas
│   ├── graficas_modelo.py                   # Genera las gráficas profesionales
│   └── tabla_ks.py                          # Tabla de captura de morosos por segmento
├── data/
│   ├── raw/                                 # Datos originales sin modificar
│   │   ├── loan_ejercicio.xlsx
│   │   └── LCDataDictionary.xlsx            # Diccionario de datos
│   └── processed/                           # Datos limpios y resultados
│       ├── loan_ejercicio_clean.xlsx        # Base limpia de análisis
│       ├── loan_data_clasificado_final.xlsx # Salida: base clasificada
│       └── datos_graficas.json              # Métricas para las gráficas
├── output/
│   └── graficas/                            # Gráficas (PNG) del modelo
│       ├── 1_histograma_mora.png
│       ├── 2_cumplimiento_condiciones.png
│       ├── 3_embudo_aprobacion.png
│       ├── 4_segmentos_donut.png
│       ├── 5_calidad_segmentos.png
│       ├── 6_curva_roc.png
│       └── 7_curva_ks.png
└── Presentación_Modelo_Aprobación_Créditos.pptx  # Presentación del proyecto (PPT con las gráficas)
```

---

## Datos

- **Fuente:** cartera crediticia tipo LendingClub.
- **Registros:** 292,849 solicitudes.
- **Variables principales:**
  - `mths_since_last_delinq`: meses desde la última morosidad (NaN = sin historial de morosidad).
  - `dti`: relación deuda-ingreso (debt-to-income).
  - `grade`: calificación del préstamo (A a G).
  - `delinq_2yrs`: número de morosidades en los últimos 2 años.
  - `total_rec_prncp` / `funded_amnt`: principal recuperado y monto financiado → recuperación.

---

## Metodología

**Condición principal (VP):**

```
mths_since_last_delinq >= 30  O  celda vacía (NaN)
```

**Condiciones secundarias (VS):**

| Condición | Variable | Criterio |
|:---|:---|:---|
| VS1 | dti | ≤ 22 |
| VS2 | grade | = A |
| VS3 | delinq_2yrs | = 0 |
| VS4 | Recuperación (principal recuperado / monto final) | ≥ 70% |

**Regla de decisión:**

```
Aprobado (Bueno) = VP  Y  (al menos 3 de 4 VS)
Rechazado (Malo) = no cumple la regla

BMB = VP36 (>= 36 meses)  Y  (al menos 3 de 4 VS)
MMM = Aprobado que no alcanza BMB
```

---

## Resultados

| Clasificación | Clientes | % del total | % de aprobados |
|:---|---:|---:|---:|
| **Bueno** (aprobado) | 73,070 | 24.95% | 100% |
| └ BMB | 69,463 | 23.72% | 95.06% |
| └ MMM | 3,607 | 1.23% | 4.94% |
| **Malo** (rechazado) | 219,779 | 75.05% | — |
| **Total** | **292,849** | **100%** | — |

### Calidad por segmento

| Segmento | Tasa de malo | Recuperación media | dti medio | % base |
|:---|---:|---:|---:|---:|
| BMB | 28.5% | 76.7% | 13.5 | 23.72% |
| MMM | 29.4% | 75.4% | 13.2 | 1.23% |
| Malo | 86.5% | 30.7% | 19.7 | 75.05% |

---

## Métricas de desempeño

El poder discriminatorio del score se mide sobre las variables de comportamiento del solicitante (dti, grade, delinq_2yrs), reportando métricas conservadoras y sin sesgo.

| Métrica | Valor |
|:---|---:|
| **AUROC** (área bajo la curva ROC) | 0.570 |
| **GINI** (2·AUROC − 1) | 0.141 |
| **KS** (Kolmogorov-Smirnov) | 0.126 |

> **Nota técnica (control de calidad):** se detectó y eliminó una **fuga de datos** (*target leakage*). La variable de recuperación (VS4) coincide con la definición del target; si se incluyera como predictor las métricas se inflarían artificialmente (AUROC ≈ 0.85). El script `metricas_desempeño.py` las calcula de forma honesta usando solo variables observables.

---

## Validación y control de calidad

- Reproducibilidad verificada con auditoría independiente (carga, clasificación y métricas).
- Consistencia confirmada: `BMB + MMM = 73,070 = Bueno`.
- No existen montos financiados en cero ni divisiones inválidas.
- Tratamiento explícito de valores NaN para no sesgar la clasificación.

---

## Cómo reproducir

**Requisitos:** Python 3.11+ · pandas · numpy · scikit-learn · openpyxl · matplotlib · python-pptx

```bash
# 1. Clasificar la base (genera data/processed/loan_data_clasificado_final.xlsx)
python src/analyze_loan_data.py

# 2. Calcular métricas de desempeño
python src/metricas_desempeño.py

# 3. Generar los datos y gráficas del modelo
python src/datos_graficas.py
python src/graficas_modelo.py
```

---

## Conclusiones

- ✅ Se cumple la restricción operativa del 25% de aprobación (24.95%).
- ✅ Segmentación clara de riesgo: 95.1% de los aprobados son BMB (bajo riesgo).
- ✅ Los rechazados concentran el 86.5% de los incumplimientos.
- ✅ Modelo interpretable, auditable y reproducible.

**Mejoras futuras:** ajuste de umbrales con validación histórica, incorporación de más variables (ingreso, utilización de crédito) y complemento con un score estadístico conservando la interpretabilidad.
