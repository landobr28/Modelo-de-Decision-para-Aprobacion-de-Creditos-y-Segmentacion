# Modelo de Aprobación de Créditos: Análisis Completo

## 1. Resumen Ejecutivo

Se desarrolló un modelo de aprobación de créditos binario que segmenta la base de clientes en tres categorías:
- **Bueno (Aprobado):** 73,070 clientes (24.95% de la base total)
- **Malo (Rechazado):** 219,779 clientes (75.05% de la base total)

Dentro de los aprobados, se realizó una sub-segmentación adicional:
- **Buenos más Buenos (BMB):** 69,463 clientes (95.06% de los aprobados)
- **Malos más Malos (MMM):** 3,607 clientes (4.94% de los aprobados)

## 2. Estructura de Datos

### Base de Datos Original
- **Archivo:** `loan_ejercicio_clean.xlsx`
- **Total de registros:** 292,849 clientes
- **Variables clave:**
  - `mths_since_last_delinq`: Meses desde la última morosidad (143,060 valores no nulos)
  - `dti`: Relación deuda-ingreso (Debt-to-Income)
  - `grade`: Calificación del préstamo (A, B, C, D, E, F, G)
  - `delinq_2yrs`: Número de morositades en los últimos 2 años
  - `total_rec_prncp`: Principal recuperado
  - `funded_amnt`: Monto del préstamo financiado

## 3. Lógica de Clasificación

### 3.1 Variable Objetivo (Condición Principal)

**Criterio:** `mths_since_last_delinq >= 30` **O** la celda está vacía (NaN)

**Justificación:** 
- Identifica clientes con un historial de pago limpio o sin registro de morosidad.
- El umbral de 30 meses establece una base razonable para considerar a un cliente como "buen pagador".
- La inclusión de valores NaN reconoce que clientes sin historial de morosidad también son considerados buenos pagadores.

### 3.2 Variables Secundarias (4 Condiciones)

| Condición | Variable | Criterio | Justificación |
|:---|:---|:---|:---|
| **1** | `dti` | `<= 22` | Mide la capacidad de pago. Un DTI bajo indica menor riesgo de sobreendeudamiento. |
| **2** | `grade` | `== 'A'` | La calificación más alta, indicando el menor riesgo crediticio. |
| **3** | `delinq_2yrs` | `== 0` | Ausencia total de morosidad en los últimos dos años. |
| **4** | `% Recuperación` | `>= 70%` | Cálculo: `total_rec_prncp / funded_amnt`. Mide la calidad del préstamo. |

**Justificación de la Combinación (Principal + 3/4 Secundarias):**

Se probaron diferentes combinaciones de criterios para alcanzar el objetivo de aprobar al 25% de la base total. La combinación de:
- **Condición Principal:** mths_since_last_delinq >= 30 o NaN
- **Al menos 3 de las 4 Condiciones Secundarias**

Resultó en una tasa de aprobación de **24.95%** (73,070 clientes), cumpliendo de manera óptima el requisito operativo de aprobar al 25% de la base total con una precisión de 0.05%.

### 3.3 Clasificación Inicial: Buenos vs Malos

**Criterio de "Bueno" (Aprobado):**
```
Bueno = (Condición Principal) AND (Al menos 3 de las 4 Condiciones Secundarias)
```

**Criterio de "Malo" (Rechazado):**
```
Malo = NOT (Bueno)
```

**Resultados:**
- Clientes "Buenos": 73,070 (24.95%)
- Clientes "Malos": 219,779 (75.05%)

## 4. Sub-Clasificación: BMB vs MMM

### 4.1 Buenos más Buenos (BMB)

**Criterio de BMB:**
```
BMB = (mths_since_last_delinq >= 36 OR NaN) AND (Al menos 3 de las 4 Condiciones Secundarias)
```

**Características:**
- Historial de pago más limpio (sin morosidad en los últimos 36 meses).
- Cumple al menos 3 de los 4 criterios de solvencia y calidad crediticia.
- Representa el segmento de **menor riesgo** dentro de la cartera aprobada.

**Resultados:**
- Total BMB: 69,463 clientes
- Porcentaje de aprobados: 95.06%

### 4.2 Malos más Malos (MMM)

**Criterio de MMM:**
```
MMM = Clientes Aprobados ("Buenos") - Clientes BMB
```

**Características:**
- Cumplen el criterio de aprobación inicial (mths_since_last_delinq >= 30 o NaN + 3 Secundarias).
- **No cumplen** el criterio más estricto de BMB (mths_since_last_delinq >= 36).
- Representan el **riesgo marginal** dentro del grupo aprobado.
- Historial de pago limpio, pero más reciente (30-36 meses sin morosidad).

**Resultados:**
- Total MMM: 3,607 clientes
- Porcentaje de aprobados: 4.94%

## 5. Código de Análisis (Python)

### 5.1 Herramientas y Dependencias

```python
import pandas as pd
import numpy as np
```

**Versiones recomendadas:**
- Python 3.11+
- Pandas 1.5+
- NumPy 1.23+

### 5.2 Pasos del Análisis

#### Paso 1: Cargar los datos
```python
df = pd.read_excel('loan_ejercicio_clean.xlsx')
total_clients = len(df)
```

#### Paso 2: Calcular la variable de recuperación
```python
df['recuperacion_pct'] = (df['total_rec_prncp'] / df['funded_amnt']) * 100
```

#### Paso 3: Definir las condiciones secundarias
```python
df['sec_cond_1'] = df['dti'] <= 22
df['sec_cond_2'] = df['grade'] == 'A'
df['sec_cond_3'] = (df['delinq_2yrs'].fillna(-1) == 0)
df['sec_cond_4'] = df['recuperacion_pct'] >= 70
```

**Nota sobre NaNs:** Para `delinq_2yrs`, se rellenan los valores NaN con -1 para que la condición `== 0` sea False. Esto asegura que solo los clientes con un valor explícito de 0 (sin morosidades) cumplan esta condición.

#### Paso 4: Contar condiciones secundarias cumplidas
```python
secondary_conditions = ['sec_cond_1', 'sec_cond_2', 'sec_cond_3', 'sec_cond_4']
df['num_sec_cond_cumplidas'] = df[secondary_conditions].sum(axis=1)
is_secondary_ok = df['num_sec_cond_cumplidas'] >= 3
```

#### Paso 5: Definir la condición principal (30 y 36 meses)
```python
is_principal_ok_30 = (df['mths_since_last_delinq'] >= 30) | (df['mths_since_last_delinq'].isna())
is_principal_ok_36 = (df['mths_since_last_delinq'] >= 36) | (df['mths_since_last_delinq'].isna())
```

#### Paso 6: Clasificación inicial (Bueno/Malo)
```python
is_bueno = is_principal_ok_30 & is_secondary_ok
df['clasificacion_bueno_malo'] = np.where(is_bueno, 'Bueno', 'Malo')
```

#### Paso 7: Sub-clasificación (BMB/MMM)
```python
is_bmb = is_principal_ok_36 & is_secondary_ok
is_mmm = is_bueno & (~is_bmb)

df['clasificacion_final'] = np.select(
    [is_bmb, is_mmm, is_bueno],
    ['BMB', 'MMM', 'Malo'],
    default='Malo'
)
```

#### Paso 8: Guardar resultados
```python
df.to_excel('loan_data_clasificado_final.xlsx', index=False)
```

### 5.3 Archivo de Salida

**Nombre:** `loan_data_clasificado_final.xlsx`

**Columnas Nuevas Agregadas:**
- `recuperacion_pct`: Porcentaje de recuperación calculado
- `sec_cond_1` a `sec_cond_4`: Booleanos indicando si cada condición secundaria se cumple
- `num_sec_cond_cumplidas`: Número total de condiciones secundarias cumplidas (0-4)
- `clasificacion_bueno_malo`: Clasificación inicial ('Bueno' o 'Malo')
- `clasificacion_final`: Clasificación final ('BMB', 'MMM', o 'Malo')

## 6. Validación y Verificación

### 6.1 Verificaciones Realizadas

1. **Suma de clientes:** BMB + MMM = 73,070 (igual a "Buenos")
2. **Porcentaje de aprobación:** 24.95% ≈ 25% (objetivo cumplido)
3. **Distribución dentro de aprobados:** BMB (95.06%) + MMM (4.94%) = 100%

### 6.2 Interpretación de Resultados

- **Objetivo Cumplido:** El modelo logra aprobar al 25% de la base total con una precisión de 0.05%.
- **Calidad de la Cartera:** El 95% de los aprobados son BMB, indicando una cartera de alta calidad crediticia.
- **Gestión de Riesgos:** El 5% de los aprobados (MMM) requiere un seguimiento más cercano durante la vida del préstamo.

## 7. Próximos Pasos Recomendados

1. **Validación Histórica:** Validar el modelo con datos históricos de *default* para confirmar que el grupo BMB tiene una tasa de incumplimiento significativamente menor que MMM.

2. **Integración en Sistemas:** Integrar el modelo en el sistema de *scoring* automático para aplicación en tiempo real.

3. **Monitoreo Post-Aprobación:** Establecer un plan de monitoreo continuo para el grupo MMM durante la vida del préstamo.

4. **Ajuste de Criterios:** Si es necesario, ajustar los criterios secundarios o el umbral de 3/4 condiciones basándose en el desempeño histórico del modelo.

## 8. Consideraciones Técnicas

### 8.1 Manejo de Valores Faltantes (NaN)

- **`mths_since_last_delinq`:** Los valores NaN se interpretan como "sin historial de morosidad", por lo que se incluyen en el criterio principal.
- **`delinq_2yrs`:** Los valores NaN se rellenan con -1 para que no cumplan la condición `== 0`.
- **`annual_inc`:** Aunque tiene algunos valores NaN, no se utiliza en este modelo.

### 8.2 Reproducibilidad

El código es completamente reproducible y puede ser ejecutado con cualquier base de datos de préstamos que tenga la misma estructura de columnas. Los resultados dependerán de los datos de entrada, pero la lógica y el proceso serán idénticos.

### 8.3 Escalabilidad

El código utiliza operaciones vectorizadas de Pandas, lo que lo hace eficiente incluso con bases de datos grandes (>1 millón de registros).

## 9. Conclusiones

El modelo de aprobación de créditos desarrollado:
- ✓ Cumple con la restricción operativa de aprobar al 25% de la base total
- ✓ Utiliza criterios múltiples y equilibrados para evaluar la solvencia crediticia
- ✓ Proporciona una segmentación clara entre clientes de alto riesgo (Malo) y bajo riesgo (BMB)
- ✓ Identifica un grupo de riesgo marginal (MMM) que requiere monitoreo especial
- ✓ Es reproducible, escalable y fácil de integrar en sistemas existentes
