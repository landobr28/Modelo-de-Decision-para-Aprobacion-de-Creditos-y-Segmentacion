import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# --- PARTE 1: Carga de Datos, Definición de Target y Cálculo del Pseudo-Score ---

# Cargar el archivo (Asegúrate de que la ruta sea correcta)
try:
    # Usé el archivo 'loan_ejercicio_clean.xlsx' que me proporcionaste
    df = pd.read_excel('loan_ejercicio_clean.xlsx')
except FileNotFoundError:
    print("Error: Asegúrate de que 'loan_ejercicio_clean.xlsx' esté en el directorio correcto.")
    exit()

# 1. Definir Variable Target (Bad_Loan)
# Target = 1 (Mal Préstamo): Si Recuperacion_Pct < 0.70
# Target = 0 (Buen Préstamo): Si Recuperacion_Pct >= 0.70
df['Recuperacion_Pct'] = np.where(
    (df['funded_amnt'].notna()) & (df['funded_amnt'] != 0),
    df['total_rec_prncp'] / df['funded_amnt'],
    0
)
df['Bad_Loan'] = np.where(df['Recuperacion_Pct'] < 0.70, 1, 0)


# 2. Aplicar Clasificación y calcular Pseudo-Score (VS_Count)

# VP: mths_since_last_delinq >= 30 O la celda está vacía (NaN)
df['VP_Cumple'] = (df['mths_since_last_delinq'].fillna(np.inf) >= 30)

# VS C1: dti <= 22
df['VS_C1'] = (df['dti'] <= 22)

# VS C2: grade == 'A'
df['VS_C2'] = (df['grade'] == 'A')

# VS C3: delinq_2yrs == 0
df['VS_C3'] = (df['delinq_2yrs'] == 0)

# VS C4: Recuperacion_Pct >= 0.70
df['VS_C4'] = (df['Recuperacion_Pct'] >= 0.70)

# Pseudo-Score: Suma de las VS cumplidas (más alto = mejor cliente)
df['VS_Count'] = df[['VS_C1', 'VS_C2', 'VS_C3', 'VS_C4']].sum(axis=1)

# --- PARTE 2: Cálculo de Métricas de Discriminación ---

# El Pseudo-Score (VS_Count) es un score de 'Bondad'.
# Para AUROC/KS, necesitamos un score de 'Maldad' (más alto = peor cliente).
# Score_Maldad = 4 - VS_Count (donde 4 es el máximo de VS_Count)
df['Score_Maldad'] = 4 - df['VS_Count']

# 1. Calcular AUROC
try:
    auc = roc_auc_score(df['Bad_Loan'], df['Score_Maldad'])
except ValueError:
    # Esto ocurre si solo hay una clase en Bad_Loan
    auc = 0.5

# 2. Calcular GINI
gini = 2 * auc - 1

# 3. Calcular KS (Kolmogorov-Smirnov)
# Agrupar por score y calcular las tasas de buenos y malos
df_ks = df.groupby('Score_Maldad').agg(
    Total=('Bad_Loan', 'count'),
    Bad=('Bad_Loan', 'sum')
).reset_index()

df_ks['Good'] = df_ks['Total'] - df_ks['Bad']
df_ks = df_ks.sort_values(by='Score_Maldad', ascending=True)

# Calcular acumulados
df_ks['Cum_Bad'] = df_ks['Bad'].cumsum()
df_ks['Cum_Good'] = df_ks['Good'].cumsum()

# Calcular porcentajes acumulados
total_bad = df_ks['Bad'].sum()
total_good = df_ks['Good'].sum()

df_ks['Pct_Cum_Bad'] = df_ks['Cum_Bad'] / total_bad
df_ks['Pct_Cum_Good'] = df_ks['Cum_Good'] / total_good

# Calcular la diferencia (KS)
df_ks['KS'] = np.abs(df_ks['Pct_Cum_Bad'] - df_ks['Pct_Cum_Good'])
ks_statistic = df_ks['KS'].max()

# --- RESULTADOS ---
print("--- Métricas de Desempeño del Modelo de Reglas ---")
print(f"AUROC: {auc:.4f}")
print(f"GINI: {gini:.4f}")
print(f"KS: {ks_statistic:.4f}")

