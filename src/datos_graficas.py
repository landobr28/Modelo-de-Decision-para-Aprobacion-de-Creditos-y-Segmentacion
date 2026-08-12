import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'data' / 'processed'

df = pd.read_excel(PROCESSED / 'loan_ejercicio_clean.xlsx')
N = len(df)

df['RP'] = np.where((df['funded_amnt'].notna()) & (df['funded_amnt'] != 0),
                    df['total_rec_prncp'] / df['funded_amnt'], 0)
df['Bad'] = (df['RP'] < 0.70).astype(int)
df['VS1'] = (df['dti'] <= 22).astype(int)
df['VS2'] = (df['grade'] == 'A').astype(int)
df['VS3'] = (df['delinq_2yrs'] == 0).astype(int)
df['VS4'] = (df['RP'] >= 0.70).astype(int)
df['VS_Count'] = df[['VS1', 'VS2', 'VS3', 'VS4']].sum(axis=1)
df['VP30'] = (df['mths_since_last_delinq'].fillna(np.inf) >= 30).astype(int)
df['VP36'] = (df['mths_since_last_delinq'].fillna(np.inf) >= 36).astype(int)
df['SecOk'] = (df['VS_Count'] >= 3).astype(int)

seg = np.select([(df['VP36'] == 1) & (df['SecOk'] == 1), (df['VP30'] == 1) & (df['SecOk'] == 1)],
                ['BMB', 'MMM'], default='Malo')
df['Seg'] = seg

datos = {
    'N': int(N),
    'aprobados': int((df['Seg'] != 'Malo').sum()),
    'rechazados': int((df['Seg'] == 'Malo').sum()),
    'tasa_aprobacion': float((df['Seg'] != 'Malo').mean() * 100),
    'BMB': int((df['Seg'] == 'BMB').sum()),
    'MMM': int((df['Seg'] == 'MMM').sum()),
    'Malo': int((df['Seg'] == 'Malo').sum()),
    'pct_bmb_aprob': float((df['Seg'] == 'BMB').sum() / max(1, (df['Seg'] != 'Malo').sum()) * 100),
    'pct_mmm_aprob': float((df['Seg'] == 'MMM').sum() / max(1, (df['Seg'] != 'Malo').sum()) * 100),
    'tasa_malo_bmb': float(df.loc[df['Seg'] == 'BMB', 'Bad'].mean() * 100),
    'tasa_malo_mmm': float(df.loc[df['Seg'] == 'MMM', 'Bad'].mean() * 100),
    'tasa_malo_malo': float(df.loc[df['Seg'] == 'Malo', 'Bad'].mean() * 100),
    'recup_bmb': float(df.loc[df['Seg'] == 'BMB', 'RP'].mean()),
    'recup_mmm': float(df.loc[df['Seg'] == 'MMM', 'RP'].mean()),
    'recup_malo': float(df.loc[df['Seg'] == 'Malo', 'RP'].mean()),
    'dti_bmb': float(df.loc[df['Seg'] == 'BMB', 'dti'].mean()),
    'dti_mmm': float(df.loc[df['Seg'] == 'MMM', 'dti'].mean()),
    'dti_malo': float(df.loc[df['Seg'] == 'Malo', 'dti'].mean()),
    'aprob_solo_vp': int((df['VP30'] == 1).sum()),
    'pct_aprob_solo_vp': float((df['VP30'] == 1).mean() * 100),
    'vs_dist': [int((df['VS_Count'] == k).sum()) for k in range(5)],
    'vs_bad': [float(df.loc[df['VS_Count'] == k, 'Bad'].mean() * 100) for k in range(5)],
    'vs_pct': [float((df['VS_Count'] == k).mean() * 100) for k in range(5)],
}
OUT = ROOT / 'data' / 'processed'
with open(OUT / 'datos_graficas.json', 'w', encoding='utf-8') as f:
    json.dump(datos, f, ensure_ascii=False, indent=2)

print('Datos para graficas generados:')
print('  aprobados:', datos['aprobados'], '| rechazados:', datos['rechazados'], '| tasa:', round(datos['tasa_aprobacion'], 2))
print('  BMB:', datos['BMB'], '| MMM:', datos['MMM'])
print('  Aprobados solo con VP30:', datos['aprob_solo_vp'], f"({datos['pct_aprob_solo_vp']:.2f}%)")
print('  Distribucion VS_Count 0-4:', datos['vs_dist'])
print('  Tasa mala por VS_Count 0-4:', [round(x, 1) for x in datos['vs_bad']])
