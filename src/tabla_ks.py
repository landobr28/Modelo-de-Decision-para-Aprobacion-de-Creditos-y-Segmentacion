import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_excel(ROOT / 'data' / 'processed' / 'loan_ejercicio_clean.xlsx')
N = len(df)
df['RP'] = np.where((df['funded_amnt'].notna()) & (df['funded_amnt'] != 0), df['total_rec_prncp'] / df['funded_amnt'], 0)
df['Bad'] = (df['RP'] < 0.70).astype(int)
df['VS1'] = (df['dti'] <= 22).astype(int)
df['VS2'] = (df['grade'] == 'A').astype(int)
df['VS3'] = (df['delinq_2yrs'] == 0).astype(int)
df['VS4'] = (df['RP'] >= 0.70).astype(int)
df['SecOk'] = df[['VS1', 'VS2', 'VS3', 'VS4']].sum(axis=1) >= 3
df['VP30'] = (df['mths_since_last_delinq'].fillna(np.inf) >= 30)
df['VP36'] = (df['mths_since_last_delinq'].fillna(np.inf) >= 36)
seg = np.select([df['VP36'] & df['SecOk'], df['VP30'] & df['SecOk']], ['BMB', 'MMM'], default='Malo')
df['Seg'] = seg

tot_bad = df['Bad'].sum()
print('Total malos:', tot_bad, f'({tot_bad/N*100:.2f}%)')
print()
for s in ['Malo', 'MMM', 'BMB']:
    sub = df[df['Seg'] == s]
    b = sub['Bad'].sum()
    print(f'{s:5s} | cuentas {len(sub):>7} | %base {len(sub)/N*100:6.2f}% | malos {b:>7} | {b/len(sub)*100:5.1f}% seg | {b/tot_bad*100:5.2f}% del total malos')
print()
# Lift: que % de malos captura rechazar 75%
rechazados = df[df['Seg'] == 'Malo']
print('Rechazando 75% de solicitudes, capturamos', rechazados['Bad'].sum(), f'= {(rechazados["Bad"].sum()/tot_bad)*100:.2f}% de los malos')
aprob = df[df['Seg'] != 'Malo']
print('Aprobando 25%, malos dentro aprobados:', aprob['Bad'].sum(), f'= {(aprob["Bad"].sum()/len(aprob))*100:.2f}% de los aprobados estan malos')
print('Buenos reales capturados:', ((df['Seg'] != 'Malo') & (df['Bad'] == 0)).sum(), '/', (df['Bad'] == 0).sum(),
      '=', f'{((df["Seg"] != "Malo") & (df["Bad"] == 0)).sum()/(df["Bad"] == 0).sum()*100:.2f}%')
