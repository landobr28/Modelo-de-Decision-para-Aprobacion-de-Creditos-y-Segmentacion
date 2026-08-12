# -*- coding: utf-8 -*-
"""Genera las graficas profesionales del modelo de aprobacion de creditos."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.metrics import roc_curve, auc as auc_score

# ---------- Configuración ----------
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'data' / 'processed'
OUT = ROOT / 'output' / 'graficas'
OUT.mkdir(parents=True, exist_ok=True)

with open(PROCESSED / 'datos_graficas.json', encoding='utf-8') as f:
    D = json.load(f)

AZUL = '#0F2A4A'
AZUL2 = '#1B3A5C'
DORADO = '#C9A24A'
GRIS = '#5A646E'
VERDE = '#2E7D32'
ROJO = '#B33A3A'
CREMA = '#F5F2EA'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.edgecolor': GRIS,
    'axes.labelcolor': AZUL,
    'xtick.color': GRIS,
    'ytick.color': GRIS,
    'text.color': AZUL,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#E3E3E3',
    'grid.linewidth': 0.7,
})


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  guardado:', nombre)


# =====================================================================
# GRÁFICA 1: Histograma de la variable objetivo (meses de morosidad)
# =====================================================================
df = pd.read_excel(PROCESSED / 'loan_ejercicio_clean.xlsx')
mora = df['mths_since_last_delinq'].dropna()
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.hist(mora, bins=80, color=AZUL, alpha=0.85, edgecolor='white', linewidth=0.3)
for x, c, lb in [(30, DORADO, 'Umbral 30m (Bueno)'), (36, ROJO, 'Umbral 36m (BMB)')]:
    ax.axvline(x, color=c, linestyle='--', linewidth=1.8)
    ax.text(x, ax.get_ylim()[1] * 0.95, lb, color=c, fontsize=9, ha='right' if x == 30 else 'left')
ax.set_title('Distribución de mths_since_last_delinq\n(197,060 registros con historial de morosidad)',
             fontsize=12, color=AZUL, weight='bold')
ax.set_xlabel('Meses desde la última morosidad (mths_since_last_delinq)')
ax.set_ylabel('Número de clientes')
guardar(fig, '1_histograma_mora.png')

# =====================================================================
# GRÁFICA 2: Distribución por cumplimiento de condiciones secundarias
# =====================================================================
vs_dist = D['vs_dist']
vs_bad = D['vs_bad']
labels = ['0', '1', '2', '3', '4']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

bars = ax1.bar(labels, vs_dist, color=AZUL, width=0.62)
bars[-2].set_color(VERDE)   # 3 condiciones
bars[-1].set_color(DORADO)  # 4 condiciones
for b, v in zip(bars, vs_dist):
    ax1.text(b.get_x() + b.get_width() / 2, v, f'{v:,}', ha='center', va='bottom', fontsize=9, color=AZUL)
ax1.set_title('Clientes por condiciones secundarias cumplidas', fontsize=11, weight='bold')
ax1.set_xlabel('Número de condiciones secundarias cumplidas')
ax1.set_ylabel('Clientes')
ax1.set_ylim(0, max(vs_dist) * 1.12)

ax2.bar(labels, vs_bad, color=[GRIS, GRIS, GRIS, VERDE, DORADO], width=0.62)
for i, v in enumerate(vs_bad):
    ax2.text(i, v, f'{v:.0f}%', ha='center', va='bottom', fontsize=9, color=AZUL)
ax2.set_title('Tasa de morosidad real por número de condiciones', fontsize=11, weight='bold')
ax2.set_xlabel('Número de condiciones secundarias cumplidas')
ax2.set_ylabel('Tasa de morosidad (%)')
ax2.set_ylim(0, 110)
fig.tight_layout()
guardar(fig, '2_cumplimiento_condiciones.png')

# =====================================================================
# GRÁFICA 3: Embudo de aprobación (intersección de variables)
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.8))
n_total = D['N']
n_vp = D['aprob_solo_vp']
n_aprob = D['aprobados']
n_rech = D['rechazados']

pasos = [
    (f'Base total\n{n_total:,} clientes', AZUL),
    (f'Variable objetivo (VP ≥ 30m)\n{n_vp:,} clientes ({D["pct_aprob_solo_vp"]:.1f}%)', AZUL2),
    (f'VP + 3 de 4 condiciones secundarias\n{n_aprob:,} APROBADOS ({D["tasa_aprobacion"]:.2f}%)', VERDE),
]
anchos = [1.0, 0.75, 0.45]
y = len(pasos)
for (txt, col), w in zip(pasos, anchos):
    y -= 1
    ax.barh(y, w, height=0.45, color=col, alpha=0.9)
    ax.text(w / 2, y, txt, ha='center', va='center', color='white', fontsize=10, weight='bold')
ax.barh(0, 0.45, height=0.45, color=ROJO, alpha=0.9)
ax.text(0.45 / 2, 0, f'Rechazados\n{n_rech:,} clientes ({(100 - D["tasa_aprobacion"]):.2f}%)',
        ha='center', va='center', color='white', fontsize=10, weight='bold')
ax.annotate('', xy=(0.45, 0.3), xytext=(0.45, 0.7),
            arrowprops=dict(arrowstyle='-|>', color=GRIS, lw=1.8))
ax.set_xlim(0, 1.15)
ax.set_ylim(-0.6, 3.1)
ax.axis('off')
ax.set_title('Embudo de aprobación: de la base total al 25% aprobado', fontsize=12, weight='bold')
guardar(fig, '3_embudo_aprobacion.png')

# =====================================================================
# GRÁFICA 4: Distribución de segmentos (donut)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
vals = [D['BMB'], D['MMM'], D['Malo']]
cols = [VERDE, DORADO, ROJO]
labs = [f'BMB – Buenos más Buenos\n{D["BMB"]:,} ({D["pct_bmb_aprob"]:.2f}% de aprobados)',
        f'MMM – Malos más Malos\n{D["MMM"]:,} ({D["pct_mmm_aprob"]:.2f}% de aprobados)',
        f'Malo – Rechazados\n{D["Malo"]:,} ({(100 - D["tasa_aprobacion"]):.2f}%)']
wedges, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor='white'))
ax.legend(wedges, labs, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=11)
ax.set_title('Segmentación de la cartera (BMB / MMM / Malo)', fontsize=13, weight='bold')
guardar(fig, '4_segmentos_donut.png')

# =====================================================================
# GRÁFICA 5: Calidad por segmento (tasa malo + recuperación)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
segs = ['BMB', 'MMM', 'Malo']
tm = [D['tasa_malo_bmb'], D['tasa_malo_mmm'], D['tasa_malo_malo']]
rec = [D['recup_bmb'], D['recup_mmm'], D['recup_malo']]
tc = [VERDE, DORADO, ROJO]

b = ax1.bar(segs, tm, color=tc, width=0.55)
for bb, v in zip(b, tm):
    ax1.text(bb.get_x() + bb.get_width() / 2, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=11, color=AZUL)
ax1.set_title('Tasa de morosidad por segmento', fontsize=11, weight='bold')
ax1.set_ylabel('Tasa de morosidad (%)')
ax1.set_ylim(0, 100)

b2 = ax2.bar(segs, [x * 100 for x in rec], color=tc, width=0.55)
for bb, v in zip(b2, rec):
    ax2.text(bb.get_x() + bb.get_width() / 2, v * 100, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=11, color=AZUL)
ax2.set_title('Recuperación media por segmento', fontsize=11, weight='bold')
ax2.set_ylabel('Recuperación del principal (%)')
ax2.set_ylim(0, 100)
fig.tight_layout()
guardar(fig, '5_calidad_segmentos.png')

# =====================================================================
# GRÁFICA 6: Curva ROC / AUROC (sin fuga de datos)
# =====================================================================
dfg = pd.read_excel(PROCESSED / 'loan_ejercicio_clean.xlsx')
dfg['RP'] = np.where((dfg['funded_amnt'].notna()) & (dfg['funded_amnt'] != 0),
                     dfg['total_rec_prncp'] / dfg['funded_amnt'], 0)
dfg['Bad'] = (dfg['RP'] < 0.70).astype(int)
dfg['VS1'] = (dfg['dti'] <= 22).astype(int)
dfg['VS2'] = (dfg['grade'] == 'A').astype(int)
dfg['VS3'] = (dfg['delinq_2yrs'] == 0).astype(int)
score = dfg['VS1'] + dfg['VS2'] + dfg['VS3']
score_maldad = 3 - score

fpr, tpr, _ = roc_curve(dfg['Bad'], score_maldad)
auroc = auc_score(fpr, tpr)
gini = 2 * auroc - 1

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color=AZUL, lw=2.4, label=f'Modelo (AUROC = {auroc:.3f})')
ax.plot([0, 1], [0, 1], color=GRIS, ls='--', lw=1.4, label='Modelo aleatorio (AUROC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.12, color=AZUL)
ax.set_xlabel('Tasa de falsos positivos (1 − Especificidad)')
ax.set_ylabel('Tasa de verdaderos positivos (Sensibilidad)')
ax.set_title('Curva ROC del modelo de reglas\n(sin fuga de datos)', fontsize=12, weight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
guardar(fig, '6_curva_roc.png')

print(f'AUROC grafica: {auroc:.4f} | GINI: {gini:.4f}')

# =====================================================================
# GRÁFICA 7: Curva KS
# =====================================================================
aux = pd.DataFrame({'score': score_maldad, 'bad': dfg['Bad']})
g = aux.groupby('score').agg(Total=('bad', 'count'), Bad=('bad', 'sum')).reset_index()
g['Good'] = g['Total'] - g['Bad']
g = g.sort_values('score')
g['Cum_Bad'] = g['Bad'].cumsum()
g['Cum_Good'] = g['Good'].cumsum()
tb, tg = g['Bad'].sum(), g['Good'].sum()
g['Pct_Bad'] = g['Cum_Bad'] / tb
g['Pct_Good'] = g['Cum_Good'] / tg
g['KS'] = (g['Pct_Bad'] - g['Pct_Good']).abs()
ks = g['KS'].max()
ks_idx = g['KS'].idxmax()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(g['score'], g['Pct_Bad'] * 100, color=ROJO, lw=2.2, marker='o', ms=4, label='Acumulado malos')
ax.plot(g['score'], g['Pct_Good'] * 100, color=VERDE, lw=2.2, marker='o', ms=4, label='Acumulado buenos')
for s in g['score']:
    row = g[g['score'] == s].iloc[0]
    ax.plot([s, s], [row['Pct_Good'] * 100, row['Pct_Bad'] * 100], color=GRIS, lw=1.4, alpha=0.75)
krow = g.loc[ks_idx]
ax.axvline(krow['score'], color=DORADO, ls='--', lw=1.6)
ax.annotate(f'KS = {ks:.3f}', xy=(krow['score'], (krow['Pct_Good'] * 100 + krow['Pct_Bad'] * 100) / 2),
            xytext=(krow['score'] + 0.15, 50), fontsize=11, color=DORADO, weight='bold',
            arrowprops=dict(arrowstyle='->', color=DORADO))
ax.set_xlabel('Score de maldad (mayor = peor cliente)')
ax.set_ylabel('Porcentaje acumulado (%)')
ax.set_title('Gráfica de Kolmogorov–Smirnov (KS)', fontsize=12, weight='bold')
ax.legend(loc='center right', fontsize=10)
ax.set_ylim(0, 105)
guardar(fig, '7_curva_ks.png')
print(f'KS grafica: {ks:.4f}')

print('\nTodas las graficas generadas en:', OUT)
