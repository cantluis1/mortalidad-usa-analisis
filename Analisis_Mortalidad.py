# =============================================================================
#  ANÁLISIS TEMPORAL Y DEMOGRÁFICO DE LAS PRINCIPALES CAUSAS DE MUERTE EN EE.UU.
#  Período: 1999–2017  |  Fuente: CDC – NCHS Leading Causes of Death
#  CÓDIGO COMPLETO CON CORRECCIONES Y SECCIONES NUEVAS
# =============================================================================
#
#  INSTRUCCIONES:
#  1. Ejecuta primero la celda de instalación (pip install ...) si no tienes
#     pmdarima o scikit-learn instalados.
#  2. Pega cada bloque separado por "# ──" como una celda nueva en Jupyter.
#  3. El CSV debe estar en la misma carpeta que el notebook.
# =============================================================================


# ── INSTALACIONES (ejecutar solo una vez) ─────────────────────────────────────
# pip install pmdarima scikit-learn statsmodels pandas numpy matplotlib seaborn


# =============================================================================
#  CELDA 1 – LIBRERÍAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from IPython.display import display, HTML

# Configuración global de estilo
plt.rcParams.update({
    'figure.dpi': 120,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print("✓ Librerías cargadas correctamente.")


# =============================================================================
#  CELDA 2 – CARGA Y LIMPIEZA DEL DATASET
# =============================================================================

FILE = r"C:\Users\luisc\proyecto-mortalidad\NCHS_-_Leading_Causes_of_Death__United_States_20260227.csv"
df_raw = pd.read_csv(FILE)
print(f"Shape original: {df_raw.shape}")

# Limpiar nombres de columnas
df_raw.columns = (
    df_raw.columns.str.strip().str.lower()
    .str.replace(' ', '_').str.replace('-', '_')
)

# Renombrar columnas clave
rename_map = {
    'year'                 : 'anio',
    '113_cause_name'       : 'causa_codigo',
    'cause_name'           : 'causa',
    'state'                : 'estado',
    'deaths'               : 'muertes',
    'age_adjusted_death_rate': 'tasa_ajustada'
}
rename_map = {k: v for k, v in rename_map.items() if k in df_raw.columns}
df = df_raw.rename(columns=rename_map).copy()

# Convertir tipos
df['muertes']      = pd.to_numeric(df['muertes'].astype(str).str.replace(',', '', regex=False), errors='coerce')
df['tasa_ajustada']= pd.to_numeric(df['tasa_ajustada'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
df['anio']         = pd.to_numeric(df['anio'], errors='coerce')

# Dataset nacional (United States, 1999–2017)
datos = df[
    (df['estado'] == 'United States') &
    (df['anio'] >= 1999) & (df['anio'] <= 2017)
][['anio', 'causa', 'muertes', 'tasa_ajustada']].copy()

# Dataset por estados (excluir United States)
datos_estados = df[
    (df['estado'] != 'United States') &
    (df['anio'] >= 1999) & (df['anio'] <= 2017)
].copy()

# Totales nacionales por año
totales_anuales = datos[datos['causa'] == 'All causes'][['anio', 'muertes']].copy()
print("\nTotales nacionales por año:")
print(totales_anuales.to_string(index=False))

# Total acumulado por causa
totales_causa = (
    datos[datos['causa'] != 'All causes']
    .groupby('causa')['muertes'].sum()
    .sort_values(ascending=False)
    .reset_index(name='total_1999_2017')
)
print("\nTotal acumulado por causa (1999–2017):")
print(totales_causa.to_string(index=False))


# =============================================================================
#  CELDA 3 – SECCIÓN 3.3  Comparación porcentual 1999 vs 2017
# =============================================================================

datos_comp = datos[
    (datos['anio'].isin([1999, 2017])) & (datos['causa'] != 'All causes')
].copy()
datos_comp['total_anual'] = datos_comp.groupby('anio')['muertes'].transform('sum')
datos_comp['porcentaje']  = datos_comp['muertes'] / datos_comp['total_anual'] * 100

orden_causas = (
    datos_comp[datos_comp['anio'] == 2017]
    .sort_values('porcentaje', ascending=False)['causa'].tolist()
)
pivot = datos_comp.pivot(index='causa', columns='anio', values='porcentaje').loc[orden_causas]

fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(pivot))
h = 0.35
bars1 = ax.barh(y + h/2, pivot[1999], h, label='1999', color='#6baed6', alpha=0.9)
bars2 = ax.barh(y - h/2, pivot[2017], h, label='2017', color='#2171b5', alpha=0.9)
for bar in list(bars1) + list(bars2):
    w = bar.get_width()
    ax.text(w + 0.1, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(pivot.index, fontsize=10)
ax.set_xlabel('Porcentaje del total anual', fontweight='bold')
ax.set_title('Comparación porcentual de muertes por causa\n1999 vs 2017 – Total Nacional', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, pivot.max().max() * 1.18)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 4 – SECCIÓN 4  Evolución de la tasa ajustada anual
# =============================================================================

total_anual = datos[datos['causa'] == 'All causes'].sort_values('anio')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(total_anual['anio'], total_anual['tasa_ajustada'], color='#2C3E50', linewidth=1.8, label='Tasa ajustada')
ax.scatter(total_anual['anio'], total_anual['tasa_ajustada'], color='#E74C3C', s=40, zorder=5)

slope, intercept, *_ = stats.linregress(total_anual['anio'], total_anual['tasa_ajustada'])
x_fit = np.array([1999, 2017])
ax.plot(x_fit, slope * x_fit + intercept, color='#18BC9C', linestyle='--', linewidth=1.5, label='Tendencia lineal')

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Tasa ajustada (por 100,000 hab.)', fontweight='bold')
ax.set_title('Evolución de la tasa ajustada de mortalidad nacional\n1999–2017', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 5 – SECCIÓN 5  Ranking de causas por año
# =============================================================================

top10_tasas = (
    datos[datos['causa'] != 'All causes']
    .groupby('causa')['tasa_ajustada'].mean()
    .nlargest(10).index.tolist()
)
datos_tasas_nac = (
    datos[datos['causa'].isin(top10_tasas)]
    .groupby(['anio', 'causa'], as_index=False)['tasa_ajustada'].mean()
    .rename(columns={'tasa_ajustada': 'tasa_nac'})
)
base_colors  = [plt.cm.tab10(i) for i in range(9)] + [(0.2, 0.2, 0.2, 1)]
causas_ord   = sorted(datos_tasas_nac['causa'].unique())
color_map    = dict(zip(causas_ord, base_colors[:len(causas_ord)]))

fig, ax = plt.subplots(figsize=(12, 6))
for causa, grp in datos_tasas_nac.groupby('causa'):
    ax.plot(grp['anio'], grp['tasa_nac'], label=causa, color=color_map[causa], linewidth=2, marker='o', markersize=4)
ax.set_xticks(range(1999, 2018, 2))
ax.tick_params(axis='x', rotation=45)
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Tasa por 100,000 habitantes', fontweight='bold')
ax.set_title('Evolución de las tasas de mortalidad ajustadas por edad\nPromedio nacional – Top 10 causas (1999–2017)', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 6 – SECCIÓN 5.2  Cambios en el ranking 1999 vs 2017
# =============================================================================

datos_nacional = datos[datos['causa'] != 'All causes'].copy()
datos_nacional['rank'] = datos_nacional.groupby('anio')['muertes'].rank(ascending=False, method='min').astype(int)

def get_rank(year):
    return (
        datos_nacional[datos_nacional['anio'] == year][['causa', 'rank', 'muertes']]
        .rename(columns={'rank': f'rank_{year}', 'muertes': f'muertes_{year}'})
    )

cambios = get_rank(1999).merge(get_rank(2017), on='causa')
cambios['cambio_rank'] = cambios['rank_2017'] - cambios['rank_1999']
cambios['pct_cambio']  = ((cambios['muertes_2017'] - cambios['muertes_1999']) / cambios['muertes_1999'] * 100).round(1)
cambios['cambio_fmt']  = cambios['cambio_rank'].apply(
    lambda x: f'▲{abs(int(x))}' if x < 0 else (f'▼{int(x)}' if x > 0 else '=')
)
tabla_cambios = (
    cambios[['causa', 'rank_1999', 'rank_2017', 'cambio_fmt', 'pct_cambio']]
    .sort_values('rank_1999')
    .rename(columns={'causa': 'Causa', 'rank_1999': 'Rank 1999',
                     'rank_2017': 'Rank 2017', 'cambio_fmt': 'Cambio', 'pct_cambio': 'Variación %'})
)

def colorear_tabla(df):
    def cc(val): return 'color: green; font-weight: bold' if '▲' in str(val) else ('color: red; font-weight: bold' if '▼' in str(val) else '')
    def cp(val):
        try: v = float(val); return f'color: {"red" if v > 0 else "green"}; {"font-weight:bold" if abs(v) > 10 else ""}'
        except: return ''
    return (df.style.map(cc, subset=['Cambio']).map(cp, subset=['Variación %'])
            .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#04376B'), ('color', 'white'), ('font-weight', 'bold')]}])
            .set_caption('Cambios en el ranking de causas de muerte (1999–2017)').hide(axis='index'))

display(colorear_tabla(tabla_cambios))


# =============================================================================
#  CELDA 7 – SECCIÓN 6  Muertes por desesperación
# =============================================================================

from scipy.ndimage import uniform_filter1d
causas_d = ['Suicide', 'Unintentional injuries']
datos_d  = datos_nacional[datos_nacional['causa'].isin(causas_d)].copy()
datos_d['anio'] = datos_d['anio'].astype(int)
colors_d = {'Suicide': '#2F4F4F', 'Unintentional injuries': '#528B8B'}

fig, ax = plt.subplots(figsize=(11, 6))
for causa, grp in datos_d.groupby('causa'):
    grp   = grp.sort_values('anio')
    ax.plot(grp['anio'], grp['tasa_ajustada'], color=colors_d[causa], linewidth=2, marker='o', markersize=5, label=causa)
    suave = uniform_filter1d(grp['tasa_ajustada'].values, size=3)
    ax.plot(grp['anio'], suave, color=colors_d[causa], linestyle='--', linewidth=1.2, alpha=0.7)
ax.axvline(2014, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(2014.1, ax.get_ylim()[0] * 1.02, '2014', color='gray', fontsize=8)
ax.set_xticks(range(1999, 2018))
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Tasa por 100,000 habitantes', fontweight='bold')
ax.set_title("Evolución de las 'Muertes por Desesperación' (1999–2017)\nSuicidio y Lesiones No Intencionales", fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 8 – SECCIÓN 7  Top 10 causas de muerte
# =============================================================================

top10 = pd.DataFrame({
    'causa': ['Heart disease', 'Cancer', 'Unintentional injuries', 'CLRD', 'Stroke',
              'Diabetes', "Alzheimer's disease", 'Kidney disease', 'Influenza and pneumonia', 'Suicide'],
    'total_muertes': [326637, 281998, 247735, 207523, 192144, 170364, 144783, 143719, 53128, 48875]
})
top10_ord   = top10.sort_values('total_muertes').copy()
norm        = plt.Normalize(top10_ord['total_muertes'].min(), top10_ord['total_muertes'].max())
colors_bar  = [plt.cm.Blues(0.35 + 0.65 * norm(v)) for v in top10_ord['total_muertes']]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top10_ord['causa'], top10_ord['total_muertes'], color=colors_bar, height=0.7)
for bar, val in zip(bars, top10_ord['total_muertes']):
    ax.text(val + 1500, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontweight='bold', fontsize=9)
ax.set_xlabel('Número de Muertes', fontweight='bold')
ax.set_title('Top 10 Causas de Muerte en EE.UU.\nTotal acumulado de muertes (1999–2017)', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
ax.set_xlim(0, top10_ord['total_muertes'].max() * 1.18)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 9 – SECCIÓN 8.1  Paneles de tasa ajustada por causa
# =============================================================================

tasas_nac = (
    datos[datos['causa'] != 'All causes']
    .groupby(['anio', 'causa'], as_index=False)['tasa_ajustada'].mean()
)
causas_u = sorted(tasas_nac['causa'].unique())
n = len(causas_u)
ncols = 2
nrows = (n + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
axes = axes.flatten()
for i, causa in enumerate(causas_u):
    ax  = axes[i]
    sub = tasas_nac[tasas_nac['causa'] == causa].sort_values('anio')
    ax.plot(sub['anio'], sub['tasa_ajustada'], color='#2171B5', linewidth=1.5, marker='o', markersize=3, alpha=0.8)
    ax.set_title(causa, fontweight='bold', fontsize=8, backgroundcolor='#F0F7FF', pad=4)
    ax.set_xticks(range(1999, 2018, 2))
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Evolución de las tasas de mortalidad ajustadas por edad\nCada panel muestra una causa (promedio nacional 1999–2017)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 10 – SECCIÓN 8.1.1  *** NUEVO *** Boxplots por semestre
# =============================================================================

def asignar_semestre(anio):
    bloque = (anio - 1999) // 2 + 1
    inicio = 1999 + (bloque - 1) * 2
    fin    = inicio + 1
    return f"S{bloque:02d} ({inicio}–{fin})"

tasas_box = tasas_nac.copy()
tasas_box['semestre'] = tasas_box['anio'].apply(asignar_semestre)
orden_sem = sorted(tasas_box['semestre'].unique(), key=lambda x: int(x.split('S')[1].split(' ')[0]))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
axes = axes.flatten()
for i, causa in enumerate(causas_u):
    ax  = axes[i]
    sub = tasas_box[tasas_box['causa'] == causa]
    sns.boxplot(data=sub, x='semestre', y='tasa_ajustada', order=orden_sem, ax=ax,
                palette='Blues', width=0.6, linewidth=1.2,
                flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=4))
    ax.set_title(causa, fontweight='bold', fontsize=9, backgroundcolor='#F0F7FF', pad=4)
    ax.set_xlabel('Semestre del período', fontsize=7)
    ax.set_ylabel('Tasa ajustada\n(por 100,000 hab.)', fontsize=7)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Distribución semestral de las tasas de mortalidad ajustadas por edad\nDiagramas de caja y bigotes – bloques de 2 años (1999–2017)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# Interpretación:
# Los diagramas de caja y bigotes permiten observar cómo varía la dispersión de las tasas
# de mortalidad para cada causa a lo largo del período. Las enfermedades cardíacas presentan
# una mediana decreciente con dispersión reducida en los semestres más recientes, mientras que
# las lesiones no intencionales muestran aumento progresivo de la mediana, coherente con la
# crisis de opioides.


# =============================================================================
#  CELDA 11 – SECCIÓN 8.2  Heatmap de muertes absolutas
# =============================================================================

pivot_heat = (
    datos[datos['causa'] != 'All causes']
    .pivot(index='causa', columns='anio', values='muertes')
)
pivot_heat = pivot_heat.loc[pivot_heat.sum(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot_heat, cmap='Blues', ax=ax, linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Número de muertes', 'shrink': 0.6})
ax.set_title('Intensidad de muertes absolutas por causa y año\nCeldas más oscuras indican mayor número de muertes (1999–2017)', fontsize=13, fontweight='bold')
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Causa', fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 12 – SECCIÓN 8.3.1  Mapa interactivo – número de muertes por estado
# =============================================================================

import plotly.express as px

abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

mapa_muertes = datos_estados[
    (datos_estados['causa'] == 'All causes') & (datos_estados['estado'].notna())
][['anio', 'estado', 'muertes']].dropna().copy()
mapa_muertes['anio']   = mapa_muertes['anio'].astype(int)
mapa_muertes['codigo'] = mapa_muertes['estado'].map(abbrev)
mapa_muertes = mapa_muertes.dropna(subset=['codigo'])

fig_map = px.choropleth(
    mapa_muertes, locations='codigo', locationmode='USA-states',
    color='muertes', animation_frame='anio', color_continuous_scale='Blues',
    scope='usa', hover_name='estado',
    hover_data={'muertes': ':,', 'codigo': False},
    labels={'muertes': 'Número de muertes', 'anio': 'Año'},
    title='Número de muertes por estado (1999–2017)',
    range_color=[mapa_muertes['muertes'].quantile(0.02), mapa_muertes['muertes'].quantile(0.98)]
)
fig_map.update_layout(title_font_size=13, height=500,
                      geo=dict(showlakes=True, lakecolor='rgb(255,255,255)'))
fig_map.show(renderer="notebook")



# =============================================================================
#  CELDA 13 – SECCIÓN 8.3.2  Mapa interactivo – tasa ajustada por estado
#  CORRECCIÓN: se exporta a HTML para que se vea correctamente
# =============================================================================

mapa_tasas = datos_estados[
    (datos_estados['causa'] == 'All causes') & (datos_estados['estado'].notna())
][['anio', 'estado', 'tasa_ajustada']].dropna().copy()
mapa_tasas['anio']   = mapa_tasas['anio'].astype(int)
mapa_tasas['codigo'] = mapa_tasas['estado'].map(abbrev)
mapa_tasas = mapa_tasas.dropna(subset=['codigo'])

fig_tasas = px.choropleth(
    mapa_tasas, locations='codigo', locationmode='USA-states',
    color='tasa_ajustada', animation_frame='anio', color_continuous_scale='Reds',
    scope='usa', hover_name='estado',
    hover_data={'tasa_ajustada': ':.1f', 'codigo': False},
    labels={'tasa_ajustada': 'Tasa ajustada (por 100,000)', 'anio': 'Año'},
    title='Tasa de mortalidad ajustada por estado (1999–2017)',
    range_color=[mapa_tasas['tasa_ajustada'].quantile(0.02),
                 mapa_tasas['tasa_ajustada'].quantile(0.98)]
)
fig_tasas.update_layout(title_font_size=13, height=500,
                        geo=dict(showlakes=True, lakecolor='rgb(255,255,255)'))
fig_tasas.show(renderer="notebook")

# Exportar a HTML para que se vea en el informe
fig_tasas.write_html("mapa_tasa_ajustada.html")
print("✓ Mapa guardado como 'mapa_tasa_ajustada.html'")



# =============================================================================
#  CELDA 14 – SECCIÓN 9  West Virginia – todas las causas
# =============================================================================

datos_wv = datos[datos['causa'] != 'All causes'].copy()
datos_wv['destacar'] = datos_wv['causa'].apply(
    lambda x: 'Unintentional injuries' if x == 'Unintentional injuries' else 'Otras causas'
)

fig, ax = plt.subplots(figsize=(12, 6), facecolor='#F4F6F8')
ax.set_facecolor('#F4F6F8')
for causa, grp in datos_wv[datos_wv['destacar'] == 'Otras causas'].groupby('causa'):
    ax.plot(grp['anio'], grp['tasa_ajustada'], color='#9AA0A6', linewidth=1, alpha=0.6)
ui = datos_wv[datos_wv['destacar'] == 'Unintentional injuries'].sort_values('anio')
ax.plot(ui['anio'], ui['tasa_ajustada'], color='#D62828', linewidth=2.5, label='Unintentional injuries')
ax.set_xticks(range(1999, 2018, 2))
ax.set_xlim(1999, 2017)
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Age-adjusted Death Rate', fontweight='bold')
ax.set_title('West Virginia', fontsize=16, fontweight='bold')
ax.text(0.5, 1.03, "Desde 2014, 'Unintentional injuries' muestra una tendencia creciente",
        transform=ax.transAxes, ha='center', fontsize=11, color='#D62828', fontweight='bold')
ax.grid(color='white', linewidth=0.8)
grey_patch = mpatches.Patch(color='#9AA0A6', label='Otras causas')
red_patch  = mpatches.Patch(color='#D62828', label='Unintentional injuries')
ax.legend(handles=[grey_patch, red_patch], loc='upper left')
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 15 – SECCIÓN 9.1  Comparación muertes 1999 vs 2017 (nacional)
# =============================================================================

comparacion = datos[
    (datos['anio'].isin([1999, 2017])) & (datos['causa'] != 'All causes')
][['anio', 'causa', 'muertes']].copy()
comparacion['anio'] = comparacion['anio'].astype(str)
orden = comparacion[comparacion['anio'] == '2017'].sort_values('muertes')['causa'].tolist()
pivot_comp = comparacion.pivot(index='causa', columns='anio', values='muertes').loc[orden]

fig, ax = plt.subplots(figsize=(11, 8))
y = np.arange(len(pivot_comp))
h = 0.35
ax.barh(y + h/2, pivot_comp['1999'], h, label='1999', color='#457B9D', alpha=0.9)
ax.barh(y - h/2, pivot_comp['2017'], h, label='2017', color='#ADD8E6', alpha=0.9)
ax.set_yticks(y)
ax.set_yticklabels(pivot_comp.index, fontsize=10)
ax.set_xlabel('Número de muertes', fontweight='bold')
ax.set_title('Comparación del número de muertes (1999 vs 2017)\nTotal Nacional', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 16 – SECCIÓN 9.2  *** CORRECCIÓN *** West Virginia vs Promedio Nacional
#  (antes decía 'Virginia' — corregido a 'West Virginia')
# =============================================================================

west_virginia = datos_estados[
    (datos_estados['estado'] == 'West Virginia') &
    (datos_estados['causa']  == 'All causes')
].sort_values('anio')

promedio_nac = (
    datos_estados[datos_estados['causa'] == 'All causes']
    .groupby('anio')['tasa_ajustada'].mean()
    .reset_index(name='tasa_promedio')
)
comp_wv = west_virginia.merge(promedio_nac, on='anio')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(comp_wv['anio'], comp_wv['tasa_ajustada'], color='darkred', linewidth=2, label='West Virginia')
ax.plot(comp_wv['anio'], comp_wv['tasa_promedio'], color='steelblue', linewidth=2, linestyle='--', label='Promedio Nacional')
ax.fill_between(comp_wv['anio'], comp_wv['tasa_ajustada'], comp_wv['tasa_promedio'], alpha=0.1, color='red')
ax.set_title('Tasa de mortalidad ajustada: West Virginia vs. Promedio Nacional', fontsize=13, fontweight='bold')
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Tasa por 100,000 hab.', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Ranking West Virginia en 2017
ranking_2017 = (
    datos_estados[(datos_estados['anio'] == 2017) & (datos_estados['causa'] == 'All causes')]
    .sort_values('tasa_ajustada', ascending=False)
    .reset_index(drop=True)
)
ranking_2017.index += 1
pos_wv = ranking_2017[ranking_2017['estado'] == 'West Virginia'].index.tolist()
print(f"Posición de West Virginia en el ranking 2017: {pos_wv}")
print(ranking_2017[['estado', 'tasa_ajustada']].head(10).to_string())


# =============================================================================
#  CELDA 17 – SECCIÓN 10  *** NUEVO *** Clustering K-Means
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster      import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics      import silhouette_score

# ── 10.1 Preparación ─────────────────────────────────────────────────────────
cluster_data = (
    datos_estados[
        (datos_estados['causa'] != 'All causes') &
        (datos_estados['estado'].notna())
    ]
    .groupby(['estado', 'causa'])['tasa_ajustada'].mean()
    .reset_index()
)
pivot_cluster = cluster_data.pivot(index='estado', columns='causa', values='tasa_ajustada').dropna()

print(f"Estados disponibles para clustering: {len(pivot_cluster)}")
print(f"Causas usadas como variables:        {pivot_cluster.shape[1]}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(pivot_cluster)


# ── 10.2 Método del codo + Silhouette ────────────────────────────────────────
inercias    = []
silhouettes = []
rango_k     = range(2, 10)

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(rango_k), inercias, 'o-', color='#2171B5', linewidth=2, markersize=6)
axes[0].set_xlabel('Número de clústeres (k)', fontweight='bold')
axes[0].set_ylabel('Inercia (Within-cluster SS)', fontweight='bold')
axes[0].set_title('Método del Codo\nSelección del número óptimo de clústeres', fontweight='bold', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

axes[1].plot(list(rango_k), silhouettes, 's-', color='#E74C3C', linewidth=2, markersize=6)
axes[1].set_xlabel('Número de clústeres (k)', fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontweight='bold')
axes[1].set_title('Silhouette Score por k\nValores más altos = mejor separación', fontweight='bold', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ── 10.3 Modelo K-Means ──────────────────────────────────────────────────────
k_optimo = 4   # Ajusta según la curva del codo

km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
km_final.fit(X_scaled)
pivot_cluster['cluster'] = km_final.labels_ + 1

print("═══════════════════════════════════════════════════════")
print("     Composición de los clústeres (estados por grupo)  ")
print("═══════════════════════════════════════════════════════")
for c in sorted(pivot_cluster['cluster'].unique()):
    estados_c = sorted(pivot_cluster[pivot_cluster['cluster'] == c].index.tolist())
    print(f"\n  Clúster {c} ({len(estados_c)} estados):")
    print("  " + ', '.join(estados_c))

perfil_clusters = (
    pivot_cluster.drop(columns=['cluster'])
    .join(pivot_cluster[['cluster']])
    .groupby('cluster').mean().round(1)
)
print("\nPerfil promedio de tasas ajustadas por clúster:")
display(
    perfil_clusters.style
    .background_gradient(cmap='Blues', axis=0)
    .set_caption('Tasa ajustada promedio por causa y clúster (1999–2017)')
    .format('{:.1f}')
)


# ── 10.4 Visualización PCA ───────────────────────────────────────────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var1  = pca.explained_variance_ratio_[0] * 100
var2  = pca.explained_variance_ratio_[1] * 100

colores = ['#2171B5', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#16A085']

fig, ax = plt.subplots(figsize=(12, 7))
for c in sorted(pivot_cluster['cluster'].unique()):
    mask = pivot_cluster['cluster'].values == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               color=colores[c - 1], s=90, alpha=0.85,
               label=f'Clúster {c}', edgecolors='white', linewidth=0.5, zorder=3)
    for idx, estado in enumerate(pivot_cluster.index):
        if pivot_cluster['cluster'].iloc[idx] == c:
            ax.annotate(estado, (X_pca[idx, 0], X_pca[idx, 1]),
                        fontsize=6, alpha=0.75, ha='center',
                        xytext=(0, 6), textcoords='offset points')

ax.set_xlabel(f'PC1 ({var1:.1f}% varianza explicada)', fontweight='bold')
ax.set_ylabel(f'PC2 ({var2:.1f}% varianza explicada)', fontweight='bold')
ax.set_title(f'Clustering K-Means (k={k_optimo}) – Visualización PCA\nEstados agrupados por perfil de mortalidad (1999–2017)', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.8)
ax.grid(alpha=0.2)
ax.text(0.5, -0.1, 'Fuente: CDC – Leading Causes of Death', transform=ax.transAxes, ha='center', fontsize=8, color='gray')
plt.tight_layout()
plt.show()


# =============================================================================
#  CELDA 18 – SECCIÓN 11  *** NUEVO *** Modelo Predictivo ARIMA
#  pip install pmdarima statsmodels
# =============================================================================

import pmdarima as pm
from statsmodels.tsa.stattools      import adfuller
from statsmodels.graphics.tsaplots  import plot_acf, plot_pacf
from statsmodels.stats.diagnostic   import acorr_ljungbox


# ── 11.1 Series de tiempo por causa ──────────────────────────────────────────
causas_modelo = [c for c in tasas_nac['causa'].unique() if c != 'All causes']
nrows_m       = (len(causas_modelo) + 1) // 2

fig, axes = plt.subplots(nrows_m, 2, figsize=(14, nrows_m * 1.8))
axes = axes.flatten()
for i, causa in enumerate(causas_modelo):
    s = (tasas_nac[tasas_nac['causa'] == causa]
         .sort_values('anio').set_index('anio')['tasa_ajustada'])
    axes[i].plot(s.index, s.values, color='#2171B5', linewidth=1.5, marker='o', markersize=3)
    axes[i].set_title(causa, fontweight='bold', fontsize=8, backgroundcolor='#F0F7FF', pad=3)
    axes[i].tick_params(labelsize=6)
    axes[i].grid(alpha=0.3)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Series de tiempo por causa – Tasa ajustada nacional (1999–2017)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# ── 11.2 Prueba ADF ───────────────────────────────────────────────────────────
res_adf = []
for causa in causas_modelo:
    s = (tasas_nac[tasas_nac['causa'] == causa]
         .sort_values('anio')['tasa_ajustada'].dropna().values)
    stat, pval, _, _, crit, _ = adfuller(s, autolag='AIC')
    res_adf.append({
        'Causa'           : causa,
        'Estadístico ADF' : round(stat, 3),
        'p-valor'         : round(pval, 4),
        'Val. crítico 5%' : round(crit['5%'], 3),
        '¿Estacionaria?'  : '✅ Sí' if pval < 0.05 else '❌ No'
    })

df_adf = pd.DataFrame(res_adf)

def col_e(v): return 'color:green;font-weight:bold' if 'Sí' in str(v) else ('color:red;font-weight:bold' if 'No' in str(v) else '')
def col_p(v):
    try: return 'color:green' if float(v) < 0.05 else 'color:red'
    except: return ''

display(
    df_adf.style
    .map(col_e, subset=['¿Estacionaria?'])
    .map(col_p, subset=['p-valor'])
    .set_caption('Prueba ADF – Estacionariedad por causa (1999–2017)')
    .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#04376B'), ('color', 'white'), ('font-weight', 'bold')]}])
    .hide(axis='index')
)


# ── 11.3 ACF y PACF – West Virginia ──────────────────────────────────────────
wv_temp = datos_estados[
    (datos_estados['estado'] == 'West Virginia') &
    (datos_estados['causa']  == 'All causes')
].sort_values('anio')[['anio', 'tasa_ajustada']].dropna()
anios_hist = wv_temp['anio'].astype(int).tolist()
serie_wv = wv_temp['tasa_ajustada'].values
serie_diff = pd.Series(serie_wv).diff().dropna().values

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(serie_diff,  lags=8, ax=axes[0], title='ACF – West Virginia (serie diferenciada d=1)')
plot_pacf(serie_diff, lags=5, ax=axes[1], method='ywm', title='PACF – West Virginia (serie diferenciada d=1)')
axes[0].set_xlabel('Rezago (años)')
axes[1].set_xlabel('Rezago (años)')
plt.suptitle('Identificación del orden ARIMA – West Virginia\nBarras fuera de la banda azul = rezagos significativos', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()


# ── 11.4 Selección automática del modelo ─────────────────────────────────────
modelo_auto = pm.auto_arima(
    serie_wv,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,
    seasonal=False,
    information_criterion='aic',
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print("═══════════════════════════════════════════════════")
print(f"  Mejor modelo seleccionado: ARIMA{modelo_auto.order}")
print(f"  AIC = {modelo_auto.aic():.2f}")
print("═══════════════════════════════════════════════════")
print(modelo_auto.summary())


# ── 11.5 Diagnóstico de residuos ─────────────────────────────────────────────
residuos = modelo_auto.resid()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(residuos, color='#2171B5', linewidth=1.5, marker='o', markersize=4)
axes[0].axhline(0, color='red', linestyle='--', alpha=0.6)
axes[0].set_title('Residuos en el tiempo', fontweight='bold')
axes[0].set_xlabel('Índice temporal')
axes[0].set_ylabel('Residuo')
axes[0].grid(alpha=0.3)

axes[1].hist(residuos, bins=8, color='#2171B5', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribución de residuos', fontweight='bold')
axes[1].set_xlabel('Residuo')
axes[1].set_ylabel('Frecuencia')
axes[1].grid(alpha=0.3)

plot_acf(residuos, lags=6, ax=axes[2], title='ACF de residuos (debe ser ruido blanco)')
axes[2].set_xlabel('Rezago')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.suptitle('Diagnóstico de residuos del modelo ARIMA – West Virginia', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

lb = acorr_ljungbox(residuos, lags=[5], return_df=True)
print("\nPrueba Ljung-Box (H0: residuos son ruido blanco):")
print(lb)
print("→ Si p-valor > 0.05: residuos son ruido blanco ✅")


# ── 11.6 Proyecciones ARIMA 2018–2022 ────────────────────────────────────────
n_periodos = 5
# anios_hist ya definido arriba con los años reales de West Virginia
anios_pred = list(range(2018, 2023))

forecast, conf_int = modelo_auto.predict(n_periods=n_periodos, return_conf_int=True, alpha=0.05)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(anios_hist, serie_wv, color='#2C3E50', linewidth=2, marker='o', markersize=5, label='Real (1999–2017)')
ax.plot(anios_pred, forecast, color='#E74C3C', linewidth=2, marker='s', markersize=6, linestyle='--', label='Proyección (2018–2022)')
ax.fill_between(anios_pred, conf_int[:, 0], conf_int[:, 1], alpha=0.15, color='#E74C3C', label='IC 95%')
ax.axvline(2017.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.text(2017.6, ax.get_ylim()[0] * 1.01, 'Proyección →', color='gray', fontsize=8)
ax.set_title(f'Proyecciones ARIMA{modelo_auto.order} – West Virginia\nTasa ajustada de mortalidad (por 100,000 hab.) 2018–2022', fontsize=13, fontweight='bold')
ax.set_xlabel('Año', fontweight='bold')
ax.set_ylabel('Tasa por 100,000 hab.', fontweight='bold')
ax.legend(loc='best')
ax.grid(axis='y', alpha=0.3)
ax.text(0.5, -0.12, 'Fuente: CDC – Leading Causes of Death', transform=ax.transAxes, ha='center', fontsize=8, color='gray')
plt.tight_layout()
plt.show()


# ── 11.7 Tabla de valores proyectados ────────────────────────────────────────
df_pred = pd.DataFrame({
    'Año'                 : anios_pred,
    'Proyección'          : forecast.round(2),
    'Límite inferior 95%' : conf_int[:, 0].round(2),
    'Límite superior 95%' : conf_int[:, 1].round(2)
})
display(
    df_pred.style
    .set_caption(f'Tabla de proyecciones ARIMA{modelo_auto.order} – West Virginia (2018–2022)')
    .set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#04376B'), ('color', 'white'), ('font-weight', 'bold')]}])
    .hide(axis='index')
    .format('{:.2f}', subset=['Proyección', 'Límite inferior 95%', 'Límite superior 95%'])
)

print("\n✓ Código completo ejecutado correctamente.")
print("  Recuerda exportar el notebook como HTML para que los mapas Plotly se vean.")
