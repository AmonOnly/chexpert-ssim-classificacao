#!/usr/bin/env python3
"""
Script para gerar gráficos a partir do relatório CSV.
Gera figuras em formato PDF para inclusão no relatório LaTeX.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurações
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Caminhos
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / 'relatorio_classificacao.csv'
OUTPUT_DIR = BASE_DIR / 'figuras'
OUTPUT_DIR.mkdir(exist_ok=True)

# Lê dados
df = pd.read_csv(CSV_PATH)

print(f"Processando {len(df)} registros...")

# Calcula diferença (confiança)
df['diferenca'] = abs(df['ssim_medio_saudaveis'] - df['ssim_medio_doentes'])

# ========== FIGURA 1: Distribuição de Classificações ==========
fig, ax = plt.subplots(figsize=(8, 5))

classes = df['classificacao'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
wedges, texts, autotexts = ax.pie(
    classes.values,
    labels=classes.index,
    autopct='%1.1f%%',
    colors=colors[:len(classes)],
    startangle=90,
    textprops={'fontsize': 12}
)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

ax.set_title('Distribuição de Classificações', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distribuicao_classes.pdf', bbox_inches='tight')
print("✓ Gráfico 1: distribuicao_classes.pdf")
plt.close()

# ========== FIGURA 2: Histograma de SSIM ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# SSIM Saudável
ax1.hist(df['ssim_medio_saudaveis'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
ax1.axvline(df['ssim_medio_saudaveis'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["ssim_medio_saudaveis"].mean():.4f}')
ax1.set_xlabel('SSIM com Referências Saudáveis')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição SSIM (Saudável)')
ax1.legend()
ax1.grid(alpha=0.3)

# SSIM Doente
ax2.hist(df['ssim_medio_doentes'], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.axvline(df['ssim_medio_doentes'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Média: {df["ssim_medio_doentes"].mean():.4f}')
ax2.set_xlabel('SSIM com Referências Doentes')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição SSIM (Doente)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'histograma_ssim.pdf', bbox_inches='tight')
print("✓ Gráfico 2: histograma_ssim.pdf")
plt.close()

# ========== FIGURA 3: Scatter Plot SSIM ==========
fig, ax = plt.subplots(figsize=(8, 8))

colors_map = {'Saudável': '#2ecc71', 'Doente': '#e74c3c', 'Indefinido': '#95a5a6'}
for classe in df['classificacao'].unique():
    subset = df[df['classificacao'] == classe]
    ax.scatter(
        subset['ssim_medio_saudaveis'],
        subset['ssim_medio_doentes'],
        c=colors_map.get(classe, '#000000'),
        label=f'{classe} (n={len(subset)})',
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )

# Linha de decisão (y = x)
lims = [
    min(df['ssim_medio_saudaveis'].min(), df['ssim_medio_doentes'].min()) - 0.01,
    max(df['ssim_medio_saudaveis'].max(), df['ssim_medio_doentes'].max()) + 0.01
]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Linha de Decisão (y=x)')

ax.set_xlabel('SSIM Médio com Referências Saudáveis', fontsize=12)
ax.set_ylabel('SSIM Médio com Referências Doentes', fontsize=12)
ax.set_title('Espaço de Classificação SSIM', fontsize=14, weight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'scatter_ssim.pdf', bbox_inches='tight')
print("✓ Gráfico 3: scatter_ssim.pdf")
plt.close()

# ========== FIGURA 4: Boxplot Comparativo ==========
fig, ax = plt.subplots(figsize=(10, 6))

data_to_plot = [
    df['ssim_medio_saudaveis'],
    df['ssim_medio_doentes'],
    df['diferenca']
]

bp = ax.boxplot(
    data_to_plot,
    labels=['SSIM (Saudável)', 'SSIM (Doente)', 'Diferença (Confiança)'],
    patch_artist=True,
    notch=True,
    showmeans=True
)

colors = ['#3498db', '#e74c3c', '#f39c12']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Distribuição de Métricas SSIM', fontsize=14, weight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplot_metricas.pdf', bbox_inches='tight')
print("✓ Gráfico 4: boxplot_metricas.pdf")
plt.close()

# ========== FIGURA 5: Distribuição de Confiança ==========
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df['diferenca'], bins=25, color='#9b59b6', alpha=0.7, edgecolor='black')
ax.axvline(df['diferenca'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["diferenca"].mean():.4f}')
ax.axvline(df['diferenca'].median(), color='green', linestyle='-.', linewidth=2, label=f'Mediana: {df["diferenca"].median():.4f}')

ax.set_xlabel('Diferença Absoluta SSIM (Confiança)', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
ax.set_title('Distribuição da Confiança na Classificação', fontsize=14, weight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confianca_distribuicao.pdf', bbox_inches='tight')
print("✓ Gráfico 5: confianca_distribuicao.pdf")
plt.close()

# ========== FIGURA 6: Top 10 Maiores Confiança ==========
fig, ax = plt.subplots(figsize=(10, 6))

top10 = df.nlargest(10, 'diferenca')[['imagem', 'diferenca', 'classificacao']].copy()
top10['imagem_short'] = top10['imagem'].str.replace('_view1_frontal.jpg', '').str.replace('patient', 'P')

colors_bar = [colors_map.get(c, '#000000') for c in top10['classificacao']]

bars = ax.barh(range(len(top10)), top10['diferenca'], color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['imagem_short'], fontsize=9)
ax.set_xlabel('Confiança (Diferença SSIM)', fontsize=12)
ax.set_title('Top 10 Classificações com Maior Confiança', fontsize=14, weight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top10_confianca.pdf', bbox_inches='tight')
print("✓ Gráfico 6: top10_confianca.pdf")
plt.close()

# ========== ESTATÍSTICAS RESUMIDAS ==========
print("\n" + "="*60)
print("ESTATÍSTICAS RESUMIDAS")
print("="*60)
print(f"Total de imagens: {len(df)}")
print(f"\nDistribuição de classes:")
for classe, count in df['classificacao'].value_counts().items():
    pct = 100 * count / len(df)
    print(f"  {classe:12s}: {count:3d} ({pct:5.2f}%)")

print(f"\nSSIM Saudável:")
print(f"  Média: {df['ssim_medio_saudaveis'].mean():.4f} ± {df['ssim_medio_saudaveis'].std():.4f}")
print(f"  Min/Max: {df['ssim_medio_saudaveis'].min():.4f} / {df['ssim_medio_saudaveis'].max():.4f}")

print(f"\nSSIM Doente:")
print(f"  Média: {df['ssim_medio_doentes'].mean():.4f} ± {df['ssim_medio_doentes'].std():.4f}")
print(f"  Min/Max: {df['ssim_medio_doentes'].min():.4f} / {df['ssim_medio_doentes'].max():.4f}")

print(f"\nConfiança (Diferença):")
print(f"  Média: {df['diferenca'].mean():.4f}")
print(f"  Mediana: {df['diferenca'].median():.4f}")
print(f"  Desvio Padrão: {df['diferenca'].std():.4f}")

print("\n" + "="*60)
print(f"✅ Todos os gráficos salvos em: {OUTPUT_DIR}")
print("="*60)
