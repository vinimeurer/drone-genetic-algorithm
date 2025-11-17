import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "rota.csv"

# lê o CSV
df = pd.read_csv(csv_path)

# garante nomes corretos e tipos
df = df.rename(columns=lambda s: s.strip())
df[['Latitude_inicial','Longitude_inicial','Latitude_final','Longitude_final']] = df[['Latitude_inicial','Longitude_inicial','Latitude_final','Longitude_final']].astype(float)

plt.figure(figsize=(9,7))
ax = plt.gca()

# plota segmentos inicial -> final (linhas azul claras com pontos)
for _, row in df.iterrows():
    ax.plot([row['Longitude_inicial'], row['Longitude_final']],
            [row['Latitude_inicial'], row['Latitude_final']],
            color='#4A90E2', linewidth=0.8, marker='o', markersize=4, alpha=0.9)

# plota todos os pontos iniciais como pontos azuis menores (opcional)
ax.scatter(df['Longitude_inicial'], df['Latitude_inicial'], s=18, color='#2B7CD3', alpha=0.8, label='Início')

# destaca pousos (Pouso == SIM) em vermelho
mask_pouso = df['Pouso'].astype(str).str.upper().str.strip() == 'SIM'
if mask_pouso.any():
    ax.scatter(df.loc[mask_pouso, 'Longitude_final'], df.loc[mask_pouso, 'Latitude_final'],
               s=80, color='red', marker='o', edgecolor='k', label='Pouso (SIM)')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Trajetos (inicial -> final)')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('map_voos.png', dpi=150)
plt.show()