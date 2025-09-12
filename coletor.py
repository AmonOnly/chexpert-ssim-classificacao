import pandas as pd
import os
import shutil

read_path = 'CheXpert-v1.0/valid'
csv_path = 'CheXpert-v1.0/valid.csv'
dest_path = 'CheXpert-v1.0/valid_Doentes_frontal'

# Cria a pasta de destino se não existir
os.makedirs(dest_path, exist_ok=True)

# Lê o CSV
df = pd.read_csv(csv_path)

# Filtra apenas imagens Frontal e caminho começando com patient64
df_filtrado = df[
    (df['Frontal/Lateral'] == 'Frontal') &
    (df['Path'].str.contains('CheXpert-v1.0/valid/patient64')) &
    (df['No Finding'] == 0)
]

print(f"Total de imagens frontais: {len(df_filtrado)}")

# Copia as imagens filtradas, adicionando o nome do paciente ao nome do arquivo
for _, row in df_filtrado.iterrows():
    src = row['Path']
    # Extrai o nome do paciente do caminho (vem antes de 'study1')
    parts = src.split('/')
    paciente = parts[2]  # patient64xxx
    filename = os.path.basename(src)
    new_filename = f"{paciente}_{filename}"
    dst = os.path.join(dest_path, new_filename)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Arquivo não encontrado: {src}")