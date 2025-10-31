import pandas as pd
import os
import shutil

# ========== CONFIGURAÇÃO ==========
# Defina quantas imagens você quer copiar (None = todas, ou um número como 100, 500, etc.)
NUM_IMAGENS = 200  # Altere aqui: None para todas, ou um número específico (ex: 100)

# Modo de seleção quando NUM_IMAGENS é definido:
# 'random' = seleção aleatória
# 'first' = pega as primeiras N do CSV filtrado
MODO_SELECAO = 'random'  # ou 'first'

# Semente para reproduzir seleção aleatória (opcional, None = aleatório a cada execução)
RANDOM_SEED = 42  # ou None
# ==================================

# Caminhos reais
read_path = '/home/lapis/Documents/IC/fase_4/Test/chexpert-ssim-classificacao-main/CheXpert-v1.0 batch 2 (train 1)'
csv_path = '/home/lapis/Documents/IC/fase_4/Test/chexpert-ssim-classificacao-main/train.csv'
dest_path = '/home/lapis/Documents/IC/fase_4/Test/chexpert-ssim-classificacao-main/valid_desconhecidos_frontal'

# Cria a pasta de destino se não existir
os.makedirs(dest_path, exist_ok=True)

# Lê o CSV
df = pd.read_csv(csv_path)

# Filtra apenas imagens Frontal com "No Finding" desconhecido (NaN ou -1)
df_filtrado = df[
    (df['Frontal/Lateral'] == 'Frontal') &
    (df['No Finding'].isnull() | (df['No Finding'] == -1) | (df['No Finding'] == ''))
].copy()

# Corrige o caminho para remover prefixo incorreto
df_filtrado['Path'] = df_filtrado['Path'].str.replace(
    r'^CheXpert-v1\.0/train/', '', regex=True
)

total_encontrado = len(df_filtrado)
print(f"Total de imagens frontais com 'No Finding' desconhecido/incerto: {total_encontrado}")

# Aplica limitação de quantidade se configurado
if NUM_IMAGENS is not None and NUM_IMAGENS > 0 and NUM_IMAGENS < total_encontrado:
    if MODO_SELECAO == 'random':
        df_filtrado = df_filtrado.sample(n=NUM_IMAGENS, random_state=RANDOM_SEED)
        print(f"Selecionando {NUM_IMAGENS} imagens aleatoriamente (seed={RANDOM_SEED})")
    elif MODO_SELECAO == 'first':
        df_filtrado = df_filtrado.head(NUM_IMAGENS)
        print(f"Selecionando as primeiras {NUM_IMAGENS} imagens")
    else:
        print(f"MODO_SELECAO '{MODO_SELECAO}' inválido, usando todas as imagens")
else:
    print(f"Copiando todas as {total_encontrado} imagens encontradas")

# Copia as imagens filtradas
copied = 0
missing = 0
for _, row in df_filtrado.iterrows():
    relative_path = row['Path']
    src = os.path.join(read_path, relative_path)  # caminho absoluto correto

    # Extrai nome do paciente
    parts = relative_path.split('/')
    paciente = next((p for p in parts if p.startswith('patient')), None)

    if paciente is None:
        print(f"Paciente não encontrado no caminho: {relative_path}")
        continue

    filename = os.path.basename(relative_path)
    new_filename = f"{paciente}_{filename}"
    dst = os.path.join(dest_path, new_filename)

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f"Arquivo não encontrado: {src}")
        missing += 1

print(f"\n✅ Cópia concluída: {copied} copiadas, {missing} não encontradas (de {len(df_filtrado)} selecionadas)")
