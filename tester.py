import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Configurações
IMG_SIZE = (224, 224)
pasta_saudaveis = 'CheXpert-v1.0/valid_normais_frontal'
pasta_doentes = 'CheXpert-v1.0/valid_Doentes_frontal'
pasta_desconhecidos = 'CheXpert-v1.0/valid_desconhecidos_frontal'

def carregar_imagens(pasta):
    imagens = []
    nomes = []
    for nome_arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, nome_arquivo)
        img = cv2.imread(caminho)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            imagens.append(img)
            nomes.append(nome_arquivo)
    return imagens, nomes

# Carrega conjuntos de referência
imgs_saudaveis, _ = carregar_imagens(pasta_saudaveis)
imgs_doentes, _ = carregar_imagens(pasta_doentes)

# Carrega desconhecidos
imgs_desconhecidos, nomes_desconhecidos = carregar_imagens(pasta_desconhecidos)

resultados = []

for idx, img_desconhecida in enumerate(imgs_desconhecidos):
    # SSIM com saudáveis
    ssim_saudaveis = [ssim(img_desconhecida, ref, data_range=ref.max() - ref.min()) for ref in imgs_saudaveis]
    media_saudaveis = np.mean(ssim_saudaveis)
    # SSIM com doentes
    ssim_doentes = [ssim(img_desconhecida, ref, data_range=ref.max() - ref.min()) for ref in imgs_doentes]
    media_doentes = np.mean(ssim_doentes)
    # Classificação
    if media_saudaveis > media_doentes:
        classe = 'Saudável'
    else:
        classe = 'Doente'
    resultados.append({
        'imagem': nomes_desconhecidos[idx],
        'ssim_medio_saudaveis': media_saudaveis,
        'ssim_medio_doentes': media_doentes,
        'classificacao': classe
    })
    print(f"{nomes_desconhecidos[idx]} | SSIM saudável: {media_saudaveis:.4f} | SSIM doente: {media_doentes:.4f} | {classe}")

# Salva relatório em CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('relatorio_classificacao.csv', index=False)