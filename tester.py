#!/usr/bin/env python3
"""
Sistema robusto de classificação de raios-X usando segmentação pulmonar e SSIM.
Processa imagens em batch, com logging detalhado e tratamento de erros.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TensorFlow/Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz warnings do TF
import tensorflow as tf
from tensorflow import keras

# ========== CONFIGURAÇÃO ==========
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "model.h5"
OUTPUT_DIR = BASE_DIR / "output"
SEG_SIZE = (256, 256)
SSIM_SIZE = (224, 224)
BATCH_SIZE = 16
MASK_THRESHOLD = 0.5
SAVE_SEGMENTED = True  # Salvar imagens segmentadas

# Configurar GPU (se disponível)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"GPU disponível: {len(gpus)} dispositivo(s)")
    except RuntimeError as e:
        logging.warning(f"Erro ao configurar GPU: {e}")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'tester.log' if OUTPUT_DIR.exists() else 'tester.log')
    ]
)

# Validar e carregar modelo
if not MODEL_PATH.exists():
    logging.error(f"❌ Modelo não encontrado: {MODEL_PATH}")
    sys.exit(1)

try:
    model_segment = keras.models.load_model(str(MODEL_PATH), compile=False)
    logging.info(f"✅ Modelo carregado: {MODEL_PATH}")
except Exception as e:
    logging.exception(f"❌ Falha ao carregar modelo: {e}")
    sys.exit(1)


def segmentar_pulmao_batch(imagens_gray: List[np.ndarray]) -> List[np.ndarray]:
    """
    Segmenta múltiplas imagens em batch para melhor performance.
    
    Args:
        imagens_gray: Lista de imagens em escala de cinza
        
    Returns:
        Lista de imagens segmentadas
    """
    if not imagens_gray:
        return []
    
    # Prepara batch
    batch = []
    for img in imagens_gray:
        img_resized = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        batch.append(img_norm)
    
    batch_array = np.expand_dims(np.array(batch), axis=-1)  # (N, 256, 256, 1)
    
    # Predição em batch
    try:
        masks = model_segment.predict(batch_array, batch_size=BATCH_SIZE, verbose=0)
    except Exception as e:
        logging.error(f"Erro na predição batch: {e}")
        return [np.zeros_like(img) for img in imagens_gray]
    
    # Aplica máscaras
    segmentadas = []
    for i, (img, mask) in enumerate(zip(batch, masks)):
        mask_bin = (mask[:, :, 0] > MASK_THRESHOLD).astype(np.uint8)
        img_seg = (img * 255.0 * mask_bin).astype(np.uint8)
        segmentadas.append(img_seg)
    
    return segmentadas


def segmentar_pulmao(imagem_gray: np.ndarray) -> np.ndarray:
    """
    Segmenta os pulmões usando o modelo U-Net (versão single image).
    
    Args:
        imagem_gray: imagem numpy (escala de cinza)
        
    Returns:
        imagem com apenas os pulmões (ou seja, com máscara aplicada)
    """
    return segmentar_pulmao_batch([imagem_gray])[0]


def normalizar_para_ssim(img: np.ndarray, target_size: Tuple[int, int] = SSIM_SIZE) -> np.ndarray:
    """
    Normaliza imagem para cálculo de SSIM consistente.
    
    Args:
        img: Imagem de entrada
        target_size: Tamanho alvo
        
    Returns:
        Imagem normalizada em [0, 1]
    """
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_float = img_resized.astype(np.float32)
    
    if img_float.max() > 1.0:
        img_float = img_float / 255.0
    
    return np.clip(img_float, 0.0, 1.0)


def calcular_ssim_robusto(img1: np.ndarray, img2: np.ndarray) -> Optional[float]:
    """
    Calcula SSIM com tratamento de erros.
    
    Args:
        img1, img2: Imagens para comparar
        
    Returns:
        Valor SSIM ou None em caso de erro
    """
    try:
        img1_norm = normalizar_para_ssim(img1)
        img2_norm = normalizar_para_ssim(img2)
        
        valor_ssim = ssim(img1_norm, img2_norm, data_range=1.0)
        
        if not np.isfinite(valor_ssim):
            logging.warning("SSIM não finito detectado")
            return None
            
        return float(valor_ssim)
    except Exception as e:
        logging.debug(f"Erro ao calcular SSIM: {e}")
        return None


def carregar_imagens(pasta: Path, usar_batch: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """
    Carrega e segmenta imagens de uma pasta com processamento otimizado.
    
    Args:
        pasta: Path da pasta com imagens
        usar_batch: Se True, usa segmentação em batch
        
    Returns:
        Tuple (lista de imagens segmentadas, lista de nomes)
    """
    if not pasta.exists():
        logging.error(f"❌ Pasta não encontrada: {pasta}")
        return [], []
    
    # Lista arquivos de imagem
    extensoes = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    arquivos = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]
    
    if not arquivos:
        logging.warning(f"⚠️ Nenhuma imagem encontrada em {pasta}")
        return [], []
    
    logging.info(f"📂 Carregando {len(arquivos)} imagens de {pasta.name}")
    
    # Carrega imagens em escala de cinza
    imagens_gray = []
    nomes = []
    
    for arquivo in tqdm(arquivos, desc=f"Lendo {pasta.name}", unit="img"):
        try:
            img = cv2.imread(str(arquivo), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logging.warning(f"⚠️ Falha ao ler: {arquivo.name}")
                continue
            
            imagens_gray.append(img)
            nomes.append(arquivo.name)
        except Exception as e:
            logging.error(f"❌ Erro ao processar {arquivo.name}: {e}")
    
    if not imagens_gray:
        return [], []
    
    # Segmenta em batch ou individualmente
    logging.info(f"🔬 Segmentando {len(imagens_gray)} imagens...")
    
    if usar_batch:
        segmentadas = []
        for i in tqdm(range(0, len(imagens_gray), BATCH_SIZE), desc="Segmentando", unit="batch"):
            batch = imagens_gray[i:i+BATCH_SIZE]
            seg_batch = segmentar_pulmao_batch(batch)
            segmentadas.extend(seg_batch)
    else:
        segmentadas = [segmentar_pulmao(img) for img in tqdm(imagens_gray, desc="Segmentando")]
    
    # Salva imagens segmentadas (opcional)
    if SAVE_SEGMENTED:
        seg_dir = OUTPUT_DIR / f"segmented_{pasta.name}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        for nome, img_seg in zip(nomes, segmentadas):
            cv2.imwrite(str(seg_dir / nome), img_seg)
    
    return segmentadas, nomes


# Configurações de pastas
pasta_saudaveis = BASE_DIR / 'valid_normais_frontal'
pasta_doentes = BASE_DIR / 'valid_Doentes_frontal'
pasta_desconhecidos = BASE_DIR / 'valid_desconhecidos_frontal'


def classificar_imagens(
    imgs_desconhecidos: List[np.ndarray],
    nomes_desconhecidos: List[str],
    imgs_saudaveis: List[np.ndarray],
    imgs_doentes: List[np.ndarray]
) -> List[Dict]:
    """
    Classifica imagens desconhecidas usando SSIM médio.
    
    Args:
        imgs_desconhecidos: Imagens a classificar
        nomes_desconhecidos: Nomes das imagens
        imgs_saudaveis: Imagens de referência saudáveis
        imgs_doentes: Imagens de referência doentes
        
    Returns:
        Lista de dicionários com resultados
    """
    resultados = []
    
    if not imgs_saudaveis and not imgs_doentes:
        logging.error("❌ Nenhuma imagem de referência disponível")
        return resultados
    
    logging.info(f"🔍 Classificando {len(imgs_desconhecidos)} imagens...")
    
    for idx, img_desconhecida in enumerate(tqdm(imgs_desconhecidos, desc="Classificando", unit="img")):
        nome = nomes_desconhecidos[idx] if idx < len(nomes_desconhecidos) else f"img_{idx}"
        
        try:
            # Calcula SSIM com saudáveis
            ssim_saudaveis = []
            if imgs_saudaveis:
                for ref in imgs_saudaveis:
                    val = calcular_ssim_robusto(img_desconhecida, ref)
                    if val is not None:
                        ssim_saudaveis.append(val)
            
            mean_saudavel = np.nanmean(ssim_saudaveis) if ssim_saudaveis else float('nan')
            
            # Calcula SSIM com doentes
            ssim_doentes = []
            if imgs_doentes:
                for ref in imgs_doentes:
                    val = calcular_ssim_robusto(img_desconhecida, ref)
                    if val is not None:
                        ssim_doentes.append(val)
            
            mean_doente = np.nanmean(ssim_doentes) if ssim_doentes else float('nan')
            
            # Classifica
            if np.isnan(mean_saudavel) and np.isnan(mean_doente):
                classe = "Indefinido"
                confianca = 0.0
            elif np.isnan(mean_doente):
                classe = "Saudável"
                confianca = mean_saudavel
            elif np.isnan(mean_saudavel):
                classe = "Doente"
                confianca = mean_doente
            else:
                if mean_saudavel > mean_doente:
                    classe = "Saudável"
                    confianca = mean_saudavel - mean_doente
                else:
                    classe = "Doente"
                    confianca = mean_doente - mean_saudavel
            
            resultados.append({
                'imagem': nome,
                'ssim_medio_saudaveis': mean_saudavel,
                'ssim_medio_doentes': mean_doente,
                'classificacao': classe,
                'confianca': confianca,
                'n_comparacoes_saudavel': len(ssim_saudaveis),
                'n_comparacoes_doente': len(ssim_doentes)
            })
            
            logging.info(
                f"{nome:30s} | SSIM saudável: {mean_saudavel:.4f} | "
                f"SSIM doente: {mean_doente:.4f} | {classe}"
            )
            
        except Exception as e:
            logging.exception(f"❌ Erro classificando {nome}: {e}")
            resultados.append({
                'imagem': nome,
                'ssim_medio_saudaveis': None,
                'ssim_medio_doentes': None,
                'classificacao': 'Erro',
                'confianca': 0.0,
                'n_comparacoes_saudavel': 0,
                'n_comparacoes_doente': 0
            })
    
    return resultados


def gerar_estatisticas(resultados: List[Dict]) -> Dict:
    """Gera estatísticas do processamento."""
    df = pd.DataFrame(resultados)
    
    stats = {
        'total_processadas': len(df),
        'total_saudaveis': len(df[df['classificacao'] == 'Saudável']),
        'total_doentes': len(df[df['classificacao'] == 'Doente']),
        'total_indefinidos': len(df[df['classificacao'] == 'Indefinido']),
        'total_erros': len(df[df['classificacao'] == 'Erro']),
    }
    
    if len(df) > 0:
        stats['confianca_media'] = df['confianca'].mean()
        stats['ssim_saudavel_medio'] = df['ssim_medio_saudaveis'].mean()
        stats['ssim_doente_medio'] = df['ssim_medio_doentes'].mean()
    
    return stats


def main():
    """Função principal."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info("🏥 Sistema de Classificação de Raios-X - Iniciando")
    logging.info("=" * 80)
    
    # Carrega conjuntos de referência
    imgs_saudaveis, _ = carregar_imagens(pasta_saudaveis)
    imgs_doentes, _ = carregar_imagens(pasta_doentes)
    
    if not imgs_saudaveis and not imgs_doentes:
        logging.error("❌ Nenhuma imagem de referência carregada. Abortando.")
        return
    
    # Carrega desconhecidos
    imgs_desconhecidos, nomes_desconhecidos = carregar_imagens(pasta_desconhecidos)
    
    if not imgs_desconhecidos:
        logging.error("❌ Nenhuma imagem desconhecida para classificar. Abortando.")
        return
    
    # Classifica
    resultados = classificar_imagens(
        imgs_desconhecidos,
        nomes_desconhecidos,
        imgs_saudaveis,
        imgs_doentes
    )
    
    # Salva relatório
    df_resultados = pd.DataFrame(resultados)
    csv_path = OUTPUT_DIR / 'relatorio_classificacao.csv'
    df_resultados.to_csv(csv_path, index=False)
    logging.info(f"💾 Relatório salvo: {csv_path}")
    
    # Gera e exibe estatísticas
    stats = gerar_estatisticas(resultados)
    logging.info("\n" + "=" * 80)
    logging.info("📊 ESTATÍSTICAS FINAIS")
    logging.info("=" * 80)
    for key, value in stats.items():
        logging.info(f"{key:30s}: {value}")
    logging.info("=" * 80)
    logging.info("✅ Processamento concluído com sucesso!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("\n⚠️ Processamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"❌ Erro fatal: {e}")
        sys.exit(1)
