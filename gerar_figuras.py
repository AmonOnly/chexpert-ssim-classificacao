#!/usr/bin/env python3
"""
Script para gerar figuras de exemplo para o relatório.
Cria imagens mostrando:
- Exemplo de segmentação (original vs segmentada)
- Exemplo de comparação (teste vs referências)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Diretórios
FIGURAS_DIR = 'figuras'
SEGMENTADOS_DIR = 'segmentados'
NORMAL_DIR = 'valid_normais_frontal'
DOENTE_DIR = 'valid_Doentes_frontal'
TESTE_DIR = 'valid_desconhecidos_frontal'

def criar_diretorio_figuras():
    """Cria o diretório de figuras se não existir."""
    os.makedirs(FIGURAS_DIR, exist_ok=True)
    logging.info(f"Diretório '{FIGURAS_DIR}' criado/verificado")

def encontrar_imagem_exemplo(diretorio, extensoes=['.jpg', '.png', '.jpeg']):
    """Encontra a primeira imagem válida em um diretório."""
    try:
        diretorio = Path(diretorio)
        if not diretorio.exists():
            logging.warning(f"Diretório não encontrado: {diretorio}")
            return None
        
        for ext in extensoes:
            imagens = list(diretorio.glob(f'*{ext}'))
            if imagens:
                return str(imagens[0])
        
        logging.warning(f"Nenhuma imagem encontrada em {diretorio}")
        return None
    except Exception as e:
        logging.error(f"Erro ao procurar imagem em {diretorio}: {e}")
        return None

def carregar_e_redimensionar(caminho, tamanho=(512, 512)):
    """Carrega e redimensiona uma imagem."""
    try:
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Erro ao carregar: {caminho}")
            return None
        
        img = cv2.resize(img, tamanho, interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        logging.error(f"Erro ao processar {caminho}: {e}")
        return None

def adicionar_bordas(img, cor=255, espessura=2):
    """Adiciona bordas brancas à imagem."""
    if img is None:
        return None
    bordered = cv2.copyMakeBorder(img, espessura, espessura, espessura, espessura,
                                   cv2.BORDER_CONSTANT, value=cor)
    return bordered

def gerar_exemplo_segmentacao():
    """Gera figura mostrando exemplo de segmentação."""
    logging.info("Gerando exemplo de segmentação...")
    
    # Buscar imagem original de teste
    imagem_teste = encontrar_imagem_exemplo(TESTE_DIR)
    if not imagem_teste:
        logging.error("Não foi possível encontrar imagem de teste")
        return False
    
    # Buscar imagem segmentada correspondente
    nome_base = Path(imagem_teste).stem
    imagem_segmentada = None
    
    # Procurar em segmentados/ diretamente
    segmentados_path = Path(SEGMENTADOS_DIR)
    if segmentados_path.exists():
        # Tentar encontrar correspondente
        possivel = segmentados_path / f"pulmao_segmentado_{nome_base}.png"
        if possivel.exists():
            imagem_segmentada = str(possivel)
        else:
            # Usar primeira imagem disponível
            imgs = list(segmentados_path.glob('pulmao_segmentado_*.png'))
            if imgs:
                imagem_segmentada = str(imgs[0])
    
    if not imagem_segmentada:
        logging.error("Nenhuma imagem segmentada encontrada")
        return False
    
    # Carregar imagens
    img_original = carregar_e_redimensionar(imagem_teste)
    img_segmentada = carregar_e_redimensionar(imagem_segmentada)
    
    if img_original is None or img_segmentada is None:
        logging.error("Erro ao carregar imagens para exemplo de segmentação")
        return False
    
    # Adicionar bordas
    img_original = adicionar_bordas(img_original)
    img_segmentada = adicionar_bordas(img_segmentada)
    
    # Salvar
    cv2.imwrite(os.path.join(FIGURAS_DIR, 'exemplo_original_1.png'), img_original)
    cv2.imwrite(os.path.join(FIGURAS_DIR, 'exemplo_segmentado_1.png'), img_segmentada)
    
    logging.info(f"✓ Exemplo de segmentação gerado: {imagem_teste} -> {imagem_segmentada}")
    return True

def gerar_exemplo_comparacao():
    """Gera figura mostrando exemplo de comparação SSIM."""
    logging.info("Gerando exemplo de comparação...")
    
    # Buscar imagens
    img_teste = encontrar_imagem_exemplo(TESTE_DIR)
    img_normal = encontrar_imagem_exemplo(NORMAL_DIR)
    img_doente = encontrar_imagem_exemplo(DOENTE_DIR)
    
    if not all([img_teste, img_normal, img_doente]):
        logging.error("Não foi possível encontrar todas as imagens para comparação")
        return False
    
    # Carregar e processar
    teste = carregar_e_redimensionar(img_teste)
    normal = carregar_e_redimensionar(img_normal)
    doente = carregar_e_redimensionar(img_doente)
    
    if teste is None or normal is None or doente is None:
        logging.error("Erro ao carregar imagens para comparação")
        return False
    
    # Adicionar bordas
    teste = adicionar_bordas(teste)
    normal = adicionar_bordas(normal)
    doente = adicionar_bordas(doente)
    
    # Salvar
    cv2.imwrite(os.path.join(FIGURAS_DIR, 'teste_exemplo.png'), teste)
    cv2.imwrite(os.path.join(FIGURAS_DIR, 'ref_normal.png'), normal)
    cv2.imwrite(os.path.join(FIGURAS_DIR, 'ref_doente.png'), doente)
    
    logging.info(f"✓ Exemplo de comparação gerado")
    logging.info(f"  Teste: {img_teste}")
    logging.info(f"  Normal: {img_normal}")
    logging.info(f"  Doente: {img_doente}")
    return True

def criar_placeholder(nome, texto):
    """Cria uma imagem placeholder caso as imagens reais não estejam disponíveis."""
    img = np.ones((512, 512), dtype=np.uint8) * 200
    
    # Adicionar texto
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, texto, (50, 256), fonte, 1, 0, 2, cv2.LINE_AA)
    
    caminho = os.path.join(FIGURAS_DIR, nome)
    cv2.imwrite(caminho, img)
    logging.info(f"Placeholder criado: {caminho}")

def main():
    """Função principal."""
    print("=" * 60)
    print("GERADOR DE FIGURAS PARA RELATÓRIO")
    print("=" * 60)
    
    # Criar diretório
    criar_diretorio_figuras()
    
    # Gerar figuras
    sucesso_seg = gerar_exemplo_segmentacao()
    sucesso_comp = gerar_exemplo_comparacao()
    
    # Criar placeholders se necessário
    if not sucesso_seg:
        logging.warning("Criando placeholders para segmentação...")
        criar_placeholder('exemplo_original_1.png', 'Imagem Original')
        criar_placeholder('exemplo_segmentado_1.png', 'Imagem Segmentada')
    
    if not sucesso_comp:
        logging.warning("Criando placeholders para comparação...")
        criar_placeholder('teste_exemplo.png', 'Imagem Teste')
        criar_placeholder('ref_normal.png', 'Ref. Normal')
        criar_placeholder('ref_doente.png', 'Ref. Doente')
    
    print("\n" + "=" * 60)
    if sucesso_seg or sucesso_comp:
        print("✓ Figuras geradas com sucesso!")
    else:
        print("⚠ Placeholders criados (imagens reais não disponíveis)")
    print(f"Diretório: {FIGURAS_DIR}/")
    
    # Listar arquivos criados
    figuras = list(Path(FIGURAS_DIR).glob('*.png'))
    if figuras:
        print(f"\nArquivos criados ({len(figuras)}):")
        for fig in sorted(figuras):
            tamanho = fig.stat().st_size / 1024
            print(f"  - {fig.name} ({tamanho:.1f} KB)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
