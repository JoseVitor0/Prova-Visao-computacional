import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


pasta_de_entrada = '/content/imagens/'

def processar_e_mostrar_passos_da_imagem(caminho_da_imagem):
    nome_da_foto = os.path.basename(caminho_da_imagem)
    foto = cv2.imread(caminho_da_imagem)

    if foto is None:
        print(f"Ops! Não consegui carregar a foto '{nome_da_foto}'. Pulando.")
        return

    print(f"Processando e mostrando as etapas para: {nome_da_foto}")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(foto, cv2.COLOR_BGR2RGB))
    plt.title(f'1. Original\n({foto.shape[1]}x{foto.shape[0]})')
    plt.axis('off')

    foto_redimensionada = cv2.resize(foto, (128, 128), interpolation=cv2.INTER_AREA)
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(foto_redimensionada, cv2.COLOR_BGR2RGB))
    plt.title(f'2. Redimensionada\n(128x128)')
    plt.axis('off')

    foto_gaussiana = cv2.GaussianBlur(foto_redimensionada, (5, 5), 0)
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(foto_gaussiana, cv2.COLOR_BGR2RGB))
    plt.title('3. Filtro Gaussiano')
    plt.axis('off')

    foto_equalizada_para_mostrar = foto_gaussiana.copy()
    titulo_adicional = ""

    if len(foto_gaussiana.shape) == 3:
        foto_cinza = cv2.cvtColor(foto_gaussiana, cv2.COLOR_BGR2GRAY)
        foto_equalizada_para_mostrar = cv2.equalizeHist(foto_cinza)
        titulo_adicional = "(Tons de Cinza)"
    else:
        foto_equalizada_para_mostrar = cv2.equalizeHist(foto_gaussiana)
        titulo_adicional = "(Já Cinza)"

    plt.subplot(1, 4, 4)
    if len(foto_equalizada_para_mostrar.shape) == 2:
        plt.imshow(foto_equalizada_para_mostrar, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(foto_equalizada_para_mostrar, cv2.COLOR_BGR2RGB))
    plt.title(f'4. Equalização de Histograma\n{titulo_adicional}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


arquivos_de_imagem = [f for f in os.listdir(pasta_de_entrada) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if not arquivos_de_imagem:
    print(f"Não achei nenhuma foto na pasta '{pasta_de_entrada}'. Por favor, verifique o caminho e as extensões.")
else:
    print(f"Encontrei {len(arquivos_de_imagem)} fotos na pasta '{pasta_de_entrada}'.")
    contagem_processada = 0

    for nome_do_arquivo in arquivos_de_imagem:
        caminho_completo_da_imagem = os.path.join(pasta_de_entrada, nome_do_arquivo)
        processar_e_mostrar_passos_da_imagem(caminho_completo_da_imagem)
        contagem_processada += 1

    print(f"\n--- Prontinho! ---")
    print(f"As etapas de processamento de {contagem_processada} fotos foram exibidas.")