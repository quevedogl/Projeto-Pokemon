"""
Extrai amostras uniformes de videos e GIFs, preservando a origem de cada frame.
Gera ate 12 imagens por arquivo (ou todas, se houver menos de 12 frames).
Os frames sao salvos em data/<classe>/ como JPG para o treinamento.
"""

import os
import shutil

import cv2
import numpy as np
from PIL import Image

from utils import letterbox, VIDEO_EXT

VIDEOS_DIR = "videos"
DATA_DIR = "data"
AMOSTRAS_POR_ARQUIVO = 12
GIF_EXT = (".gif",)
EXTENSOES_ACEITAS = VIDEO_EXT + GIF_EXT
CLASSES_ESPERADAS = {"pikachu", "gengar", "bulbasaur", "fundo"}


def indices_uniformes(quantidade):
    """Seleciona indices distribuidos ao longo do arquivo."""
    if quantidade <= 0:
        return []
    return sorted(set(
        int(round(x))
        for x in np.linspace(
            0,
            quantidade - 1,
            min(AMOSTRAS_POR_ARQUIVO, quantidade)
        )
    ))


def extrair_video(caminho, pasta_saida, nome_base):
    """Extrai frames de um video usando OpenCV."""
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        print(f"  [AVISO] Nao foi possivel abrir: {os.path.basename(caminho)}")
        return 0

    quantidade = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if quantidade <= 0:
        cap.release()
        print(f"  [AVISO] Arquivo sem frames: {os.path.basename(caminho)}")
        return 0

    salvos = 0
    for indice in indices_uniformes(quantidade):
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, frame = cap.read()
        if not ok:
            continue

        imagem = letterbox(frame)
        nome = f"{nome_base}__{indice:06d}.jpg"
        destino = os.path.join(pasta_saida, nome)
        if cv2.imwrite(destino, imagem, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            salvos += 1

    cap.release()
    return salvos


def extrair_gif(caminho, pasta_saida, nome_base):
    """Extrai frames uniformes de um GIF usando Pillow."""
    salvos = 0

    try:
        with Image.open(caminho) as gif:
            quantidade = getattr(gif, "n_frames", 1)

            for indice in indices_uniformes(quantidade):
                gif.seek(indice)
                quadro = gif.convert("RGB")
                frame_rgb = np.array(quadro)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                imagem = letterbox(frame_bgr)
                nome = f"{nome_base}__{indice:06d}.jpg"
                destino = os.path.join(pasta_saida, nome)
                if cv2.imwrite(
                    destino,
                    imagem,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                ):
                    salvos += 1

    except Exception as erro:
        print(f"  [AVISO] Nao foi possivel ler o GIF "
              f"{os.path.basename(caminho)}: {erro}")
        return 0

    return salvos


if not os.path.isdir(VIDEOS_DIR):
    raise SystemExit("Pasta videos/ nao encontrada.")

classes = sorted(
    pasta for pasta in os.listdir(VIDEOS_DIR)
    if os.path.isdir(os.path.join(VIDEOS_DIR, pasta))
)
print("Classes encontradas:", classes)

faltando = CLASSES_ESPERADAS - set(classes)
if faltando:
    raise SystemExit(f"Faltam pastas de classe: {sorted(faltando)}")

# Remove apenas imagens antigas do dataset; os arquivos originais sao preservados.
if os.path.isdir(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

resumo = {}

for classe in classes:
    pasta_arquivos = os.path.join(VIDEOS_DIR, classe)
    pasta_saida = os.path.join(DATA_DIR, classe)
    os.makedirs(pasta_saida, exist_ok=True)

    arquivos = sorted(
        arquivo for arquivo in os.listdir(pasta_arquivos)
        if arquivo.lower().endswith(EXTENSOES_ACEITAS)
    )

    if len(arquivos) < 2:
        raise SystemExit(
            f"A classe '{classe}' precisa de pelo menos 2 arquivos "
            f"(videos ou GIFs)."
        )

    total = 0

    for arquivo in arquivos:
        caminho = os.path.join(pasta_arquivos, arquivo)
        nome_base = os.path.splitext(arquivo)[0].replace("__", "_")

        if arquivo.lower().endswith(GIF_EXT):
            salvos = extrair_gif(caminho, pasta_saida, nome_base)
        else:
            salvos = extrair_video(caminho, pasta_saida, nome_base)

        total += salvos
        print(f"  {classe}/{arquivo}: {salvos} frames")

    resumo[classe] = total

print("\nTotal por classe:", resumo)

if len(set(resumo.values())) > 1:
    print(
        "Observacao: as classes podem ter quantidades diferentes de imagens. "
        "Tente manter uma quantidade semelhante de arquivos por classe."
    )

print("\nExtracao concluida. Proximo passo: py -3.12 train.py")
print("Confira o dataset antes de iniciar o treinamento.")
