"""
Extrai amostras uniformes dos videos, preservando a origem de cada frame.
Gera 12 imagens por video (ou todas, se o video tiver menos de 12 frames).
Isso evita que videos longos dominem o dataset e mantem as classes equilibradas
quando cada classe possui a mesma quantidade de videos.
"""
import os
import shutil
import cv2
from utils import letterbox, VIDEO_EXT

VIDEOS_DIR = "videos"
DATA_DIR = "data"
AMOSTRAS_POR_VIDEO = 12
CLASSES_ESPERADAS = {"pikachu", "gengar", "bulbasaur", "fundo"}

if not os.path.isdir(VIDEOS_DIR):
    raise SystemExit("Pasta videos/ nao encontrada.")

classes = sorted(
    d for d in os.listdir(VIDEOS_DIR)
    if os.path.isdir(os.path.join(VIDEOS_DIR, d))
)
print("Classes encontradas:", classes)

faltando = CLASSES_ESPERADAS - set(classes)
if faltando:
    raise SystemExit(f"Faltam pastas de classe: {sorted(faltando)}")

# Remove apenas imagens antigas do dataset; os videos originais sao preservados.
if os.path.isdir(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

resumo = {}
for classe in classes:
    pasta_videos = os.path.join(VIDEOS_DIR, classe)
    pasta_saida = os.path.join(DATA_DIR, classe)
    os.makedirs(pasta_saida, exist_ok=True)
    videos = sorted(
        f for f in os.listdir(pasta_videos)
        if f.lower().endswith(VIDEO_EXT)
    )
    if len(videos) < 2:
        raise SystemExit(f"A classe '{classe}' precisa de pelo menos 2 videos.")

    total = 0
    for video in videos:
        caminho = os.path.join(pasta_videos, video)
        cap = cv2.VideoCapture(caminho)
        if not cap.isOpened():
            print(f"  [AVISO] Nao foi possivel abrir: {video}")
            continue

        quantidade = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if quantidade <= 0:
            cap.release()
            print(f"  [AVISO] Video sem frames: {video}")
            continue

        indices = sorted(set(
            int(round(x)) for x in
            __import__("numpy").linspace(
                0, max(0, quantidade - 1),
                min(AMOSTRAS_POR_VIDEO, quantidade)
            )
        ))
        nome_base = os.path.splitext(video)[0].replace("__", "_")
        salvos = 0
        for indice in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
            ok, frame = cap.read()
            if not ok:
                continue
            imagem = letterbox(frame)
            nome = f"{nome_base}__{indice:06d}.jpg"
            cv2.imwrite(
                os.path.join(pasta_saida, nome),
                imagem,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            )
            salvos += 1
        cap.release()
        total += salvos
        print(f"  {classe}/{video}: {salvos} frames")
    resumo[classe] = total

print("\\nTotal por classe:", resumo)
if len(set(resumo.values())) > 1:
    print("Observacao: pequenas diferencas podem ocorrer se algum video nao puder ser lido.")
print("\\nExtracao concluida. Proximo passo: python train.py")
print("Confira o PDF dataset_imagens.pdf gerado pelo treinamento.")
