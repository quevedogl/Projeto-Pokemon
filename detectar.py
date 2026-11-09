"""
Reconhecimento por video ou webcam.
Exemplos:
  python detectar.py video_teste.mp4
  python detectar.py 0
  python detectar.py video_teste.mp4 --janela 15 --limiar 0.60
"""
import os, json, argparse
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

parser = argparse.ArgumentParser()
parser.add_argument("fonte", nargs="?", default="0")
parser.add_argument("--janela", type=int, default=8)
parser.add_argument("--limiar", type=float, default=0.60)
parser.add_argument("--min-trecho", type=float, default=0.5)
parser.add_argument("--modelo", default="modelo.keras")
parser.add_argument("--sem-janela", action="store_true",
                    help="nao abre janela; util para processamento sem interface")
args = parser.parse_args()

if args.janela < 1:
    raise SystemExit("--janela deve ser pelo menos 1.")
if not 0 <= args.limiar <= 1:
    raise SystemExit("--limiar deve estar entre 0 e 1.")
if not os.path.isfile(args.modelo):
    raise SystemExit(f"Modelo nao encontrado: {args.modelo}. Rode python train.py primeiro.")
if not os.path.isfile("class_names.json"):
    raise SystemExit("class_names.json nao encontrado. Rode python train.py primeiro.")

modelo = tf.keras.models.load_model(args.modelo)
with open("class_names.json", encoding="utf-8") as f:
    classes = json.load(f)

fonte = args.fonte
webcam = fonte.isdigit()
captura = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not captura.isOpened():
    raise SystemExit(f"Nao foi possivel abrir a fonte: {fonte}")

fps = captura.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

nome_base = "webcam" if webcam else os.path.splitext(os.path.basename(fonte))[0]
escritor = None
saida = None
if not webcam:
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{nome_base}.mp4"
    escritor = cv2.VideoWriter(
        saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )
    if not escritor.isOpened():
        captura.release()
        raise SystemExit("Nao foi possivel criar o video de saida.")

historico = deque(maxlen=args.janela)
rotulos = []
frame_num = 0
try:
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        if webcam:
            frame = cv2.flip(frame, 1)

        imagem = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
        probabilidades = modelo(imagem[None, ...], training=False).numpy()[0]
        historico.append(probabilidades)
        media = np.mean(historico, axis=0)
        indice = int(np.argmax(media))
        confianca = float(media[indice])
        rotulo = classes[indice] if confianca >= args.limiar else "?"
        rotulos.append(rotulo)

        if rotulo == "?":
            texto, cor = f"? ({confianca:.0%})", (0, 0, 255)
        elif rotulo == "fundo":
            texto, cor = f"Sem Pokemon ({confianca:.0%})", (0, 200, 255)
        else:
            texto, cor = f"{rotulo} ({confianca:.0%})", (0, 220, 0)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(frame, texto, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cor, 2)
        cv2.putText(frame, f"t={frame_num / fps:.1f}s | janela={args.janela} | limiar={args.limiar:.0%}",
                    (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
        if escritor is not None:
            escritor.write(frame)
        if not args.sem_janela:
            cv2.imshow("Reconhecimento de Pokemon", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_num += 1
finally:
    captura.release()
    if escritor is not None:
        escritor.release()
    cv2.destroyAllWindows()

if saida:
    print(f"Video anotado salvo em: {saida}")

# Linha do tempo: agrupa rotulos consecutivos e ignora fundo, ? e trechos curtos.
trechos = []
inicio = 0
while inicio < len(rotulos):
    fim = inicio
    while fim + 1 < len(rotulos) and rotulos[fim + 1] == rotulos[inicio]:
        fim += 1
    rotulo = rotulos[inicio]
    duracao = (fim - inicio + 1) / fps
    if rotulo not in ("?", "fundo") and duracao >= args.min_trecho:
        trechos.append((inicio / fps, (fim + 1) / fps, rotulo))
    inicio = fim + 1

linhas = ["Linha do tempo - Pokemon identificados:"]
for comeco, termino, nome in trechos:
    linhas.append(f"  {comeco:.1f}s a {termino:.1f}s -> {nome}")
encontrados = sorted({nome for _, _, nome in trechos})
linhas.append("Pokemon encontrados: " + (", ".join(encontrados) if encontrados else "nenhum"))
texto = "\\n".join(linhas)
print("\\n" + texto)
with open(f"linha_do_tempo_{nome_base}.txt", "w", encoding="utf-8") as f:
    f.write(texto + "\\n")
