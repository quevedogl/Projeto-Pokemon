"""
Treinamento Pokemon com MobileNetV2.
Padrao: separacao por video (evita vazamento entre treino e validacao).
"""
import os, json, random, argparse
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["video", "foto"], default="video")
parser.add_argument("--excluir", action="append", default=[])
parser.add_argument("--epocas", type=int, default=8)
parser.add_argument("--epocas-ft", type=int, default=8)
parser.add_argument("--ft-lr", type=float, default=1e-5)
parser.add_argument("--ft-camadas", type=int, default=30)
parser.add_argument("--tag", default="")
parser.add_argument("--sem-pdf", action="store_true")
args = parser.parse_args()

DATA_DIR, IMG_SIZE, BATCH, SEED = "data", 224, 16, 42
VAL_FRACTION = 0.25
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

classes = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))
if len(classes) < 2:
    raise SystemExit("Dataset insuficiente: adicione pelo menos duas classes.")
print("Classes:", classes)
print("Split:", args.split)

exclusoes = defaultdict(list)
for regra in args.excluir:
    if ":" not in regra:
        raise SystemExit("--excluir deve estar no formato classe:texto")
    classe, trecho = regra.split(":", 1)
    exclusoes[classe].append(trecho.lower())

train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, classe in enumerate(classes):
    pasta = os.path.join(DATA_DIR, classe)
    arquivos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
    if classe in exclusoes:
        arquivos = [f for f in arquivos if not any(t in f.lower() for t in exclusoes[classe])]
    if not arquivos:
        raise SystemExit(f"Nenhuma imagem disponivel para a classe '{classe}'.")

    caminhos = [os.path.join(pasta, f) for f in arquivos]
    if args.split == "video":
        por_video = defaultdict(list)
        for nome, caminho in zip(arquivos, caminhos):
            # O sufixo __000123 identifica o frame; o prefixo identifica o video.
            origem = nome.rsplit("__", 1)[0] if "__" in nome else nome
            por_video[origem].append(caminho)
        videos = sorted(por_video)
        if len(videos) < 2:
            raise SystemExit(
                f"'{classe}' tem {len(videos)} video(s) apos filtros. "
                "Sao necessarios pelo menos 2."
            )
        random.Random(SEED).shuffle(videos)
        n_val = max(1, min(len(videos) - 1, round(len(videos) * VAL_FRACTION)))
        videos_validacao = set(videos[:n_val])
        for video in videos:
            destino = (val_paths, val_y) if video in videos_validacao else (train_paths, train_y)
            destino[0].extend(por_video[video])
            destino[1].extend([idx] * len(por_video[video]))
        ntr = sum(len(por_video[v]) for v in videos if v not in videos_validacao)
        nva = sum(len(por_video[v]) for v in videos if v in videos_validacao)
        print(f"  {classe}: {ntr} treino / {nva} validacao; "
              f"videos validacao={sorted(videos_validacao)}")
    else:
        rng = random.Random(SEED)
        rng.shuffle(caminhos)
        n_val = max(1, round(len(caminhos) * VAL_FRACTION))
        val_paths.extend(caminhos[:n_val]); val_y.extend([idx] * n_val)
        train_paths.extend(caminhos[n_val:]); train_y.extend([idx] * (len(caminhos) - n_val))
        print(f"  {classe}: {len(caminhos)-n_val} treino / {n_val} validacao (por foto)")

if not train_paths or not val_paths:
    raise SystemExit("Nao foi possivel montar treino e validacao.")

def carregar(caminho, rotulo):
    img = tf.io.read_file(caminho)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, rotulo

def dataset(caminhos, rotulos, treino):
    ds = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    if treino:
        ds = ds.shuffle(len(caminhos), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

treino_ds = dataset(train_paths, train_y, True)
validacao_ds = dataset(val_paths, val_y, False)

aumento = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomContrast(0.15),
], name="aumento_dados")

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False

entrada = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = aumento(entrada)
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
saida = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
modelo = tf.keras.Model(entrada, saida)

def avaliar(titulo, salvar=False):
    verdade = np.asarray(val_y, dtype=int)
    previsoes = np.argmax(modelo.predict(validacao_ds, verbose=0), axis=1)
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for real, previsto in zip(verdade, previsoes):
        matriz[real, previsto] += 1
    print(f"\\n===== {titulo} =====")
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(" " * 14 + " ".join(f"{c[:10]:>10}" for c in classes))
    for i, classe in enumerate(classes):
        print(f"{classe[:12]:>12} " + " ".join(f"{v:>10}" for v in matriz[i]))
    print("\\nClasse          precisao    recall")
    for i, classe in enumerate(classes):
        precisao = matriz[i, i] / max(1, matriz[:, i].sum())
        recall = matriz[i, i] / max(1, matriz[i, :].sum())
        print(f"{classe[:14]:<14} {precisao:>8.1%} {recall:>8.1%}")
    acuracia = np.trace(matriz) / max(1, matriz.sum())
    print(f"Acuracia validacao: {acuracia:.1%}")
    if salvar:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matriz, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        ax.set_title("Matriz de confusao - validacao")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matriz[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(f"matriz_confusao{args.tag}.png", dpi=160)
        plt.close(fig)
    return acuracia

print("\\n=== FASE 1: MobileNetV2 congelada ===")
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
parada1 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
hist1 = modelo.fit(treino_ds, validation_data=validacao_ds,
                   epochs=args.epocas, callbacks=[parada1])
modelo.save(f"modelo_fase1{args.tag}.keras")
acc1 = avaliar("Fase 1", salvar=False)

print(f"\\n=== FASE 2: fine-tuning (ultimas {args.ft_camadas} camadas; lr={args.ft_lr}) ===")
base.trainable = True
for camada in base.layers[:-args.ft_camadas]:
    camada.trainable = False
# BatchNormalization permanece congelada para estabilidade em dataset pequeno.
for camada in base.layers:
    if isinstance(camada, tf.keras.layers.BatchNormalization):
        camada.trainable = False
modelo.compile(optimizer=tf.keras.optimizers.Adam(args.ft_lr),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
parada2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
hist2 = modelo.fit(treino_ds, validation_data=validacao_ds,
                   epochs=args.epocas_ft, callbacks=[parada2])

modelo.save(f"modelo{args.tag}.keras")
with open("class_names.json", "w", encoding="utf-8") as arq:
    json.dump(classes, arq, ensure_ascii=False, indent=2)
acc2 = avaliar(f"Fase 2 (fine-tuning lr={args.ft_lr})", salvar=True)
print(f"\\nResumo: fase 1={acc1:.1%}; fase 2={acc2:.1%}")

n1 = len(hist1.history["loss"])
fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
for ax, metrica, titulo in [
    (eixos[0], "accuracy", "Acuracia"),
    (eixos[1], "loss", "Loss")
]:
    ax.plot(hist1.history[metrica] + hist2.history[metrica], label="treino")
    ax.plot(hist1.history["val_" + metrica] + hist2.history["val_" + metrica],
            label="validacao")
    ax.axvline(n1 - 0.5, color="gray", linestyle="--", label="inicio fine-tuning")
    ax.set_title(titulo); ax.set_xlabel("Epoca"); ax.legend()
fig.tight_layout(); fig.savefig(f"curvas{args.tag}.png", dpi=160)
plt.close(fig)

def gerar_pdf_dataset(pasta="data", destino="dataset_imagens.pdf"):
    paginas = []
    fonte = ImageFont.load_default()
    for classe in classes:
        arquivos = sorted(
            f for f in os.listdir(os.path.join(pasta, classe))
            if f.lower().endswith(".jpg")
        )
        largura, altura = 1240, 1754
        pagina = Image.new("RGB", (largura, altura), "white")
        draw = ImageDraw.Draw(pagina)
        draw.text((35, 30), f"Classe: {classe} | {len(arquivos)} imagens", fill="black", font=fonte)
        cols, cell, topo = 6, 190, 75
        for i, nome in enumerate(arquivos[:48]):
            img = Image.open(os.path.join(pasta, classe, nome)).convert("RGB")
            img.thumbnail((cell - 8, cell - 8))
            x = 35 + (i % cols) * cell
            y = topo + (i // cols) * cell
            pagina.paste(img, (x, y))
        paginas.append(pagina)
    paginas[0].save(destino, save_all=True, append_images=paginas[1:],
                    resolution=150)
    print(f"PDF do dataset salvo: {destino}")

if not args.sem_pdf:
    gerar_pdf_dataset()
print("\\nTreinamento concluido.")
print(f"Arquivos: modelo{args.tag}.keras, curvas{args.tag}.png, matriz_confusao{args.tag}.png")
