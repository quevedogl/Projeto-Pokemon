# Projeto Pokemon - Reconhecimento em video

## O que ja esta incluido
- Os 16 videos originais enviados, separados em `videos/`.
- Dataset `data/` refeito com 12 amostras uniformes por video.
- `extrair_frames.py`, `train.py`, `detectar.py` e `utils.py` revisados.
- `dataset_imagens.pdf` com amostras por classe.
- `requirements.txt` e este guia.

## Importante sobre os videos atuais
Os videos foram inspecionados visualmente. Alguns mostram o Pokemon grande e
nitido; outros mostram a imagem muito pequena dentro da tela do celular,
com bastante borda preta. O codigo aproveita os videos, mas nao consegue
recuperar detalhes que nao aparecem na gravacao. O dataset equilibrado
evita que uma classe tenha muito mais imagens que outra, mas nao garante
acuracia alta. Para melhorar, grave futuramente os Pokemons ocupando boa
parte da imagem e varie os cenarios.

## 1. Preparar o ambiente (Windows / PowerShell)
Recomendado usar Python 3.11 ou 3.12 em ambiente virtual:

    py -3.12 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Se o comando `py -3.12` nao existir, confira as versoes com `py --list`.
O TensorFlow pode ter suporte diferente conforme a versao do Python.

## 2. Extrair os frames
    python extrair_frames.py

O script recria apenas a pasta `data/`; nao apaga nem altera os videos.
Sao selecionados ate 12 frames distribuidos ao longo de cada video.
Como ha 4 videos por classe, o esperado e cerca de 48 imagens por classe.

## 3. Treinar
    python train.py

O treinamento tem duas fases: MobileNetV2 congelada e fine-tuning com
learning rate menor. O split padrao e por video, evitando que frames do
mesmo video aparecam simultaneamente no treino e na validacao.

Arquivos gerados:
- `modelo.keras`
- `modelo_fase1.keras`
- `class_names.json`
- `curvas.png`
- `matriz_confusao.png`
- `dataset_imagens.pdf`

A primeira execucao pode precisar baixar os pesos ImageNet da MobileNetV2.

## 4. Testar
Grave um video de teste de 30 a 60 segundos, em ambiente diferente, com
cada Pokemon aparecendo separadamente. Guarde fora de `videos/` para nao
contaminar o treinamento. Depois execute, por exemplo:

    python detectar.py video_teste.mp4

Gera `saida_video_teste.mp4` e `linha_do_tempo_video_teste.txt`.
Webcam:

    python detectar.py 0

Ajustes:

    python detectar.py video_teste.mp4 --janela 15 --limiar 0.70

## 5. Experimentos da atividade
- Split por video x por foto:

      python train.py --split video --tag _video
      python train.py --split foto --tag _foto --sem-pdf

- Fine-tuning com taxa maior:

      python train.py --ft-lr 1e-3 --tag _ft1e3 --sem-pdf

- Excluir frames de costas (se os nomes dos arquivos de origem contiverem
  "costas"):

      python train.py --excluir pikachu:costas --tag _semcostas --sem-pdf

- Pokemon desconhecido e dois Pokemon no mesmo quadro: usar videos de teste
  separados, fora da pasta `videos/`. O classificador escolhe uma classe
  para o quadro inteiro; nao desenha caixas nem localiza dois objetos.

## Observacao
O modelo treinado nao esta incluido porque precisa ser treinado no ambiente
local, com TensorFlow instalado e os pesos iniciais carregados. Os resultados
de acuracia dependem dos dados e nao devem ser inventados ou presumidos.
