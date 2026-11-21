FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_DIR=/app/models

# Solo lo necesario para OpenCV headless y descargar los modelos
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# PyTorch solo-CPU (la version normal trae CUDA y pesa varios GB) + dependencias
# (sus dependencias se instalan desde PyPI: el indice de pytorch trae metadatos que pip rechaza)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir typing-extensions filelock sympy networkx jinja2 fsspec && \
    pip install --no-cache-dir --no-deps torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Segundo lector de dorsales: Florence-2 (Microsoft, MIT), version nativa de transformers (sin codigo remoto),
# revision fija = la misma que se probo. Medido: dorsales correctos 4 -> 10 en las fotos del cliente, 0 mezclas.
ENV HF_HOME=/app/models/hf \
    FLORENCE_REPO=florence-community/Florence-2-base \
    FLORENCE_REVISION=00921df66db728a9ceb750f5eca43e5c203a2051 \
    FLORENCE_HILOS=2 \
    MALLOC_ARENA_MAX=2 \
    ANALISIS_HILOS=1 \
    LECTOR_SMOL=0 \
    SMOL_REPO=HuggingFaceTB/SmolVLM-256M-Instruct \
    SMOL_REVISION=d039ab0d4b9311b617e8ffec1d06c8464364aeb3
# Tercer lector: SmolVLM-256M (Apache 2.0). Se le pregunta el numero; acerto 6/6 numeros que nadie leia.
RUN python -c "from transformers import AutoProcessor, AutoModelForImageTextToText as M; import os; r=os.environ['SMOL_REPO']; v=os.environ['SMOL_REVISION']; AutoProcessor.from_pretrained(r, revision=v); M.from_pretrained(r, revision=v)"
RUN python -c "from transformers import AutoProcessor, Florence2ForConditionalGeneration as F; import os; r=os.environ['FLORENCE_REPO']; v=os.environ['FLORENCE_REVISION']; AutoProcessor.from_pretrained(r, revision=v); F.from_pretrained(r, revision=v)"
ENV HF_HUB_OFFLINE=1

# Modelos biometricos oficiales de OpenCV Zoo (Apache-2.0 / MIT, uso comercial permitido).
# Se verifica la huella SHA-256 de cada archivo: Render usa exactamente lo que se probo.
RUN mkdir -p /app/models && cd /app/models && \
    Z=https://github.com/opencv/opencv_zoo/raw/main/models && \
    curl -fsSL --retry 5 -o object_detection_yolox_2022nov.onnx     $Z/object_detection_yolox/object_detection_yolox_2022nov.onnx && \
    curl -fsSL --retry 5 -o face_detection_yunet_2023mar.onnx       $Z/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    curl -fsSL --retry 5 -o face_recognition_sface_2021dec.onnx     $Z/face_recognition_sface/face_recognition_sface_2021dec.onnx && \
    curl -fsSL --retry 5 -o text_detection_en_ppocrv3_2023may.onnx  $Z/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx && \
    curl -fsSL --retry 5 -o text_recognition_CRNN_EN_2021sep.onnx   $Z/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx && \
    printf '%s\n' \
      "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063  object_detection_yolox_2022nov.onnx" \
      "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet_2023mar.onnx" \
      "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  face_recognition_sface_2021dec.onnx" \
      "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587  text_detection_en_ppocrv3_2023may.onnx" \
      "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544  text_recognition_CRNN_EN_2021sep.onnx" \
      | sha256sum -c -

COPY . .

EXPOSE 10000

# UNA sola copia del servidor: Render pone WEB_CONCURRENCY=2 y cada copia cargaba los modelos (~2 GB c/u)
# -> 2 x 2 GB > 4 GB -> "Ran out of memory" (medido 21 nov 2026). El analisis va en un trabajador aislado.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --timeout-keep-alive 75"]
