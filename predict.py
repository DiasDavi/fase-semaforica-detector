from config import config
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from metrics import iou
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

# Argumentos para receber a imagem de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["image"])[0]
imagePath = args["image"]
imagePaths = []

# Verifica se é um arquivo de texto com múltiplas imagens ou uma única imagem
if filetype == "text/plain":
    filenames = open(args["image"]).read().strip().split("\n")
    for f in filenames:
        p = os.path.join(config.BASE_PATH, config.IMAGES_PATH, f)
        imagePaths.append(p)
else:
    imagePaths.append(imagePath)

print("[INFO] Carregando o modelo detector de objetos...")
model = load_model(config.MODEL_PATH, custom_objects={"iou": iou})

for imagePath in imagePaths:
    # Carregar e preparar a imagem
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Realizar a predição
    preds = model.predict(image)[0]  # Saída será (NUM_MAX_BBOX, 4)

    # Carregar a imagem original para exibição
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Iterar sobre todos os bounding boxes preditos
    for i in range(config.NUM_MAX_BBOX):
        (startX, startY, endX, endY) = preds[i]

        # Converter para coordenadas da imagem original
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        # Desenhar o bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Exibir a imagem com as detecções
    cv2.imshow("Output", image)
    cv2.waitKey(0)
