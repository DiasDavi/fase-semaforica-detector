# --- Importações ---
import tensorflow as tf
from numba import cuda
from config import config
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from extract_annotations import AnnotationDataset
from image_data_generator import MultiBoxDataGenerator
from metrics import iou

# --- Carregar as anotações ---
print("[INFO] Carregando o conjunto de anotações...")
ann = AnnotationDataset()
print("[SUCESSO] Conjunto de anotações carregado.\n")

data_paths = []
targets = []

print("[INFO] Iniciando processamento das imagens...")
for row in ann.rows():
    (folder, filename, size, objects) = row.values()
    img_path = os.path.join(config.BASE_PATH, folder, filename)
    
    # Cria uma lista para armazenar os bounding boxes desta imagem
    boxes = []
    for obj in objects:
        bndbox = obj['bndbox']
        xmin, ymin, xmax, ymax = bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']
        boxes.append((xmin, ymin, xmax, ymax))
    
    data_paths.append(img_path)
    targets.append(boxes)

print("[SUCESSO] Processamento das imagens concluído.\n")

# --- Divisão de treino e teste ---
print("[INFO] Realizando a divisão dos dados em treino e teste...")
split = train_test_split(data_paths, targets, test_size=0.3, random_state=42)

trainFilenames, testFilenames = split[:2]
trainTargets, testTargets = split[2:4]

print(f"[INFO] Número de imagens de treino: {len(trainFilenames)}")
print(f"[INFO] Número de imagens de teste: {len(testFilenames)}")
print("[SUCESSO] Divisão concluída.\n")

print("[INFO] Salvando os nomes dos arquivos de teste...")
with open(config.TEST_FILENAMES, "w") as f:
    f.write("\n".join(testFilenames))
print("[SUCESSO] Nomes dos arquivos de teste salvos com sucesso.\n")

# --- Criar os Generators ---
train_generator = MultiBoxDataGenerator(trainFilenames, trainTargets, batch_size=config.BATCH_SIZE, max_boxes=config.NUM_MAX_BBOX)
test_generator = MultiBoxDataGenerator(testFilenames, testTargets, batch_size=config.BATCH_SIZE, max_boxes=config.NUM_MAX_BBOX)

# --- Criar o Modelo ---
input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
base_model.trainable = False  

x = GlobalAveragePooling2D()(base_model.output)  # Reduz parâmetros
x = Dense(64, activation="relu")(x)  # Apenas uma camada densa
x = Dense(config.NUM_MAX_BBOX * 4, activation="sigmoid")(x)
output = Reshape((config.NUM_MAX_BBOX, 4))(x)

model = Model(inputs=input_tensor, outputs=output)
model.summary()

# --- Compilar e Treinar o Modelo ---
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

print("[INFO] Treinando o modelo...")
H = model.fit(train_generator, validation_data=test_generator, epochs=config.NUM_EPOCHS, verbose=1)
print("[SUCESSO] Treinamento concluído.\n")

# --- Salvar o Modelo ---
print("[INFO] Salvando o modelo...")
model.save(config.MODEL_PATH, save_format="h5")
print("[SUCESSO] Modelo salvo com sucesso.\n")

# --- Gerar Gráficos do Treinamento ---
print("[INFO] Gerando gráficos do treinamento...")
plt.style.use("ggplot")
plt.figure(figsize=(10, 4))  # Ajusta tamanho

# Gráfico de Loss
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history.get("loss", []), label="Perda no treino")
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history.get("val_loss", []), label="Perda na validação")
plt.title("Perda no treinamento")
plt.xlabel("Epoca #")
plt.ylabel("Perda")
plt.legend(loc="upper right")
plt.grid(True)

# Gráfico de IoU
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history.get("mean_io_u", []), label="IoU medio no treino")
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history.get("val_mean_io_u", []), label="IoU medio na validação")
plt.title("Intersecção sobre União (IoU)")
plt.xlabel("Epoca #")
plt.ylabel("IoU")
plt.legend(loc="lower right")
plt.grid(True)

# Ajusta layout e salva
plt.tight_layout()
plt.savefig(config.PLOT_PATH)
plt.show()  # Exibe na tela
print("[SUCESSO] Gráficos salvos com sucesso.\n")