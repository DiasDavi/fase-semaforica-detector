from config import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from extract_annotations import AnnotationDataset
from image_data_generator import MultiBoxDataGenerator
from metrics import iou

# --- 1. Carregar as anotações ---
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

# --- 2. Divisão de treino e teste ---
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

# --- 3. Definir parâmetros e criar os Generators ---
train_generator = MultiBoxDataGenerator(trainFilenames, trainTargets, batch_size=config.BATCH_SIZE, max_boxes=config.NUM_MAX_BBOX)
test_generator = MultiBoxDataGenerator(testFilenames, testTargets, batch_size=config.BATCH_SIZE, max_boxes=config.NUM_MAX_BBOX)

# --- 4. Criar o Modelo ---
input_tensor = Input(shape=(224, 224, 3))
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=input_tensor)
vgg.trainable = False  # Congela os pesos do VGG16

x = Flatten()(vgg.output)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
# Saída: max_boxes * 4 valores, normalizados entre 0 e 1 (graças à ativação sigmoide)
x = Dense(config.NUM_MAX_BBOX * 4, activation="sigmoid")(x)
output = Reshape((config.NUM_MAX_BBOX, 4))(x)

model = Model(inputs=input_tensor, outputs=output)
model.summary()

# --- 5. Compilar e Treinar o Modelo ---
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="mse", metrics=[iou, 'accuracy'], run_eagerly=True)

print("[INFO] Treinando o modelo...")
H = model.fit(train_generator, validation_data=test_generator, epochs=config.NUM_EPOCHS, verbose=1)
print("[SUCESSO] Treinamento concluído.\n")

# --- 6. Salvar o Modelo ---
print("[INFO] Salvando o modelo...")
model.save(config.MODEL_PATH, save_format="h5")
print("[SUCESSO] Modelo salvo com sucesso.\n")

# --- 7. Gerar Gráficos do Treinamento ---
print("[INFO] Gerando gráficos do treinamento...")
plt.style.use("ggplot")
plt.figure(figsize=(12, 5))

# Gráfico de Loss
plt.subplot(1, 3, 1)
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")

# Gráfico de IoU
plt.subplot(1, 3, 2)
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["iou"], label="train_iou")
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_iou"], label="val_iou")
plt.title("Training IoU")
plt.xlabel("Epoch #")
plt.ylabel("IoU")
plt.legend(loc="lower right")

# Gráfico de Acurácia
plt.subplot(1, 3, 3)
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

# Salvar o gráfico
plt.tight_layout()
plt.savefig(config.PLOT_PATH)
print("[SUCESSO] Gráficos salvos com sucesso.\n")