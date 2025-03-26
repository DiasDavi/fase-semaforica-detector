from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class ImageDataGenerator(Sequence):
    def __init__(self, image_paths, targets, batch_size=32, img_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.targets = targets
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images_paths = [self.image_paths[i] for i in batch_indexes]
        batch_targets_paths = np.array([self.targets[i] for i in batch_indexes], dtype="float32")

        batch_images = []
        batch_targets = []

        for path, target in zip(batch_images_paths, batch_targets_paths):
            # Carrega imagem original para obter dimensões corretas
            img = load_img(path)
            width, height = img.size  

            # Redimensiona para img_size
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img)

            batch_images.append(img)

            # Normaliza bounding boxes com base no tamanho original
            x_min, y_min, x_max, y_max = target
            xmin = float(x_min) / width
            ymin = float(y_min) / height
            xmax = float(x_max) / width
            ymax = float(y_max) / height
            batch_targets.append((xmin, ymin, xmax, ymax))

        batch_images = np.array(batch_images, dtype="float32") / 255.0
        batch_targets = np.array(batch_targets, dtype="float32")           

        return batch_images, batch_targets

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class MultiBoxDataGenerator(Sequence):
    def __init__(self, image_paths, targets, batch_size=32, img_size=(224, 224), max_boxes=5, shuffle=True):
        """
        image_paths: lista de caminhos para as imagens.
        targets: lista onde cada item é uma lista de bounding boxes para a imagem (cada caixa: (xmin, ymin, xmax, ymax)).
        """
        self.image_paths = image_paths
        self.targets = targets
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_boxes = max_boxes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images_paths = [self.image_paths[i] for i in batch_indexes]
        batch_targets_list = [self.targets[i] for i in batch_indexes]

        batch_images = []
        batch_bboxes = []

        for path, boxes in zip(batch_images_paths, batch_targets_list):
            # Carrega a imagem original para obter as dimensões originais
            img_orig = load_img(path)
            width, height = img_orig.size  

            # Carrega a imagem redimensionada para a rede
            img = load_img(path, target_size=self.img_size)
            img = img_to_array(img)
            batch_images.append(img)

            normalized_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                xmin = xmin / width
                ymin = ymin / height
                xmax = xmax / width
                ymax = ymax / height
                normalized_boxes.append((xmin, ymin, xmax, ymax))

            # Converte para numpy array para usar .shape
            boxes = np.array(normalized_boxes, dtype="float32")

            # Se houver mais caixas que max_boxes, utiliza apenas as primeiras
            if boxes.shape[0] > self.max_boxes:
                boxes = boxes[:self.max_boxes]

            # Se houver menos, preenche com zeros (ou outro valor que indique "não existe")
            pad = self.max_boxes - boxes.shape[0]
            if pad > 0:
                boxes = np.pad(boxes, ((0, pad), (0, 0)), mode='constant', constant_values=0)

            batch_bboxes.append(boxes)

        # Agora todas as caixas têm o mesmo tamanho (max_boxes)
        batch_images = np.array(batch_images, dtype="float32") / 255.0
        batch_bboxes = np.array(batch_bboxes, dtype="float32")

        return batch_images, batch_bboxes