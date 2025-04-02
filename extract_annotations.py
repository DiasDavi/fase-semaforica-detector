from config import config
import os
import xml.etree.ElementTree as ET
import cv2
import json

class AnnotationDataset:
    def __init__(self):
        self.base_path = config.BASE_PATH
        self.images_path = config.IMAGES_PATH
        self.annotations = []

        # Tenta carregar as anotações do arquivo JSON, se existir
        if os.path.exists(config.ANNOTS_PATH):
            self.load_annotations()

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotation = {
            'folder': "images",
            'filename': root.find('filename').text,
            'size': {
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'depth': int(root.find('size/depth').text)
            },
            'object': []
        }

        for obj in root.findall('object'):
            obj_data = {
                'name': obj.find('name').text,
                'bndbox': {
                    'xmin': int(obj.find('bndbox/xmin').text),
                    'ymin': int(obj.find('bndbox/ymin').text),
                    'xmax': int(obj.find('bndbox/xmax').text),
                    'ymax': int(obj.find('bndbox/ymax').text)
                }
            }
            annotation['object'].append(obj_data)

        return annotation

    def process_folders(self):
        dataset = []
        for file in os.listdir(self.images_path):
            if file.endswith('.xml'):
                xml_path = os.path.join(self.images_path, file)
                annotation = self.parse_xml(xml_path)
                dataset.append(annotation)

        self.annotations = dataset

        with open(config.ANNOTS_PATH, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)

        return dataset

    def load_annotations(self):
        """Carrega as anotações a partir do arquivo JSON."""
        with open(config.ANNOTS_PATH, 'r') as json_file:
            self.annotations = json.load(json_file)

    def img(self, index, show):
        if not self.annotations:
            print("Nenhuma anotação foi carregada. Execute process_folders() primeiro.")
            return

        annotation = self.annotations[index]
        img_path = os.path.join(self.base_path, annotation['folder'], annotation['filename'])
        img = cv2.imread(img_path)

        for obj in annotation['object']:
            xmin, ymin, xmax, ymax = obj['bndbox'].values()
            label = obj['name']

            # Desenha retângulo na imagem
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Adiciona label acima do bounding box
            cv2.putText(img, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if show:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            return img

    def metadados(self, index):
        if not self.annotations:
            print("Nenhuma anotação foi carregada. Execute process_folders() primeiro.")
            return

        annotation = self.annotations[index]['object']
        return annotation
    
    def rows(self):
        return self.annotations


# Criar dataset e processar as pastas
ann = AnnotationDataset()
ann.process_folders()

# ann.img(3, True)
