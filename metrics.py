import tensorflow as tf
from config import config
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    # y_true e y_pred: (batch_size, max_boxes, 4)
    # print("\ny_true:", y_true[0])
    # print("\ny_pred:", y_pred[0])
    
    iou_list = []
    for true_bboxes, pred_bboxes in zip(y_true, y_pred):

        for i in range (config.NUM_MAX_BBOX):
            true = true_bboxes[i]
            pred = pred_bboxes[i]
            # Calcular as coordenadas da interseção
            xmin_inter = tf.maximum(true[0], pred[0])  # max(xmin1, xmin2)
            ymin_inter = tf.maximum(true[1], pred[1])  # max(ymin1, ymin2)
            xmax_inter = tf.minimum(true[2], pred[2])  # min(xmax1, xmax2)
            ymax_inter = tf.minimum(true[3], pred[3])  # min(ymax1, ymax2)

            # Verificar interseção
            # print(f"true: {true.numpy()}")
            # print(f"pred: {pred.numpy()}")
            # print(f"Intersection coords: xmin: {xmin_inter}, ymin: {ymin_inter}, xmax: {xmax_inter}, ymax: {ymax_inter}")

            # Calcular a largura e altura da interseção
            inter_width = tf.maximum(0.0, xmax_inter - xmin_inter)
            inter_height = tf.maximum(0.0, ymax_inter - ymin_inter)
            intersection_area = inter_width * inter_height

            # Verificar as áreas da interseção
            # print(f"Intersection area: {intersection_area}")

            # Calcular as áreas das caixas verdadeiras e previstas
            true_area = (true[2] - true[0]) * (true[3] - true[1])
            pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])

            # Verificar as áreas das caixas
            # print(f"True area: {true_area}, Pred area: {pred_area}")

            # Calcular a área da união
            union_area = true_area + pred_area - intersection_area

            # Verificar a área de união
            # print(f"Union area: {union_area}")

            # Calcular o IoU
            iou = intersection_area / union_area
            # print(f"IoU: {iou.numpy()}")

            iou_list.append(iou)

    # Retornar todos os valores de IoU
    return iou_list