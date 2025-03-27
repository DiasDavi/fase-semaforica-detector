# Treino do detector de semaforos

## Descrição do Projeto

Este projeto tem como objetivo processar, gerar e avaliar anotações de detecção de objetos a partir de imagens. O sistema inclui scripts para extrair anotações de arquivos XML, criar um gerador de dados para treinamento de modelos e calcular métricas para avaliação de bounding boxes.

## Estrutura do Projeto

```plaintext
.
├── config/                       
│   ├── config.py                 # Arquivo de configurações
├── dataset/                       
│   ├── dayClip1                  # Frames e anotações XML
│   ├── dayClip2                  # Frames e anotações XML
│   ├── dayClip3                  # Frames e anotações XML
│   ├── annotations.json          # Anotações em formato JSON
├── output/                       
│   ├── detector.h5               # Modelo treinado
│   ├── plot.png                  # Gráficos do treinamento
│   ├── test_files.txt            # Lista dos arquivos utilizados no treino
├── requirements/                       
│   ├── requirements-gpu.txt      # Dependências para GPU
│   ├── requirements.txt          # Dependências para CPU
├── extract_annotations.py        # Conversor de anotações XML para JSON
├── image_data_generator.py       # Gerador de dados para treinamento
├── metrics.py                    # Cálculo de métricas de desempenho
├── train.py                      # Script principal de treinamento
```

## Dependências
A versão do python utilizado foi o Python 3.10.11

Recomendo criar um ambiente virtual para instalar as dependencias
```bash
python -m venv venv
```

para inicializar o ambiente virtual:
```bash
.\venv\Scripts\activate
```

Caso esteja utilizando GPU
```bash
pip install -r requirements/requirements-gpu.txt
```
Caso contrario
```bash
pip install -r requirements/requirements.txt
```

## Uso dos Scripts

### Extração de Anotações
O script `extract_annotations.py` converte anotações no formato XML (Pascal VOC) para JSON.

### Treinamento do Modelo
O script `train.py` é responsável por treinar a rede neural com os dados processados.

#### Resultados
Os arquivos gerados após a execução dos scripts incluem:

* `output/detector.h5`: Modelo treinado.
* `output/plot.png`: Gráficos do treinamento.
* `output/test_files.txt`: Lista dos arquivos utilizados no treinamento.

### Previsão do imagens
O script `predict.py` é responsavel por realizar a previsão dos bboxes da imagem
```bash
python predict.py --image CAMINHO_DA_IMAGEM
```
