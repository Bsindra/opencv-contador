#! /bin/bash

# Instalando dependências
pip3 install -r requirements

# Instalando pesos do Yolo v4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# Alterando a permissão de execução
sudo chmod +x main.py
