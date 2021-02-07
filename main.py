import cv2
import numpy as np
import os
import time

# Fonte dos arquivos 
# Fazer alterável
fileDir = os.path.dirname(os.path.realpath('__file__'))

labels_path = os.path.join(fileDir, 'coco.names')
weights_path = os.path.join(fileDir, 'yolov4.weights')
cfg_path = os.path.join(fileDir, 'yolov4.cfg')
video_path = os.path.join(fileDir, 'source.mp4')

# Importando Labels
LABELS = open(labels_path).read().strip().split("\n")

# Criação da Rede Neural
net = cv2.dnn.readNet(cfg_path, weights_path)

# Definindo Camadas de Saída
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(ln)

# Função para Construir o Blob
def blob_imagem(net, imagem, mostrar_texto=True):
    inicio = time.time() 
    blob = cv2.dnn.blobFromImage(imagem, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    termino = time.time()
    if mostrar_texto:
        print("YOLO levou {:.2f} segundos".format(termino - inicio))
    return net, imagem, layerOutputs

# Função para detectar objetos
def deteccoes(detection, threshold, caixas, confiancas, IDclasses):
    scores = detection[5:] 
    classeID = np.argmax(scores)  
    confianca = scores[classeID]

    if confianca > threshold:
        caixa = detection[0:4] * np.array([W, H, W, H])     
        (centerX, centerY, width, height) = caixa.astype("int")
            
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        caixas.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        IDclasses.append(classeID)
      
    return caixas, confiancas, IDclasses

# Sorteando cores para os retângulos
np.random.seed(42)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype="uint8")

# Função para desenhar retângulos
def funcoes_imagem(imagem, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=False):  
    (x, y) = (caixas[i][0], caixas[i][1])
    (w, h) = (caixas[i][2], caixas[i][3])

    cor = [int(c) for c in COLORS[IDclasses[i]]]
    cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2) 
    texto = "{}: {:.4f}".format(LABELS[IDclasses[i]], confiancas[i])
    if mostrar_texto:
        print("> " + texto)
        print(x,y,w,h)
    cv2.putText(imagem, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    return imagem,x,y,w,h

# Configurando Renderização
capture = cv2.VideoCapture(video_path)
conectado, video = capture.read()
threshold = 0.5
nms_threshold = 0.3
fonte_pequena, fonte_media = 0.4, 0.6
font = cv2.FONT_HERSHEY_SIMPLEX
amostras_exibir = 20
amostra_atual = 0

video_largura = video.shape[1]
video_altura = video.shape[0]
video_largura, video_altura

# Redimensionando
def redimensionar(largura, altura, largura_maxima = 600): 
    if (largura > largura_maxima):
        proporcao = largura / altura
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = largura
        video_altura = altura

    return video_largura, video_altura

# Saída
nome_arquivo = 'resultado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID') # MP4V
fps = 24
saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))

# Magia
while (cv2.waitKey(1) < 0):
    conectado, frame = capture.read()
    if not conectado:
        break
    t = time.time()
    frame = cv2.resize(frame, (video_largura, video_altura))
    try:
        (H, W) = frame.shape[:2]
    except:
        print('Erro')
        continue

    imagem_cp = frame.copy() 
    net, frame, layerOutputs = blob_imagem(net, frame)
    caixas = []       
    confiancas = []   
    IDclasses = []    

    for output in layerOutputs:
        for detection in output:
            caixas, confiancas, IDclasses = deteccoes(detection, threshold, caixas, confiancas, IDclasses)

    objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, nms_threshold)

    if len(objs) > 0:
        for i in objs.flatten():
            frame, x, y, w, h = funcoes_imagem(frame, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=True)
            objeto = imagem_cp[y:y + h, x:x + w]
  
    cv2.putText(frame, " frame processado em {:.2f} segundos".format(time.time() - t), (20, video_altura-20), font, fonte_pequena, (250, 250, 250), 0, lineType=cv2.LINE_AA)

    saida_video.write(frame)
    
# Finalização
print('Terminou')
saida_video.release()
    