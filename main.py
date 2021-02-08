import cv2
import numpy as np
import os
import time

# Configurações iniciais
fileDir = os.path.dirname(os.path.realpath('__file__'))
labels_path = os.path.join(fileDir, 'coco.names')
weights_path = os.path.join(fileDir, 'yolov4.weights')
cfg_path = os.path.join(fileDir, 'yolov4.cfg')
video_path = os.path.join(fileDir, 'source.mp4')
threshold = 0.35
nms_threshold = 0.1

# Importando Labels
LABELS = open(labels_path).read().strip().split("\n")

# Criação da Rede Neural
net = cv2.dnn.readNet(cfg_path, weights_path)

# Definindo Camadas de Saída
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Configurando Renderização
fonte_pequena, fonte_media = 0.4, 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Região de Interesse
x1, y1, x2, y2 = 650, 450, 820, 550

# Função para Construir o Blob
def blob_imagem(net):
    blob = cv2.dnn.blobFromImage(interest, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    return net, layerOutputs

# Função para detectar objetos
def deteccoes(detection, threshold, caixas, confiancas, IDclasses):
    scores = detection[5:] 
    classeID = np.argmax(scores)  
    confianca = scores[classeID]

    #Checa se o objeto detectado é válido de acordo com a margem de erro
    if confianca > threshold:
        caixa = detection[0:4] * np.array([W, H, W, H])     
        (centerX, centerY, width, height) = caixa.astype("int")
            
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        caixas.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        IDclasses.append(classeID)
      
    return caixas, confiancas, IDclasses

# Função para desenhar retângulos nos objetos
def funcoes_imagem(imagem, i, confiancas, caixas, LABELS):  
    (x, y) = (caixas[i][0], caixas[i][1])
    (w, h) = (caixas[i][2], caixas[i][3])
    xx = int(x/8 + x1)
    yy = int(y/8 + y1)
    xx2 = int((x+w)/8 + x1)
    yy2 = int((y+h)/8 + y1)
    cv2.rectangle(imagem, (xx, yy), (xx2 , yy2), [255, 255, 0], 2)
    texto = "{}: {:.4f}".format(LABELS[IDclasses[i]], confiancas[i])
    cv2.putText(imagem, texto, (xx, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 0], 2)
    objeto = LABELS[IDclasses[i]]
    return imagem,objeto,x,y,w,h

# Saída
capture = cv2.VideoCapture(video_path)
conectado, video = capture.read()

nome_arquivo = 'resultado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID') # MP4V
fps = 24
video_altura = video.shape[0]
video_largura = video.shape[1]
saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))

# Função principal
#Contadores
counter = False
carros = 0
motos = 0

#Captador de quadros (frames) do vídeo
while (cv2.waitKey(1) < 0):
    #Checa se vídeo foi carregado corretamente ou se já terminou
    conectado, frame = capture.read()
    if not conectado:
        break
    try:
        (H, W) = frame.shape[:2]
    except:
        print('Erro')
        continue
    
    #Define Região de Interesse
    interest = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), [255, 255, 255], 2)

    #Faz o Blob da imagem
    net, layerOutputs = blob_imagem(net)

    #Inicialização de variáveis
    caixas = []       
    confiancas = []   
    IDclasses = []

    #Classificador de objetos detectados
    for output in layerOutputs:
        for detection in output:
            caixas, confiancas, IDclasses = deteccoes(detection, threshold, caixas, confiancas, IDclasses)

    objs = cv2.dnn.NMSBoxes(caixas, confiancas, threshold, nms_threshold)

    if len(objs) == 0:
            counter = False

    if len(objs) > 0:
        for i in objs.flatten():
            frame, objeto, x, y, w, h = funcoes_imagem(frame, i, confiancas, caixas, LABELS)
            #Contador de Carros/Motos
            if counter == False:
                if objeto == 'car' or 'truck':
                    carros += 1
                    counter = True
                if objeto == 'motorbike':
                    motos += 1
                    counter = True
    
    #Escreve contador na tela
    total = "Carros: " + str(carros)
    total_2 = "Motos: " + str(motos)
    cv2.putText(frame, total, (20, video_altura-20), font, 1, (250, 250, 250), 0, lineType=cv2.LINE_AA)
    cv2.putText(frame, total_2, (20, video_altura-50), font, 1, (250, 250, 250), 0, lineType=cv2.LINE_AA)

    #Escreve o quadro modificado no arquivo de saída
    saida_video.write(frame)
    
# Finalização e liberação de memória
print('Terminou')
saida_video.release()
    
