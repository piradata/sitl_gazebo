import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import rospy
from std_msgs.msg import String, Float64

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from pyzbar import pyzbar

import sys
#necessario para que o sistema não tente importar o opencv do ROS
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

#from matplotlib import pyplot as plt
#import pandas as pd

import multiprocessing
from multiprocessing import Pipe

from collections import deque
import argparse
import mss


sct = mss.mss()
# Tela a ser capturada
#monitor = {"top": 52, "left": 67, "width": 647, "height": 712}
monitor = {"top": 54, "left": 68, "width": 600, "height": 555}

####################### Kalman #################################

meas=[]
pred=[]
mp = np.array((2,1), np.float32) # medidas
tp = np.zeros((2,1), np.float32) # posição prevista
kalman = cv2.KalmanFilter(4,2) # inicializa filtro de Kalman
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32) # Matriz de variaveis observaveis
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32) # Matriz de transição
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03 # Ruido de medição

# constroi a divisao de argumentos e divide os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
        help="caminho para o arquivo de video (opcional)")
ap.add_argument("-b", "--buffer", type=int, default=64,
        help="tamanho maximo do buffer")
args = vars(ap.parse_args())

#pontos do filtro de Kalman
predp = deque(maxlen=args["buffer"])
pontos = deque(maxlen=args["buffer"])


class YOLOff(object):
    _defaults = {
        #"caminho_modelo": '/home/gabryelsr/Pictures/databank/YOLOff_treino2trained_weights_final.h5',
        "caminho_modelo": '/home/gabryelsr/Pictures/databank/YOLOff_treino2_naviotrained_weights_final.h5',
        "caminho_ancoras": '/home/gabryelsr/Pictures/databank/YOLO/model_data/yolo_anchors.txt',
        #"caminho_classes": '/home/gabryelsr/Pictures/databank/YOLO/objetos_treino2.txt',
        "caminho_classes": '/home/gabryelsr/Pictures/databank/YOLO/objetos_treino_modelo.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "tam_img_modelo" : (416, 416),
        "tam_texto" : 0.10,
    }

    @classmethod
    def pega_def(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Atributo nao reconhecido: '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # configura valores
        self.__dict__.update(kwargs) # atualiza com novos parametros do usuario
        self.nomes_classes = self._pega_classe()
        self.ancoras = self._pega_ancoras()
        self.sessao = K.get_session()
        self.caixas, self.scores, self.classes = self.gera()

    def _pega_classe(self):
        caminho_classes = os.path.expanduser(self.caminho_classes)
        with open(caminho_classes) as f:
            nomes_classes = f.readlines()
        nomes_classes = [c.strip() for c in nomes_classes]
        return nomes_classes

    def _pega_ancoras(self):
        caminho_ancoras = os.path.expanduser(self.caminho_ancoras)
        with open(caminho_ancoras) as f:
            ancoras = f.readline()
        ancoras = [float(x) for x in ancoras.split(',')]
        return np.array(ancoras).reshape(-1, 2)

    def gera(self):
        caminho_modelo = os.path.expanduser(self.caminho_modelo)
        assert caminho_modelo.endswith('.h5'), 'Os pesos do modelo em Keras precisam ser um arquivo .h5'

        # Carrega modelo ou coonstroi o modelo e carrega os pesos
        num_ancoras = len(self.ancoras)
        num_classes = len(self.nomes_classes)
        versao_pequena = num_ancoras==6 # Configuracao padrao
        try:
            self.modelo_yolo = load_model(caminho_modelo, compile=False)
        except:
            self.modelo_yolo = tiny_yolo_body(Input(shape=(None,None,3)), num_ancoras//2, num_classes) \
                if versao_pequena else yolo_body(Input(shape=(None,None,3)), num_ancoras//3, num_classes)
            self.modelo_yolo.load_weights(self.caminho_modelo) # necessario que o modelo, ancoras e classes estejam de acordo
        else:
            assert self.modelo_yolo.layers[-1].output_shape[-1] == \
                num_ancoras/len(self.modelo_yolo.output) * (num_classes + 5), \
                'Desacordo entre modelo e tamanho de ancoras e classes'

        print('{} modelo, ancoras, e classes carregados'.format(caminho_modelo))

        # Gera cores para as caixas
        tuple_hsv = [(x / len(self.nomes_classes), 1., 1.)
                      for x in range(len(self.nomes_classes))]
        self.cores = list(map(lambda x: colorsys.hsv_to_rgb(*x), tuple_hsv))
        self.cores = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.cores))

        np.random.shuffle(self.cores)  # Embaralha cores para nao confundir classes adjacentes.

        # Gera alvos do tensor de saida para caixas de contorno filtradas
        self.tam_img_entrada = K.placeholder(shape=(2, ))
        caixas, scores, classes = yolo_eval(self.modelo_yolo.output, self.ancoras,
                len(self.nomes_classes), self.tam_img_entrada,
                score_threshold=self.score, iou_threshold=self.iou)
        return caixas, scores, classes

    def detecta_img(self, imagem):
        if self.tam_img_modelo != (None, None):
            assert self.tam_img_modelo[0]%32 == 0, 'Requer multiplos de 32'
            assert self.tam_img_modelo[1]%32 == 0, 'Requer multiplos de 32'
            img_contornada = image_preporcess(np.copy(imagem), tuple(reversed(self.tam_img_modelo)))
            dados_imagemm = img_contornada

        caixas_saida, scores_saida, classes_saida = self.sessao.run(
            [self.caixas, self.scores, self.classes],
            feed_dict={
                self.modelo_yolo.input: dados_imagemm,
                self.tam_img_entrada: [imagem.shape[0], imagem.shape[1]],#[imagem.size[1], imagem.size[0]],
                K.learning_phase(): 0
            })

        #print('Encontradas {} caixas para {}'.format(len(caixas_saida), 'img'))

        espessura = (imagem.shape[0] + imagem.shape[1]) // 600
        tam_fonte=0.2
        lista_de_objetos = []
        
        for i, c in reversed(list(enumerate(classes_saida))):
            classe_prevista = self.nomes_classes[c]
            caixa = caixas_saida[i]
            score = scores_saida[i]

            rotulo = '{} {:.2f}'.format(classe_prevista, score)
            #rotulo = '{}'.format(classe_prevista)
            scores = '{:.2f}'.format(score)

            cima, esquerda, baixo, direita = caixa
            cima = max(0, np.floor(cima + 0.5).astype('int32'))
            esquerda = max(0, np.floor(esquerda + 0.5).astype('int32'))
            baixo = min(imagem.shape[0], np.floor(baixo + 0.5).astype('int32'))
            direita = min(imagem.shape[1], np.floor(direita + 0.5).astype('int32'))

            meio_v = (baixo-cima)/2+cima
            meio_h = (direita-esquerda)/2+esquerda

            # Coloca o retangulo no objeto itentificado
            cv2.rectangle(imagem, (esquerda, cima), (direita, baixo), self.cores[c], espessura)

            # Obtem tamanho do texto
            (largura_texto, altura_texto), linha_base = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, tam_fonte, 1)

            # Coloca o retangulo do texto logo acima do objeto
            cv2.rectangle(imagem, (esquerda, cima), (esquerda + largura_texto, cima - altura_texto - linha_base), self.cores[c], thickness=cv2.FILLED)

            # Coloca texto sobre o retangulo
            cv2.putText(imagem, rotulo, (esquerda, cima-2), cv2.FONT_HERSHEY_SIMPLEX, tam_fonte, (0, 0, 0), 1)

            # Adiciona tudo na uma lista
            lista_de_objetos.append([cima, esquerda, baixo, direita, meio_h, meio_v, classe_prevista, rotulo, scores])

        return imagem, lista_de_objetos

    def fecha_sessao(self):
        self.sessao.close()

    def detecta_imagem(self, imagem, placa_encontrada):
        #imagem = cv2.imread(imagem, cv2.IMREAD_COLOR)
        imagem_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        cor_imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        
        placaqr = pyzbar.decode(cor_imagem_original)
        
        if placaqr != []:
            lista_de_objetos=[]
            for placa in placaqr:
                    (x, y, w, h) = placa.rect
                    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    dadoPlaca = placa.data.decode("utf-8")
                    tipoPlaca = placa.type
                    texto = "{} ({})".format(dadoPlaca, tipoPlaca)
                    r_imagem = cv2.putText(imagem, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    print("[INFO] Encontrado {} placa: {}".format(tipoPlaca, dadoPlaca))
                    lista_de_objetos.append([y, x, y+h, x+w, (x+w)/2, (y+h)/2, 'placa', texto, '1'])
        else:
            if placa_encontrada==False:
                r_imagem, lista_de_objetos = self.detecta_img(cor_imagem_original)
            else:
                r_imagem=cor_imagem_original
                lista_de_objetos=[]
        return r_imagem, lista_de_objetos

def INCLUI_imagem(p_entrada):
    #Abre a imagem
    for n in range(1217):
        imagem = str('/home/gabryelsr/Pictures/databank/tankers/image'+str(n)+'.png')
        img1 = cv2.imread(imagem, cv2.IMREAD_COLOR)
        img2 = np.array(img1)
        # Coloca imagem no pipe
        p_entrada.send(img2)
        
def PEGA_tela(p_entrada):
    while True:
        #Captura imagem da tela
        imagem = np.array(sct.grab(monitor))
        #Coloca imagem no pipe
        p_entrada.send(imagem)
    
def PROCESSA_imagem(p_saida):
    yolo = YOLOff()
    objetivo="meianau"
    rospy.init_node('camera_kf', anonymous=True)
    pub = rospy.Publisher('dados_camera_kf', String)#, queue_size=10)
    pub2 = rospy.Publisher('encontra_placa', String)
    rate = rospy.Rate(10) # 10hz
    placa_encontrada=False
    itens=[]
    while True and not rospy.is_shutdown():
        imagem = p_saida.recv()
        r_imagem, lista_de_objetos = yolo.detecta_imagem(imagem, placa_encontrada)
        print(lista_de_objetos)
        for item in lista_de_objetos:
            item.append(rospy.get_time())
            itens.append(item)
        prob=0.0
        ROI=[]
        for obj in lista_de_objetos:
            if obj[6]=='placa':
                if placa_encontrada==False:
                    placa_encontrada=True
                    yolo.fecha_sessao()
                ROI=obj
                rospy.loginfo('Placa Encontrada!')
                pub2.publish('Placa Encontrada!')
            elif obj[6]==objetivo:
                if np.float(obj[8])>=prob:
                    prob=np.float(obj[8])
                    ROI=obj
        if ROI==[]:
            for obj in lista_de_objetos:
                if obj[6]=="navio":
                    if np.float(obj[8])>=prob:
                        prob=np.float(obj[8])
                        if objetivo=="proa":
                            ROI=obj
                            ROI[4]=ROI[4]+int(ROI[4]/3)
                        elif objetivo=="meianau":
                            ROI=obj
                        elif objetivo=="popa":
                            ROI=obj
                            ROI[4]=ROI[4]-int(ROI[4]/3)
        if ROI==[]:
            if pontos:
                x=int(pontos[0][0])
                y=int(pontos[0][1])
            else:
                x=int(r_imagem.shape[1]/2)
                y=int(r_imagem.shape[0]/2)
        else:
            x=int(np.float(ROI[4]))
            y=int(np.float(ROI[5]))
            
        centro=(x,y)
        # Atualiza a medida para previsão com filtro de Kalman
        mp = np.array([[np.float32(x)],[np.float32(y)]])
        meas.append((x,y))
        # atualiza o queue de pontos
        pontos.appendleft(centro)
        # Filtro de Kalman
        kalman.correct(mp)
        tp = kalman.predict()
        predp.appendleft((int(np.float(tp[0])),int(np.float(tp[1]))))
        # percorre os pontos encontrados
        for i in range(1, len(pontos)):
                # se algum dos pontos for None (vazio), ignora-os
                if pontos[i - 1] is None or pontos[i] is None:
                        continue
                # senão, calcula a espessura da linha e desenha as linhas       
                #espessura = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                #cv2.line(tela, pontos[i - 1], pontos[i], (0, 255, 255), espessura)
                cv2.line(r_imagem, pontos[i - 1], pontos[i], (0, 255, 255),2)
                #desenha linha do filtro de Kalman (posições futuras previstas)
                cv2.line(r_imagem,predp[i-1],predp[i],(0,0,255),2)
        #r_imagem=cv2.cvtColor(r_imagem, cv2.COLOR_BGR2RGB)
        #plt.imshow(r_imagem)
        #plt.show()
        if int(np.float(tp[0]))>int(r_imagem.shape[1]/2):
            position = ((int) (2*r_imagem.shape[1]/3), (int) (2*r_imagem.shape[0]/3))
            cv2.putText(
                 r_imagem,
                 ">>>",
                 position,
                 cv2.FONT_HERSHEY_SIMPLEX,
                 2,
                 (209, 80, 0, 255),
                 3)
        elif int(np.float(tp[0]))<int(r_imagem.shape[1]/2):
            position = ((int) (r_imagem.shape[1]/3), (int) (2*r_imagem.shape[0]/3))
            cv2.putText(
                 r_imagem,
                 "<<<",
                 position,
                 cv2.FONT_HERSHEY_SIMPLEX,
                 2,
                 (209, 80, 0, 255),
                 3)
        else:
            position = ((int) (r_imagem.shape[1]/2), (int) (2*r_imagem.shape[0]/3))
            cv2.putText(
                 r_imagem,
                 "^",
                 position,
                 cv2.FONT_HERSHEY_SIMPLEX,
                 2,
                 (209, 80, 0, 255),
                 3)
            
        cv2.imshow("Imagem YOLOff", r_imagem)
        centrokalman=[r_imagem.shape[1], r_imagem.shape[0], int(predp[0][0]), int(predp[0][1])]
        dados_cam = str(centrokalman).strip('[]')
        rospy.loginfo(dados_cam)
        pub.publish(dados_cam)
        #rate.sleep()
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
             cv2.destroyAllWindows()
             df=pd.DataFrame(itens)
             df.to_csv('dados_camera.csv', index=False, header=False)
             break
    yolo.fecha_sessao()
    
if __name__=="__main__":
    p_saida, p_entrada = Pipe()

    # cria novo processo
    p1 = multiprocessing.Process(target=PEGA_tela, args=(p_entrada,))
    p2 = multiprocessing.Process(target=PROCESSA_imagem, args=(p_saida,))

    # inicia processos
    p1.start()
    p2.start()