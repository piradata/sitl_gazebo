import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import rospy
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseStamped

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

import serial

from datetime import datetime

####################### Encoder #################################
try:
    ser = serial.Serial('/dev/ttyACM1', 9600)
except:
    ser = serial.Serial('/dev/ttyACM0', 9600)
#trecho=" Posicao do encoder: "
controle2=False
msg=None
while controle2==False:
    msg=ser.readline()
    if msg is not None:
        try:
            controle2=True
            msg=msg.decode("utf-8")
            yaw_enc=float(msg)
        except:
            controle2=False
            msg=None
            pass


sct = mss.mss()
# Tela a ser capturada
#monitor = {"top": 52, "left": 67, "width": 647, "height": 712}
#monitor = {"top": 54, "left": 68, "width": 600, "height": 555}
monitor = {"top": 54, "left": 155, "width": 519, "height": 390}

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

pos_msg=None
def local_pose_callback(data):
    global pos_msg
    pos_msg=data
    
def PROCESSA_imagem(p_saida):
    yolo = YOLOff()
    objetivo="meianau"
    rospy.init_node('camera_kf', anonymous=True)
    pub = rospy.Publisher('dados_camera_kf', String)#, queue_size=10)
    pub2 = rospy.Publisher('encontra_placa', String)
    rate = rospy.Rate(10) # 10hz
    placa_encontrada=False
    itens=[]
    kp=30
    
    tref=datetime.now()
    
    controle2=False
    msg=None
    while controle2==False:
        msg=ser.readline()
        if msg is not None:
            controle2=True
            try:
                msg=msg.decode("utf-8")
                yaw_enc=float(msg)
            except:
                controle2=False
                msg=None
                pass
    
    old_frame = p_saida.recv()
    # Parametros do detector de vertices Shi-Tomasi
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )  
    # Parametros do fluxo optico de Lucas-Kanade
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Cria cores aleatórias
    color = np.random.randint(0,255,(100,3))
    # Obtem primeiro frame e vertices de interesse
    #ret, old_frame = cap.read()
    #blur = cv2.GaussianBlur(old_frame,(5,5),0)
    #smooth = cv2.addWeighted(blur,1.5,old_frame,-0.5,0)
    #old_frame=smooth.copy()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # cria mascara para desenhar fluxo
    mask = np.zeros_like(old_frame)
    p1_backup=[]
    st_backup=[]
    err_backup=[]
    p0_backup=[]
    yaw_lk=[]
    yaw_lk.append(yaw_enc)
    ti=datetime.now()
    
    deteccoes=[]
    tempo_dec=[]
    yaw_compass=[]
    
    yaw_yoloff=[]
    yaw_yoloff.append(yaw_enc)
    
    yaw_of=[]
    yaw_of.append(yaw_enc)
    
    yaw_encoder=[]
    yaw_encoder.append(yaw_enc)
    

    while True and not rospy.is_shutdown():
        imagem = p_saida.recv()
        frame=imagem.copy()
        #ret,frame = cap.read()
        try:
            #blur = cv2.GaussianBlur(frame,(5,5),0)
            #smooth = cv2.addWeighted(blur,1.3,frame,-0.4,0)
            #frame=smooth.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
        # calcula fluxo optico
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            # cria mascara para desenhar fluxo
            mask = np.zeros_like(old_frame)
            if p0 is None:
                 p0=p0_backup.reshape(-1,1,2)
                 p1 = p1_backup
                 st = st_backup
                 err = err_backup
            else:
                # calcula fluxo optico
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # seleciona pontos para mapear
        good_new = p1[st==1]
        good_old = p0[st==1]
        # desenha os tracados de fluxo optico
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        tf=datetime.now()
        dt=((tf-ti).seconds+(tf-ti).microseconds/10e5)
        ti=tf
        omega=((a-c)/frame.shape[1])/dt
        imgr = img #cv2.resize(img, (960, 540))
        cv2.imshow('Fluxo Optico',imgr)
        # Atualiza frame e pontos anteriores
        old_gray = frame_gray.copy()
        old_frame=frame.copy()
        p0_backup=p0
        p0 = good_new.reshape(-1,1,2)
        p1_backup=p1
        st_backup=st
        err_backup=err
        
        ti_yolo=datetime.now()
        r_imagem, lista_de_objetos = yolo.detecta_imagem(imagem, placa_encontrada)
        print(lista_de_objetos)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, local_pose_callback)
        controle=False
        while controle==False:
            dados_pos=pos_msg
            if pos_msg is not None:
                controle=True
                #px=pos_msg.pose.position.x
                #py=pos_msg.pose.position.y
                #pz=pos_msg.pose.position.z
                qx=pos_msg.pose.orientation.x
                qy=pos_msg.pose.orientation.y
                qz=pos_msg.pose.orientation.z
                qw=pos_msg.pose.orientation.w
                yaw = np.arctan2(2.0*(qy*qx + qw*qz), (qw*qw + qx*qx - qy*qy - qz*qz))*180/np.pi
                controle2=False
                msg=None
                while controle2==False:
                    msg=ser.readline()
                    if msg is not None:
                        controle2=True
                        try:
                            msg=msg.decode("utf-8")
                            yaw_enc=float(msg)
                        except:
                            controle2=False
                            msg=None
                            pass
                nyaw = yaw_of[-1] + yaw_of[-1]*omega*dt
                if abs(nyaw)>=360:
                    nyaw=nyaw-int(nyaw/360)*360
                yaw_lk.append(nyaw)

        for item in lista_de_objetos:
            for i in range(len(item)):
                yaw_yolo = yaw + kp*((item[4]-r_imagem.shape[1]/2)/r_imagem.shape[1]/2)
                item.append(yaw_yolo)
            tnow=datetime.now()
            delta_t=float((tnow-tref).seconds+(tnow-tref).microseconds/10e5)
            item.append(delta_t)
            #item.append(px)
            #item.append(py)
            #item.append(pz)
            item.append(yaw)
            item.append(yaw_enc)
            item.append(yaw_lk[-1])
            itens.append(item)
        prob=0.0
        ROI=[]
        for obj in lista_de_objetos:
            if obj[6]=='placa':
                detectado=obj[6]
                if placa_encontrada==False:
                    placa_encontrada=True
                    yolo.fecha_sessao()
                ROI=obj
                rospy.loginfo('Placa Encontrada!')
                pub2.publish('Placa Encontrada!')
            elif obj[6]==objetivo:
                detectado=obj[6]
                if np.float(obj[8])>=prob:
                    prob=np.float(obj[8])
                    ROI=obj
        if ROI==[]:
            for obj in lista_de_objetos:
                if obj[6]=="navio":
                    detectado=obj[6]
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
            detectado="vazio"
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
        tf_yolo=datetime.now()
        dt_yolo=((tf_yolo-ti_yolo).seconds+(tf_yolo-ti_yolo).microseconds/10e5)
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
            
        tnow=datetime.now()
        delta_t=float((tnow-tref).seconds+(tnow-tref).microseconds/10e5)
        
        #yaw_yolo = yaw + kp*((predp[0][0]-r_imagem.shape[1]/2)/r_imagem.shape[1]/2)
        try:
            omega_yolo=((predp[0][0]-predp[1][0])/r_imagem.shape[1])/dt_yolo
        except:
            omega_yolo=0
        yaw_yolo = yaw_yoloff[-1] + yaw_yoloff[-1]*omega_yolo*dt_yolo

        
        controle2=False
        msg=None
        while controle2==False:
            msg=ser.readline()
            if msg is not None:
                controle2=True
                try:
                    msg=msg.decode("utf-8")
                    yaw_enc=float(msg)
                except:
                    controle2=False
                    msg=None
            
        
        deteccoes.append(detectado)
        tempo_dec.append(delta_t)
        yaw_compass.append(yaw)
        yaw_yoloff.append(yaw_yolo)
        yaw_of.append(yaw_lk[-1])
        yaw_encoder.append(yaw_enc)
            
        cv2.imshow("Imagem YOLOff", r_imagem)
        centrokalman=[r_imagem.shape[1], r_imagem.shape[0], int(predp[0][0]), int(predp[0][1])]
        dados_cam = str(centrokalman).strip('[]')
        rospy.loginfo(dados_cam)
        pub.publish(dados_cam)
        #rate.sleep()
        
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
             cv2.destroyAllWindows()
             df=pd.DataFrame(itens)
             df.to_csv('dados_camera_validacao_completo.csv', index=False, header=False)
             
             arq={'deteccao': deteccoes,
                      'tempo': tempo_dec,
                      'yaw': yaw_compass,
                      'yaw_yoloff': yaw_yoloff.pop(0),
                      'yaw_lk': yaw_of.pop(0),
                      'yaw_enc': yaw_encoder.pop(0)}
             arq=pd.DataFrame(data=arq)
             arq.to_csv('dados_camera_validacao_completo2.csv', index=False)
             
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