import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
#necessario para que o sistema não tente importar o opencv do ROS
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

from matplotlib import pyplot as plt

import multiprocessing
from multiprocessing import Pipe

class YOLOff(object):
    _defaults = {
        "caminho_modelo": '/home/gabryelsr/Pictures/databank/YOLOff_treino2trained_weights_final.h5',
        "caminho_ancoras": '/home/gabryelsr/Pictures/databank/YOLO/model_data/yolo_anchors.txt',
        "caminho_classes": '/home/gabryelsr/Pictures/databank/YOLO/objetos_treino2.txt',
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
            assert self.tam_img_modelo[0]%32 == 0, 'Multiples of 32 required'
            assert self.tam_img_modelo[1]%32 == 0, 'Multiples of 32 required'
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

            meio_h = (baixo-cima)/2+cima
            meio_v = (direita-esquerda)/2+esquerda

            # Coloca o retangulo no objeto itentificado
            cv2.rectangle(imagem, (esquerda, cima), (direita, baixo), self.cores[c], espessura)

            # Obtem tamanho do texto
            (largura_texto, altura_texto), linha_base = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, tam_fonte, 1)

            # Coloca o retangulo do texto logo acima do objeto
            cv2.rectangle(imagem, (esquerda, cima), (esquerda + largura_texto, cima - altura_texto - linha_base), self.cores[c], thickness=cv2.FILLED)

            # Coloca texto sobre o retangulo
            cv2.putText(imagem, rotulo, (esquerda, cima-2), cv2.FONT_HERSHEY_SIMPLEX, tam_fonte, (0, 0, 0), 1)

            # Adiciona tudo na uma lista
            lista_de_objetos.append([cima, esquerda, baixo, direita, meio_v, meio_h, classe_prevista, rotulo, scores])

        return imagem, lista_de_objetos

    def fecha_sessao(self):
        self.sessao.close()

    def detecta_imagem(self, imagem):
        #imagem = cv2.imread(imagem, cv2.IMREAD_COLOR)
        imagem_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        cor_imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)
        
        r_imagem, lista_de_objetos = self.detecta_img(cor_imagem_original)
        return r_imagem, lista_de_objetos

def INCLUI_imagem(p_entrada):
    #Abre a imagem
    for n in range(1217):
        imagem = str('/home/gabryelsr/Pictures/databank/tankers/image'+str(n)+'.png')
        img1 = cv2.imread(imagem, cv2.IMREAD_COLOR)
        img2 = np.array(img1)
        # Coloca imagem no pipe
        p_entrada.send(img2)
    
def PROCESSA_imagem(p_saida):
    yolo = YOLOff()
    while True:
        imagem = p_saida.recv()
        r_imagem, lista_de_objetos = yolo.detecta_imagem(imagem)
        print(lista_de_objetos)
        r_imagem=cv2.cvtColor(r_imagem, cv2.COLOR_BGR2RGB)
        #plt.rcParams['figure.dpi'] = 360
        plt.imshow(r_imagem)
        plt.show()
        #cv2.imshow("Imagem YOLOff", r_imagem)
        if cv2.waitKey(25) & 0xFF == ord("q"):
             cv2.destroyAllWindows()
             break
    yolo.fecha_sessao()
    
if __name__=="__main__":
    p_saida, p_entrada = Pipe()

    # cria novo processo
    p1 = multiprocessing.Process(target=INCLUI_imagem, args=(p_entrada,))
    p2 = multiprocessing.Process(target=PROCESSA_imagem, args=(p_saida,))

    # inicia processos
    p1.start()
    p2.start()