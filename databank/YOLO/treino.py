import os
#Configura se roda na GPU (0,1,2,...) ou processador (-1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

#Modelos da rede YOLO3 (estado da arte)
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    #caminho_anotacao = 'objetos_anotados_treino2.txt'
    caminho_anotacao = 'objetos_anotados_treino_modelo.txt'
    #caminho_log = '/home/gabryelsr/Pictures/databank/YOLOff_treino2'
    caminho_log = '/home/gabryelsr/Pictures/databank/YOLOff_treino2_navio'
    #caminho_classes = 'objetos_treino2.txt'
    caminho_classes = 'objetos_treino_modelo.txt'
    #caminho_ancoras = 'model_data/yolo_anchors.txt'
    caminho_ancoras = 'model_data/yolo_anchors.txt'
    nomes_classes = obtem_classes(caminho_classes)
    num_classes = len(nomes_classes)
    ancoras = obtem_ancoras(caminho_ancoras)

    formato_entrada = (416,416) # multiplo de 32, altura/largura

    verifica_versao_compacta = len(ancoras)==6 # configuracao padrao
    if verifica_versao_compacta:
        model = cria_modelinho(formato_entrada, ancoras, num_classes,
            congela_corpo=2, caminho_pesos='model_data/tiny_yolo_weights.h5')
    else:
        model = cria_modelo(formato_entrada, ancoras, num_classes, congela_corpo=2, caminho_pesos='model_data/yolo_weights.h5')


    loga = TensorBoard(log_dir=caminho_log)
    checkpoint = ModelCheckpoint(caminho_log + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduz_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    parada_antecipada = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    divide_valor = 0.1
    with open(caminho_anotacao) as f:
        linhas = f.readlines()
    np.random.shuffle(linhas)
    num_valor = int(len(linhas)*divide_valor)
    num_treino = len(linhas) - num_valor

    # Treina com camadas congeladas primeiro para obter loss estavel.
    # Ajusta numero de epocas para o dataset (suficiente para estabilizar modelo)
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # usa camada Lambda yolo_loss customizada.
            'yolo_loss': lambda y_verdadeiro, y_previsto: y_previsto})

        tamanho_lote = 16
        print('Treino em {} amostras, validacao em {} amostras, com tamanho de lote {}'.format(num_treino, num_valor, tamanho_lote))
        model.fit_generator(agregador_gerador_dados(linhas[:num_treino], tamanho_lote, formato_entrada, ancoras, num_classes),
                steps_per_epoch=max(1, num_treino//tamanho_lote),
                validation_data=agregador_gerador_dados(linhas[num_treino:], tamanho_lote, formato_entrada, ancoras, num_classes),
                validation_steps=max(1, num_valor//tamanho_lote),
                epochs=100,
                initial_epoch=0,
                callbacks=[loga, checkpoint])
        model.save_weights(caminho_log + 'trained_weights_stage_1.h5')

    # Descongela e continua o treinamento para melhorar precisao
    # Treina por mais tempo se o resultado nao for bom
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_verdadeiro, y_previsto: y_previsto}) # recompila e aplica a mudança
        print('Descongelar todas as camadas')

        tamanho_lote = 8 # Maior memoria de GPU é necessaria depois de descongelar o corpo da rede
        print('Treino em {} amostras, validacao em {} amostras, com tamanho de lote {}'.format(num_treino, num_valor, tamanho_lote))
        model.fit_generator(agregador_gerador_dados(linhas[:num_treino], tamanho_lote, formato_entrada, ancoras, num_classes),
            steps_per_epoch=max(1, num_treino//tamanho_lote),
            validation_data=agregador_gerador_dados(linhas[num_treino:], tamanho_lote, formato_entrada, ancoras, num_classes),
            validation_steps=max(1, num_valor//tamanho_lote),
            epochs=100,
            initial_epoch=50,
            callbacks=[loga, checkpoint, reduz_lr, parada_antecipada])
        model.save_weights(caminho_log + 'trained_weights_final.h5')

    # Treinamento futuro, se necessario.


def obtem_classes(caminho_classes):
    # Carrega as classes
    with open(caminho_classes) as f:
        nomes_classes = f.readlines()
    nomes_classes = [c.strip() for c in nomes_classes]
    return nomes_classes

def obtem_ancoras(caminho_ancoras):
    # Carrega ancoras de um arquivo
    with open(caminho_ancoras) as f:
        ancoras = f.readline()
    ancoras = [float(x) for x in ancoras.split(',')]
    return np.array(ancoras).reshape(-1, 2)


def cria_modelo(formato_entrada, ancoras, num_classes, carrega_pretreinado=True, congela_corpo=2,
            caminho_pesos='model_data/yolo_weights.h5'):
    # Cria modelo de treinamento
    K.clear_session() # Inicia nova sessao
    imagem_entrada = Input(shape=(None, None, 3))
    h, w = formato_entrada
    num_ancoras = len(ancoras)

    y_verdadeiro = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_ancoras//3, num_classes+5)) for l in range(3)]

    corpo_modelo = yolo_body(imagem_entrada, num_ancoras//3, num_classes)
    print('Cria modelo YOLOv3 com {} ancoras e {} classes.'.format(num_ancoras, num_classes))

    if carrega_pretreinado:
        corpo_modelo.load_weights(caminho_pesos, by_name=True, skip_mismatch=True)
        print('Carrega pesos {}.'.format(caminho_pesos))
        if congela_corpo in [1, 2]:
            # Congela o corpo da darknet53 ou congela tudo exceto as 3 camadas de saida.
            num = (185, len(corpo_modelo.layers)-3)[congela_corpo-1]
            for i in range(num): corpo_modelo.layers[i].trainable = False
            print('Congelas as primeiras {} camadas de um total de {} camadas'.format(num, len(corpo_modelo.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': ancoras, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*corpo_modelo.output, *y_verdadeiro])
    model = Model([corpo_modelo.input, *y_verdadeiro], model_loss)

    return model

def cria_modelinho(formato_entrada, ancoras, num_classes, carrega_pretreinado=True, congela_corpo=2,
            caminho_pesos='model_data/tiny_yolo_weights.h5'):
    #Cria modelo de treinamento para Tiny YOLOv3
    K.clear_session() # Inicia nova sessao
    imagem_entrada = Input(shape=(None, None, 3))
    h, w = formato_entrada
    num_ancoras = len(ancoras)

    y_verdadeiro = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_ancoras//2, num_classes+5)) for l in range(2)]

    corpo_modelo = tiny_yolo_body(imagem_entrada, num_ancoras//2, num_classes)
    print('Cria modelo Tiny YOLOv3 com {} ancoras e {} classes'.format(num_ancoras, num_classes))

    if carrega_pretreinado:
        corpo_modelo.load_weights(caminho_pesos, by_name=True, skip_mismatch=True)
        print('Carrega pesos {}.'.format(caminho_pesos))
        if congela_corpo in [1, 2]:
            # Congela o corpo da darknet ou congela tudo exceto as 3 camadas de saida.
            num = (20, len(corpo_modelo.layers)-2)[congela_corpo-1]
            for i in range(num): corpo_modelo.layers[i].trainable = False
            print('Congela as primeiras {} camadas de um total de {} camadas.'.format(num, len(corpo_modelo.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': ancoras, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*corpo_modelo.output, *y_verdadeiro])
    model = Model([corpo_modelo.input, *y_verdadeiro], model_loss)

    return model

def gerador_dados(linhas_anotacao, tamanho_lote, formato_entrada, ancoras, num_classes):
    # Gerador de dados
    n = len(linhas_anotacao)
    i = 0
    while True:
        dados_imagem = []
        dados_box = []
        for b in range(tamanho_lote):
            if i==0:
                np.random.shuffle(linhas_anotacao)
            imagem, box = get_random_data(linhas_anotacao[i], formato_entrada, random=True)
            dados_imagem.append(imagem)
            dados_box.append(box)
            i = (i+1) % n
        dados_imagem = np.array(dados_imagem)
        dados_box = np.array(dados_box)
        y_verdadeiro = preprocess_true_boxes(dados_box, formato_entrada, ancoras, num_classes)
        yield [dados_imagem, *y_verdadeiro], np.zeros(tamanho_lote)

def agregador_gerador_dados(linhas_anotacao, tamanho_lote, formato_entrada, ancoras, num_classes):
    n = len(linhas_anotacao)
    if n==0 or tamanho_lote<=0: return None
    return gerador_dados(linhas_anotacao, tamanho_lote, formato_entrada, ancoras, num_classes)

if __name__ == '__main__':
    _main()
