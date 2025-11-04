import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Garante que o diretório para salvar as imagens exista
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

def criarGerador():
    """
    Cria o modelo do Gerador.
    Recebe um vetor de ruído (100,) e gera uma imagem (28, 28, 1).
    """
    modelo = tf.keras.Sequential(name="Gerador")
    modelo.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.LeakyReLU())

    modelo.add(layers.Reshape((7, 7, 256)))

    modelo.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.LeakyReLU())

    modelo.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    modelo.add(layers.BatchNormalization())
    modelo.add(layers.LeakyReLU())

    modelo.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return modelo

def criarDiscriminador():
    """
    Cria o modelo do Discriminador.
    Recebe uma imagem (28, 28, 1) e a classifica como real ou falsa.
    """
    modelo = tf.keras.Sequential(name="Discriminador")
    modelo.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    modelo.add(layers.LeakyReLU())
    modelo.add(layers.Dropout(0.3))

    modelo.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    modelo.add(layers.LeakyReLU())
    modelo.add(layers.Dropout(0.3))

    modelo.add(layers.Flatten())
    modelo.add(layers.Dense(1)) # Sem ativação, pois a loss function espera logits

    return modelo

def calcularLossDiscriminador(outputReal, outputFalso):
    """Calcula a perda do discriminador."""
    lossReal = crossEntropy(tf.ones_like(outputReal), outputReal)
    lossFalso = crossEntropy(tf.zeros_like(outputFalso), outputFalso)
    lossTotal = lossReal + lossFalso
    return lossTotal

def calcularLossGerador(outputFalso):
    """Calcula a perda do gerador, visando enganar o discriminador."""
    return crossEntropy(tf.ones_like(outputFalso), outputFalso)

@tf.function
def executarPassoDeTreino(imagens):
    """Executa um único passo de treinamento para o gerador e o discriminador."""
    ruido = tf.random.normal([TAMANHO_BATCH, dimensaoRuido])

    with tf.GradientTape() as tapeGerador, tf.GradientTape() as tapeDiscriminador:
        imagensGeradas = gerador(ruido, training=True)

        outputReal = discriminador(imagens, training=True)
        outputFalso = discriminador(imagensGeradas, training=True)

        lossGerador = calcularLossGerador(outputFalso)
        lossDiscriminador = calcularLossDiscriminador(outputReal, outputFalso)

    gradientesGerador = tapeGerador.gradient(lossGerador, gerador.trainable_variables)
    gradientesDiscriminador = tapeDiscriminador.gradient(lossDiscriminador, discriminador.trainable_variables)

    otimizadorGerador.apply_gradients(zip(gradientesGerador, gerador.trainable_variables))
    otimizadorDiscriminador.apply_gradients(zip(gradientesDiscriminador, discriminador.trainable_variables))

def treinarModelo(dataset, epochs):
    """Função principal de treinamento."""
    for epoch in range(epochs):
        inicio = time.time()

        for batchImagens in dataset:
            executarPassoDeTreino(batchImagens)

        # Produz e salva imagens a cada 50 epochs
        if (epoch + 1) % 50 == 0:
            gerarESalvarImagens(gerador, epoch + 1, seed)
            print(f'Epoch {epoch + 1} concluída. Imagens salvas.')

        print(f'Tempo para epoch {epoch + 1} é {time.time()-inicio:.2f} seg')

    # Gera imagens no final do treinamento
    gerarESalvarImagens(gerador, epochs, seed)
    print("Treinamento concluído.")

def gerarESalvarImagens(modelo, epoch, inputTeste):
    """Gera imagens a partir do ruído e as salva em um arquivo."""
    predicoes = modelo(inputTeste, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predicoes.shape[0]):
        plt.subplot(4, 4, i+1)
        # Desnormaliza a imagem de [-1, 1] para [0, 1]
        plt.imshow(predicoes[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'./gan_images/image_at_epoch_{epoch:04d}.png')
    plt.close()


# --- Configuração e Execução ---

# Carregar e preparar o dataset MNIST
(imagensTreino, _), (_, _) = tf.keras.datasets.mnist.load_data()
imagensTreino = imagensTreino.reshape(imagensTreino.shape[0], 28, 28, 1).astype('float32')
# Normalizar as imagens para o intervalo [-1, 1]
imagensTreino = (imagensTreino - 127.5) / 127.5

BUFFER_SIZE = 60000
TAMANHO_BATCH = 256

# Criar os batches de dados
datasetTreino = tf.data.Dataset.from_tensor_slices(imagensTreino).shuffle(BUFFER_SIZE).batch(TAMANHO_BATCH)

# Criar os modelos
gerador = criarGerador()
discriminador = criarDiscriminador()

# Otimizadores
otimizadorGerador = tf.keras.optimizers.Adam(1e-4)
otimizadorDiscriminador = tf.keras.optimizers.Adam(1e-4)

# Função de perda (Binary Cross Entropy)
crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Parâmetros de treinamento
epochs = 20000  # Número alto de épocas, conforme solicitado
dimensaoRuido = 100
numExemplosParaGerar = 16

# Seed de ruído para gerar sempre as mesmas imagens de exemplo e visualizar a evolução
seed = tf.random.normal([numExemplosParaGerar, dimensaoRuido])

# Iniciar o treinamento
print("Iniciando o treinamento da GAN...")
treinarModelo(datasetTreino, epochs)
