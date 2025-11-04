# Meu Repositório de Projetos com GANs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Em%20Andamento-green?style=for-the-badge)

Bem-vindo! Este repositório é dedicado à exploração e implementação de **Redes Adversárias Geradoras (Generative Adversarial Networks - GANs)**. Aqui você encontrará desde implementações básicas até projetos mais avançados e aplicações práticas.

## O que são GANs (Generative Adversarial Networks)?

Propostas por Ian Goodfellow em 2014, GANs são uma classe de modelos de *deep learning* que colocam duas redes neurais para competir uma contra a outra em um jogo de soma zero.

O objetivo é aprender a gerar dados novos e sintéticos que sejam indistinguíveis de um conjunto de dados real. Elas são famosas por criar imagens, vídeos ou áudios hiper-realistas.

---

## A Arquitetura Central: O Jogo de Gato e Rato

Uma GAN é composta por duas redes neurais que "batalham" entre si durante o treinamento:

### 1. O Gerador (Generator)
* **O Falsificador:** Seu trabalho é criar dados "falsos" (ex: uma imagem de um rosto que não existe).
* **Input:** Ele recebe um vetor de ruído aleatório (chamado de *espaço latente* ou $z$) como ponto de partida.
* **Objetivo:** Enganar o Discriminador, fazendo-o acreditar que sua criação é real.

### 2. O Discriminador (Discriminator)
* **O Policial (ou Crítico):** Seu trabalho é atuar como um classificador binário.
* **Input:** Ele recebe tanto dados reais (do seu dataset de treino) quanto dados falsos (criados pelo Gerador).
* **Objetivo:** Acertar quais dados são reais e quais são falsos.

### O Processo de Treinamento
O treinamento é uma dança delicada:

1.  O **Gerador** cria um lote de imagens falsas a partir do ruído.
2.  O **Discriminador** recebe um lote misto de imagens reais e falsas e tenta classificá-las.
3.  **Backpropagation do Discriminador:** O Discriminador é penalizado se classificar uma imagem real como falsa, ou uma falsa como real. Ele se atualiza para ficar melhor em detectar fraudes.
4.  **Backpropagation do Gerador:** O Gerador é penalizado se o Discriminador conseguir identificar suas imagens como falsas. Ele se atualiza (com os "pesos congelados" do Discriminador) para produzir imagens cada vez melhores.

> **Equilíbrio:** O sistema atinge a convergência (um "Equilíbrio de Nash") quando o Gerador cria imagens tão realistas que o Discriminador não consegue mais diferenciar o real do falso, acertando com uma probabilidade de apenas 50% (um palpite aleatório).

### A Função de Custo (A Matemática por Trás do Jogo)

O "jogo" entre o Gerador ($G$) e o Discriminador ($D$) é formalmente descrito por uma **função de custo minimax** (essencialmente, uma Entropia Cruzada Binária). O objetivo é encontrar um equilíbrio.

A fórmula original do artigo de Ian Goodfellow é:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

Vamos quebrar essa fórmula:

* $\min_G \max_D$: O Gerador ($G$) tenta **minimizar** o resultado, enquanto o Discriminador ($D$) tenta **maximizar**.
* $\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)]$: O "valor esperado" (a média) da confiança do Discriminador de que os dados reais ($x$) são reais. O $D$ quer que $D(x)$ seja 1 (e $\log(1) = 0$).
* $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$: O valor esperado da confiança do Discriminador de que os dados falsos ($G(z)$) são falsos.
    * O $D$ quer que $D(G(z))$ seja 0 (para que $\log(1-0) = 0$).
    * O $G$ quer que $D(G(z))$ seja 1 (para que $\log(1-1) = -\infty$), minimizando a função.

---

#### Objetivos de Treinamento na Prática

Na prática, otimizamos as duas redes separadamente, tratando-as como duas funções de perda distintas:

**1. Objetivo do Discriminador ($D$)**
O $D$ quer **maximizar** a probabilidade de classificar corretamente reais como reais e falsos como falsos. Ele é treinado usando gradiente ascendente sobre esta função de perda (que é a mesma $V(D,G)$ acima):

$$
\text{Loss}_D = - \left( \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))] \right)
$$
*(Nota: Maximizar $V$ é o mesmo que minimizar $-V$, que é como a maioria das bibliotecas (ex: PyTorch) funciona, já que elas usam gradiente descendente).*

**2. Objetivo do Gerador ($G$)**
O $G$ quer **minimizar** a probabilidade de o $D$ classificar suas criações como falsas.

* **Objetivo Original (Saturante):** $\min_G \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$
    * **Problema:** No início, quando o $G$ é ruim, $D(G(z))$ fica perto de 0. O gradiente de $\log(1-x)$ perto de $x=0$ é muito pequeno (saturação de gradiente), e o $G$ aprende muito devagar.

* **Objetivo Prático (Não-Saturante):** Para resolver isso, invertemos o objetivo. O $G$ é treinado para **maximizar** a probabilidade de o $D$ achar que suas imagens são *reais*. Isso dá gradientes mais fortes no início.

$$
\text{Loss}_G = - \mathbb{E}_{z \sim p_z}[\log D(G(z))]
$$

---

## Projetos Neste Repositório

Aqui está uma visão geral dos projetos incluídos neste repositório.

### 1. `projetoGAN`
* **Descrição:** [TODO: Adicionar uma breve descrição. Ex: Uma implementação de uma GAN simples ou DCGAN (Deep Convolutional GAN) do zero usando PyTorch/TensorFlow para gerar dígitos do dataset MNIST.]
* **Tecnologias:** `Python`, `[PyTorch/TensorFlow]`, `Numpy`
* **Status:** `[Concluído / Em Andamento]`

### 2. `projetoGFPGAN`
* **Descrição:** Um projeto que utiliza a arquitetura **GFPGAN (Generative Facial Prior GAN)**. Este modelo é especializado em "restauração cega de faces" (*blind face restoration*), ou seja, corrigir e melhorar a qualidade de fotos de rostos antigos, borrados ou de baixa resolução com resultados impressionantes.
* **Tecnologias:** `Python`, `PyTorch`, `GFPGAN lib`
* **Status:** `[Concluído / Em Andamento]`

### (Em Breve)
* [TODO: StyleGAN, CycleGAN e PIX2PIX]

---

## Como Executar os Projetos

Cada pasta de projeto contém seu próprio conjunto de instruções. No entanto, a estrutura geral para executá-los é:

### Pré-requisitos
* Python 3.8+
* `pip` (para gerenciamento de pacotes)
* É altamente recomendado usar um ambiente virtual:

```bash
# Criar um ambiente virtual
python -m venv venv

# Ativar (Linux/macOS)
source venv/bin/activate

# Ativar (Windows)
.\venv\Scripts\activate
