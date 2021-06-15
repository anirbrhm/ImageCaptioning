# Summary

## Introduction
Being able to automatically describe the content of an image using properly formed English sentences is a very challenging task, but it could have great impact, for instance by helping visually impaired people better understand the content of images on the web.

<img src="https://evergreen.team/assets/images/articles/machine-learning/image_captioning_train.png" alt="Image Captioning"/>

This task is significantly harder, for example, than the well-studied image classification or object recognition tasks, which have been a main focus in the computer vision community. 

Indeed, a description must capture not only the objects contained in an image, but it also must express how these objects relateto each other as well as their attributes and the activities they are involved in. Moreover, the above semantic knowledge has to be expressed in a natural language like English, which means that a language model is needed in addition to visual understanding.

Most previous attempts have proposed to stitch together existing solutions of the above sub-problems, in order to go from an image to its description. In contrast, this model presents a single joint model that takes an image I as input, and is trained to maximize the likelihood p(S|I) of producing a target sequence of words S = {S1, S2, . . .} where each word S(t) comes from a given dictionary, that describes the image adequately. 

## Inspiration 
The main inspiration of this work comes from recent advances in machine translation, where the task is to transform a sentence S written in a source language, into its translation T in the target language, by maximizing p(T|S). For many years, machine translation was also achieved by a series of separate tasks (translating words individually, aligning words, reordering, etc), but recent work has shown that translation can be done in a much simpler way using Recurrent Neural Networks (RNNs) and still reach state-of the-art performance. An “encoder” RNN reads the source sentence and transforms it into a rich fixed-length vector representation, which in turn in used as the initial
hidden state of a “decoder” RNN that generates the target sentence.

<img src="https://miro.medium.com/max/4000/0*UldQZaGB0w3omI3Y.png" alt="Machine Translation"/>

## Contribution of the Paper 
The contributions of the paper are as follows. 
* First, we present an end-to-end system for the problem. It is a neural net which is fully trainable using stochastic gradient descent. Second, our model combines state-of-art sub-networks for vision and language models. These can be pre-trained on larger corpora and thus can take advantage of additional data. Finally, it yields significantly better performance compared to state-of-the-art approaches. 

## Model architecture



#### Encoder
We need to provide image as fixed size of vector to generate text, hence a convolutional neural network is used to encode our images. Here transfer learning is preferred, pretained on ImageNet dataset, to cater the constratints of resources and computation. Torchvision has many pretrained models, and any model can be used like ResNet, AlexNet, Inception_v3, DenseNet. We have to remove the last softmax layer which is for classification purpose, as we only need an encoded vector, of size 512 or 1024 usually.
<p align="center">
  <img width="242" height="39" src="assets/paper4.JPG">
</p>

#### Decoder
This part is a Recurrent Neural Network with LSTM(Long Short Term Memory) cells. A Bidirectional Recurrent Neural Network (BRNN) to compute
the word representations. The BRNN takes a sequence of N words (encoded in a 1-of-k representation) and transforms each one into an h-dimensional vector. However, the representation of each word is enriched by a variably-sized context around that word. Using the index t = 1 . . . N to denote the position of a word in a sentence, the precise form of the BRNN is as follows(*here f is ReLU activation function, hence f(x) = max(0,x)*) : 
<p align="center">
  <img width="277" height="181" src="assets/paper5.JPG">
</p>
<p align="center">
  <img width="277" height="250" src="assets/paper2.JPG">
  <img width="277" height="250" src="assets/paper3.JPG">
</p>

#### Loss
Since the supervision is at the level of entire images and sentences, the strategy is to formulate an image-sentence score as a function of the individual regionword scores. Obviously, a sentence-image pair should have a high matching score if its words have a confident support in the image. The dot product between the i-th region and t-th word as a
measure of similarity and use it to define the score between image k and sentence l as:
<p align="center">
  <img  src="assets/paper7.JPG">
</p>
Here, gk is the set of image fragments in image k and gl
is the set of sentence fragments in sentence l. The indices
k, l range over the images and sentences in the training set.
<p align="center">
  <img width="270" height="240" src="assets/paper6.JPG">
</p>

## Results :
* Training time (model = resnet18):
<pre><code>Epoch : 0 , Avg_loss = 3.141907, Time = 9.89 mins
Epoch : 1 , Avg_loss = 2.978030, Time = 9.89 mins
Epoch : 2 , Avg_loss = 2.879061, Time = 9.88 mins
Epoch : 3 , Avg_loss = 2.800483, Time = 9.90 mins
Epoch : 4 , Avg_loss = 2.734463, Time = 9.88 mins
Epoch : 5 , Avg_loss = 2.676081, Time = 9.90 mins
Epoch : 6 , Avg_loss = 2.625130, Time = 9.89 mins
Epoch : 7 , Avg_loss = 2.579518, Time = 9.90 mins
Epoch : 8 , Avg_loss = 2.538572, Time = 9.90 mins
Epoch : 9 , Avg_loss = 2.501715, Time = 9.90 mins
      </code></pre>
<p align="center">
  <img width="697" height="398" src="assets/training_loss.JPG">
</p>

> ***NOTE** : Here loss plotted is total loss of dataset, which is for 6000 X 5 = 30000 captions. Mean loss is loss plotted divided by 30000 .*

* Validation time (model = resnet18):
<pre><code>
0 3.007883
2 3.035437
4 3.088141
6 3.161836
8 3.253206
</code></pre>
<p align="center">
  <img width="697" height="398" src="assets/validation_loss.JPG">
</p>

> ***NOTE** : Train for more than 50 epochs for optimum results . Training is highly GPU intensive!*

## Acknowledgement
- [Medium : Captioning Images with CNN and RNN, using PyTorch](https://medium.com/@stepanulyanin/captioning-images-with-pytorch-bc592e5fd1a3)
- [Word Embeddings](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
