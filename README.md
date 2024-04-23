# $\beta$ Variational AutoEncoder for Transfer Learning

## Abstract: 

In this paper, the potency of transfer learning is explored in an image reconstruction task. A novel ResNet18 based $\beta$-variational auto-encoder ($\beta$-VAE) architecture is proposed and developed. The architecture's performance is examined under separate training contexts. Namely, when the model is trained from random-initialization, when the encoder module of the model is initialized with pre-trained weights following training on an image classification task, and when the encoder is constructed with frozen pre-trained weight. An exploration of $\beta$-VAE specific hyper-parameters is also conducted. Further analysis is performed in order to observe the generative abilities of the models, as well as to observe the intermediate learned kernels. We find that a model trained from random-initialization outperforms a model pre-trained weights initialization in all metrics. 

Full code in Local Version: https://github.com/AlisaYang07/DL-AutoEncoder/tree/local_ver


DeepLearning CSC413 Final Project

Restnet18 for Cifar10 Classification was referenced from:
https://github.com/huyvnphan/PyTorch_CIFAR10
![image](https://github.com/AlisaYang07/DL-AutoEncoder/assets/61921004/77576c5a-2439-4f8c-8678-3a26d0fefd7d)
