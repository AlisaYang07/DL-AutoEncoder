# DL-AutoEncoder
DeepLearning CSC413 Final Project

Explanation of Files:
VAE.py - Architecture of VAE wrapper class
autoencoder.py - Initial AE implementation - not used in final product
cat_224x224.jpg - Photo used for Feature mapping
data.py - Methods for loading and treating data
decoder.py - Decoder module architecture
encoder.py - Encoder module architecture
main.py - Equivalent to main_VAE.py, used for automated testing
main_VAE.py - Main file which performs creation and training of Model
requiremments.txt - Python libraries used
resnet18.pt - pretrained weights for resnet18 trained on CIFAR-10
training_VAE.py - Training loop for main and main_VAE

Please Refer to this ipynb for the results and analysis:
https://github.com/AlisaYang07/DL-AutoEncoder/blob/local_ver/main_VAE.ipynb

Graph were created in this Google Sheet:
https://docs.google.com/spreadsheets/d/1XbzHv7R3oRD4gq7R8qFFl8cYE-J4NErH89ZGOhAG0RU/edit#gid=0

All model .py and .csv of the training history can be found in this zip:


Restnet18 for Cifar10 Classification was referenced from:
https://github.com/huyvnphan/PyTorch_CIFAR10
