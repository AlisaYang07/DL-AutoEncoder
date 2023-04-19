# DL-AutoEncoder
DeepLearning CSC413 Final Project

Explanation of Files:<br/>
* VAE.py - Architecture of VAE wrapper class<br/>
* autoencoder.py - Initial AE implementation - not used in final product<br/>
* cat_224x224.jpg - Photo used for Feature mapping<br/>
* data.py - Methods for loading and treating data<br/>
* decoder.py - Decoder module architecture<br/>
* encoder.py - Encoder module architecture<br/>
* main.py - Equivalent to main_VAE.py, used for automated testing<br/>
* main_VAE.py - Main file which performs creation and training of Model<br/>
* requiremments.txt - Python libraries used<br/>
* resnet18.pt - pretrained weights for resnet18 trained on CIFAR-10<br/>
* training_VAE.py - Training loop for main and main_VAE<br/>

Please Refer to this ipynb for the results and analysis:
https://github.com/AlisaYang07/DL-AutoEncoder/blob/local_ver/main_VAE.ipynb

Graph were created in this Google Sheet:
https://docs.google.com/spreadsheets/d/1XbzHv7R3oRD4gq7R8qFFl8cYE-J4NErH89ZGOhAG0RU/edit#gid=0

All model .pt and .csv of the training history can be found in this zip:
https://drive.google.com/file/d/18XBtCzN4yyXEFOzHrt8WKbcroEnmGmKG/view?usp=sharing


Restnet18 for Cifar10 Classification was referenced from:
https://github.com/huyvnphan/PyTorch_CIFAR10
