import torch.nn as nn
import torch
##https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class VAE(nn.Module):
    def __init__(self,encoder_model, decoder_model, z_dim = 128, latent_dim = 64):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.FC_mean  = nn.Linear(z_dim , latent_dim)
        self.FC_var   = nn.Linear (z_dim , latent_dim)
        # self.LeakyReLU = nn.LeakyReLU(0.2)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).cuda()       # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        encoder = self.encoder(x)
        # relu_encoder = self.LeakyReLU(encoder)
        mean = self.FC_mean(encoder)
        var = self.FC_var(encoder)
        z = self.reparameterization(mean, torch.exp(0.5 * var))
        decoder = self.decoder(z)
        decoder = torch.sigmoid(decoder)

        return decoder, mean, var