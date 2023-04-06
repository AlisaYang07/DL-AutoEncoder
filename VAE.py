import torch.nn as nn
import torch
##https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class VAE(nn.Module):
    def __init__(self,encoder_model, decoder_model, z_dim = 512, latent_dim = 64):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model

        self.z_size = latent_dim
        self.feature_volume = z_dim

        self.FC_mean  = nn.Linear(z_dim , latent_dim)
        self.FC_var   = nn.Linear (z_dim , latent_dim)

        # projection
        # self.relu = nn.ReLU()

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    # def q(self, encoded):
    #     unrolled = encoded.view(-1, self.feature_volume)
    #     return self.FC_mean(unrolled), self.FC_var(unrolled)

    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).cuda()       # sampling epsilon        
        z = mean + var *epsilon                          # reparameterization trick
        return z

    def forward(self, x):
        encoder = self.encoder(x)
        out = self.LeakyReLU(encoder)
        mean = self.FC_mean(out)
        log_var = self.FC_var(out)

        # mean, log_var = self.q(encoder)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        
        out = self.decoder(z)
        # out = self.sigmoid(decoder)

        return out, mean, log_var