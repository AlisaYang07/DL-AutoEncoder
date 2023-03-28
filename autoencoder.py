import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self,encoder_model, decoder_model, z_dim = 10):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out