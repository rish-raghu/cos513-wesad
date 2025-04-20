
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_MLP(nn.Module):
    def __init__(self, input_dim, n_enc_layers, enc_dim, latent_dim, n_dec_layers, dec_dim):
        super(VAE_MLP, self).__init__()

        # Encoder
        self.input_dim = input_dim
        enc_layers = [nn.Linear(input_dim, enc_dim)]
        for _ in range(n_enc_layers-1):
            enc_layers.append(nn.Linear(enc_dim, enc_dim))
        self.enc_layers = nn.ModuleList(enc_layers)
        self.mu = nn.Linear(enc_dim, latent_dim)
        self.logvar = nn.Linear(enc_dim, latent_dim)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, dec_dim)]
        for _ in range(n_dec_layers-1):
            dec_layers.append(nn.Linear(dec_dim, dec_dim))
        self.dec_layers = nn.ModuleList(dec_layers)
        self.out = nn.Linear(dec_dim, input_dim)

    def encode(self, x):
        for layer in self.enc_layers:
            x = F.relu(layer(x))
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.dec_layers:
            z = F.relu(layer(z))
        return self.out(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
