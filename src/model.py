
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    # def forward(self, x): # AE Only
    #     z = self.encode(x.view(-1, self.input_dim))
    #     return self.decode(z), z


class TimeVAE(nn.Module):
    def __init__(self, input_dim, input_channels, n_enc_layers, enc_dim, kernel, latent_dim, n_dec_layers, dec_dim, poly=None):
        super(TimeVAE, self).__init__()

        # Encoder
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        enc_layers = [nn.Conv1d(input_channels, enc_dim, kernel)]
        for _ in range(n_enc_layers-1):
            enc_layers.append(nn.Conv1d(enc_dim, enc_dim, kernel))
        self.enc_layers = nn.ModuleList(enc_layers)
        conv_out_dim = input_dim
        for _ in range(len(enc_layers)):
            conv_out_dim = math.floor(conv_out_dim - kernel) + 1
        self.dense_enc = nn.Linear(enc_dim * conv_out_dim, enc_dim)
        self.mu = nn.Linear(enc_dim, latent_dim)
        self.logvar = nn.Linear(enc_dim, latent_dim)

        # Decoder base
        self.dense_dec = nn.Linear(latent_dim, dec_dim)
        dec_layers = [nn.ConvTranspose1d(1, dec_dim, kernel)]
        for _ in range(n_dec_layers-1):
            dec_layers.append(nn.ConvTranspose1d(dec_dim, dec_dim, kernel))
        conv_out_dim = dec_dim
        self.dec_layers = nn.ModuleList(dec_layers)
        for _ in range(len(dec_layers)):
            conv_out_dim = conv_out_dim + kernel - 1
        self.out = nn.Linear(dec_dim * conv_out_dim, input_dim * input_channels)

        # Decoder trend
        self.poly = poly
        if poly is not None:
            self.trend_dense_1 = nn.Linear(latent_dim, dec_dim)
            self.trend_dense_2 = nn.Linear(dec_dim, input_channels * (poly+1))
            r = torch.arange(input_dim) / input_dim
            pow = torch.arange(poly+1)
            polyspace = torch.pow(r.expand(poly+1, -1), pow[:, None])
            self.register_buffer('polyspace', polyspace)


    def encode(self, x):
        b = len(x)
        for layer in self.enc_layers:
            x = F.relu(layer(x))
        x = F.relu(self.dense_enc(x.view(b, -1)))
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        b = len(z)
        x = F.relu(self.dense_dec(z).view(-1, 1, self.dec_dim))
        for layer in self.dec_layers:
            x = F.relu(layer(x))
        x = self.out(x.view(b, -1))
        x = x.view(b, self.input_dim, self.input_channels)

        if self.poly is not None:
            x_trend = F.relu(self.trend_dense_1(z))
            x_trend = F.relu(self.trend_dense_2(x_trend))
            x_trend = x_trend.view(b, self.input_channels, self.poly+1)
            x_trend = x_trend @ self.polyspace
            x += x_trend.transpose(-2, -1)
        
        return x

    def forward(self, x):
        mu, logvar = self.encode(x.transpose(-2, -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

if __name__=="__main__":
    kernel=9
    model = TimeVAE(512, 10, 5, 256, kernel, 64, 5, 256, 3).cuda()
    x = torch.randn((4, 512, 10)).cuda()
    recon, mu, logvar = model(x)
    print(recon.shape, mu.shape, logvar.shape)

