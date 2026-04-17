import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=6670, hidden_dims=[2048, 1024, 512, 256], latent_dim=128):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential()
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.encoder.add_module(f"enc_fc{i+1}", nn.Linear(prev_dim, h_dim))
            self.encoder.add_module(f"enc_relu{i+1}", nn.ReLU())
            prev_dim = h_dim

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        self.decoder = nn.Sequential()
        prev_dim = latent_dim
        for i, h_dim in enumerate(reversed(hidden_dims)):
            self.decoder.add_module(f"dec_fc{i+1}", nn.Linear(prev_dim, h_dim))
            self.decoder.add_module(f"dec_relu{i+1}", nn.ReLU())
            prev_dim = h_dim

        self.fc_output = nn.Linear(prev_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return torch.sigmoid(self.fc_output(h))

    def forward(self, x):
        mu, logvar, h = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, h, z


def vae_loss(recon_x, x, mu, logvar):
    """MSE reconstruction loss + KL divergence."""
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


if __name__ == "__main__":
    vae = VAE(input_dim=6670, latent_dim=128)
    x = torch.randn(100, 6670)
    recon_x, mu, logvar, h, z = vae(x)
    loss = vae_loss(recon_x, x, mu, logvar)
    print("VAE Loss:", loss.item())
    print("Feature map shape:", h.shape)
    print("Latent variable shape:", z.shape)
