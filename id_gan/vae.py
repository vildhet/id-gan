import os
import math
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .data import get_dataset
from .config import get_config


DEFAULT_HIDDEN_DIMS = [32, 64, 128, 512]


class VAE(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_latent=20,
        hidden_dims=None,
        input_size=64
    ):
        # Enforce this restriction to simplify decoder
        assert utils.is_power_of_two(input_size)

        super().__init__()

        self.n_latent = n_latent

        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS

        # Encoder
        encoder_layers = []
        in_channels = n_channels

        for out_channels in hidden_dims:
            layer_group = nn.Sequential(
                nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            encoder_layers.append(layer_group)
            in_channels = out_channels

        out_size = math.ceil(input_size / (2 ** len(hidden_dims)))

        self.encoder = nn.Sequential(
            *encoder_layers,
            nn.Flatten(),
            nn.Linear(out_size ** 2 * hidden_dims[-1], n_latent * 2)
        )

        # Decoder
        decoder_layers = []
        in_channels = hidden_dims[-1]

        for out_channels in hidden_dims[-2::-1]:
            layer_group = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            decoder_layers.append(layer_group)
            in_channels = out_channels

        # Make sure that the last outputs has the same dimensions as original image
        decoder_last_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, n_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(n_channels),
            nn.Tanh(),
        )
        decoder_layers.append(decoder_last_layer)

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, out_size ** 2 * hidden_dims[-1]),
            nn.Unflatten(1, (hidden_dims[-1], out_size, out_size)),
            *decoder_layers
        )

    def get_device(self):
        return next(self.parameters()).device

    def encode(self, inp):
        """
        Encode input image batch to the mu, var
        """

        latent = self.encoder(inp)
        mu = latent[:,:self.n_latent]
        log_var = latent[:,self.n_latent:]

        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        """
        Re-parameterization trick to sample from N(mu, var) from N(0, 1).
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, input):
        """
        VAE forward method. Runs encoder and decoder.

        :param input: input images
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)

        return output, mu, log_var

    @torch.no_grad()
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.n_latent)
        z = z.to(self.get_device())
        return self.decode(z)


def vae_loss(orig_image, recon_image, mu, log_var, beta):
    recon_loss = F.mse_loss(recon_image, orig_image, reduction='sum')
    recon_loss = recon_loss / orig_image.shape[0]

    kl_div = -0.5 * (1 + log_var - mu.pow(2) - torch.exp(log_var))
    kl_div = kl_div.sum(1).mean(0)

    return recon_loss + beta * kl_div


def create_vae_model(config):
    vae_config = config["vae"]

    input_size = config["input_size"]
    assert input_size[0] == input_size[1], "Input image should be square"

    model = VAE(
        n_channels=input_size[2],
        n_latent=vae_config["latent"],
        hidden_dims=vae_config["dims"],
        input_size=input_size[0]
    )
    return model


def train_vae(
    config_name,
    batch_size=64,
    num_workers=0,
    epochs=10,
    output_dir="output",
):
    config = get_config(config_name)
    vae_config = config["vae"]

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)

    # Load data
    dataset = get_dataset(config_name)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Initialize model
    model = create_vae_model(config)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=vae_config["lr"])

    # Main training loop
    num_steps = len(data_loader) * epochs
    losses = []

    with tqdm(total=num_steps, desc="[epoch ?] loss = ?") as pbar:
        for epoch in range(epochs):
            for inp_images in data_loader:
                if isinstance(inp_images, (tuple, list)):
                    # assume that the first item is image
                    inp_images = inp_images[0]

                inp_images = inp_images.to(device)
                recon_images, mu, log_var = model(inp_images)

                loss = vae_loss(inp_images, recon_images, mu, log_var, vae_config["beta"])

                model.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                mean_loss = np.mean(losses[-50:])
                pbar.set_description(f"[epoch {epoch}] loss = {mean_loss:.4f}")
                pbar.update(1)

    os.makedirs(output_dir, exist_ok=True)
    vae_checkpoint_path = os.path.join(output_dir, f"{config_name}_vae.pt")

    print(f"Saving VAE model to {vae_checkpoint_path}")

    model.cpu()

    torch.save(model.state_dict(), vae_checkpoint_path)

    return {
        "loss": losses
    }


def load_vae(config_name, checkpoint_dir="output"):
    config = get_config(config_name)
    model = create_vae_model(config)

    vae_checkpoint_path = os.path.join(checkpoint_dir, f"{config_name}_vae.pt")
    print(f"Loading VAE model from {vae_checkpoint_path}")

    model.load_state_dict(torch.load(vae_checkpoint_path))
    model.eval()

    return model
