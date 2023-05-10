import os
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset
from .config import get_config
from .vae import load_vae
from . import utils


DEFAULT_HIDDEN_DIMS = [32, 64, 128, 256]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_latent=20,
        hidden_dims=None,
        input_size=64
    ):
        """
        Generator class.

        :param hidden_dims: reversed list of hidden dimensions to match the discriminator
        """

        assert utils.is_power_of_two(input_size)
        super().__init__()

        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS

        conv_layers = []
        in_channels = hidden_dims[-1]

        for out_channels in hidden_dims[-2::-1]:
            layer_group = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            conv_layers.append(layer_group)
            in_channels = out_channels

        last_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, n_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.Tanh(),
        )
        conv_layers.append(last_layer)

        out_size = input_size // (2 ** len(hidden_dims))

        self.layers = nn.Sequential(
            nn.Linear(n_latent, out_size ** 2 * hidden_dims[-1]),
            nn.Unflatten(1, (hidden_dims[-1], out_size, out_size)),
            *conv_layers,
        )


    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        n_channels=3,
        hidden_dims=None,
        input_size=64
    ):
        assert utils.is_power_of_two(input_size)
        super().__init__()

        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS

        conv_layers = []
        in_channels = n_channels

        for i, out_channels in enumerate(hidden_dims):
            group_layers = [
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]

            if i != 0:
                group_layers.append(nn.BatchNorm2d(out_channels, 0.8))

            layer_group = nn.Sequential(*group_layers)
            conv_layers.append(layer_group)
            in_channels = out_channels

        out_size = input_size // (2 ** len(hidden_dims))

        self.layers = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(out_size ** 2 * hidden_dims[-1], 1),
        )

    def forward(self, inp):
        x = self.layers(inp)
        return x.view(-1)


def create_gan_models(config):
    gan_config = config["gan"]
    vae_config = config["vae"]

    input_size = config["input_size"]
    assert input_size[0] == input_size[1], "Input image should be square"

    generator = Generator(
        n_channels=input_size[2],
        n_latent=gan_config["latent"] + vae_config["latent"],
        hidden_dims=gan_config["dims"],
        input_size=input_size[0]
    )

    discriminator = Discriminator(
        n_channels=input_size[2],
        hidden_dims=gan_config["dims"],
        input_size=input_size[0]
    )

    return generator, discriminator


def train_gan(
    config_name,
    output_dir="output",
    batch_size=64,
    num_workers=0,
    epochs=10,
):
    config = get_config(config_name)
    gan_config = config["gan"]
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)

    # Load previously trained VAE model
    vae = load_vae(config_name, checkpoint_dir=output_dir)
    vae.to(device)
    vae.train()

    # Load data
    dataset = get_dataset(config_name)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Initialize model
    generator, discriminator = create_gan_models(config)
    generator.train()
    discriminator.train()

    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=gan_config["lr"], betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=gan_config["lr"], betas=(0.5, 0.999))

    criterion_d = nn.BCEWithLogitsLoss()

    # Main training loop
    num_steps = len(data_loader) * epochs
    losses_g, losses_d = [], []

    with tqdm(total=num_steps, desc="[epoch ?] G_loss = ?, D_loss = ?") as pbar:
        for epoch in range(epochs):
            for real_images in data_loader:
                if isinstance(real_images, (tuple, list)):
                    # assume that the first item is image
                    real_images = real_images[0]

                real_images = real_images.to(device)

                # Train Discriminator
                discriminator.zero_grad()

                bs = real_images.size(0)
                real_labels = torch.full((bs,), 1.0, dtype=torch.float, device=device)

                real_logits = discriminator(real_images)
                loss_d_real = criterion_d(real_logits, real_labels)
                loss_d_real.backward()

                # Get latent from VAE
                with torch.no_grad():
                    vae_z = vae.reparameterize(*vae.encode(real_images))
                gan_z = torch.randn((bs, gan_config["latent"]), device=device)
                
                z = torch.cat([gan_z, vae_z], 1)
                
                fake_images = generator(z)
                fake_labels = torch.full((bs,), 0.0, dtype=torch.float, device=device)

                fake_logits = discriminator(fake_images.detach())
                loss_d_fake = criterion_d(fake_logits, fake_labels)
                loss_d_fake.backward()

                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_d.step()

                loss_d = loss_d_real + loss_d_fake
                losses_d.append(loss_d.item())

                # Train Generator
                generator.zero_grad()
                fake_logits = discriminator(fake_images)

                loss_g = criterion_d(fake_logits, real_labels)
                loss_g.backward()

                nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_g.step()

                losses_g.append(loss_g.item())

                # Update progress
                mean_g_loss = np.mean(losses_g[-50:])
                mean_d_loss = np.mean(losses_d[-50:])
                pbar.set_description(
                    f"[epoch {epoch}] G_loss = {mean_g_loss:.4f}, D_loss = {mean_d_loss:.4f}"
                )
                pbar.update(1)


    os.makedirs(output_dir, exist_ok=True)

    checkpoint_g_path = os.path.join(output_dir, f"{config_name}_gan_g.pt")
    checkpoint_d_path = os.path.join(output_dir, f"{config_name}_gan_d.pt")

    print(f"Saving GAN model to {checkpoint_g_path}, {checkpoint_d_path}")

    generator.cpu()
    discriminator.cpu()

    torch.save(generator.state_dict(), checkpoint_g_path)
    torch.save(discriminator.state_dict(), checkpoint_d_path)

    return {
        "loss_g": losses_g,
        "loss_d": losses_d,
    }


def load_gan(config_name, checkpoint_dir="output"):
    config = get_config(config_name)
    generator, _ = create_gan_models(config)

    checkpoint_path = os.path.join(checkpoint_dir, f"{config_name}_gan_g.pt")
    print(f"Loading GAN model from {checkpoint_path}")

    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()

    return generator
