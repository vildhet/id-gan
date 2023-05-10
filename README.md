# High-Fidelity Synthesis with Disentangled Representation

Implementation of the ID-GAN in PyTorch (https://arxiv.org/pdf/2001.04296.pdf)

## Training on MNIST

```python
import id_gan

# Train and save VAE
id_gan.train_vae("mnist", epochs=20, batch_size=512, num_workers=8)

# Train and save GAN based on VAE
id_gan.train_gan("mnist", epochs=60, batch_size=512, num_workers=8)

# Load generator from GAN model
gan = id_gan.load_gan("mnist")

# Sample
with torch.no_grad():
    z = torch.randn(16, 40)
    images = gan(z).numpy()
```

## Training on CelebA dataset

The same code as above, but with the parameter `celeba` instead of `mnist`

## Adding new dataset
1. Add dataset config to the `id_gan/config.py`
2. Add function to create dataset to the `id_gan/data.py`
