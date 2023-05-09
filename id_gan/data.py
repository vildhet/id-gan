from torchvision import transforms
from torchvision.datasets import MNIST, CelebA

from .config import CONFIG


DATA_PATH = "data"


def get_mnist_dataset():
    config = CONFIG["mnist"]
    input_size = config["input_size"][:2]

    image_transforms = transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    return MNIST(DATA_PATH, train=True, transform=image_transforms, download=True)


def get_celeba_dataset():
    config = CONFIG["celeba"]
    input_size = config["input_size"][:2]

    image_transforms = transforms.Compose([
        transforms.Resize(input_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return CelebA(DATA_PATH, transform=image_transforms, download=True)


def get_dataset(config_name):
    try:
        get_dataset_func = globals()[f"get_{config_name}_dataset"]
    except KeyError():
        raise RuntimeError(f"Error: Function 'id_gan.data.get_{config_name}_dataset' is not defined.") from None

    return get_dataset_func()
