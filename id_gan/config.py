

CONFIG = {
    "mnist": {
        "input_size": (32, 32, 1),
        "vae": {
            "latent": 20,
            "dims": [32, 64, 128, 256],
            "lr": 3e-4,
            "beta": 4,
        }
    }
}


def get_config(config_name):
    try:
        return CONFIG[config_name]
    except KeyError:
        raise RuntimeError(f"Error: Config for '{config_name}' is not defined.") from None
