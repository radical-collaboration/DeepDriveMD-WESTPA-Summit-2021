import yaml
import json
from pathlib import Path
from typing import TypeVar, Type
from typing import Optional, List, Union
from pydantic import BaseSettings as _BaseSettings

_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class AAEModelConfig(BaseSettings):
    # Input HDF5 file path
    data_path: Path = Path("")
    # Path to write output results to (model weights, etc)
    output_path: Path = Path("")
    # Path to previous weights file used to initialize the network
    init_weights: Optional[Path] = None
    # Name of the dataset in the HDF5 file.
    dataset_name: str = "point_cloud"
    # Name of the RMSD data in the HDF5 file.
    rmsd_name: str = "rmsd"
    # Name of the fraction of contacts data in the HDF5 file.
    fnc_name: str = "fnc"
    # Number of input points in point cloud
    num_points: int = 3375
    # Number of features per point in addition to 3D coordinates
    num_features: int = 0
    # Number of epochs to train
    epochs: int = 10
    # Training batch size
    batch_size: int = 32

    # Optimizer params
    # PyTorch Optimizer name
    optimizer_name: str = "Adam"
    # Learning rate
    optimizer_lr: float = 0.0001

    # Model hyperparameters
    # Latent dimension of the AAE
    latent_dim: int = 64
    # Encoder filter sizes
    encoder_filters: List[int] = [64, 128, 256, 256, 512]
    # Encoder kernel sizes
    encoder_kernel_sizes: List[int] = [5, 5, 3, 1, 1]
    # Generator filter sizes
    generator_filters: List[int] = [64, 128, 512, 1024]
    # Discriminator filter sizes
    discriminator_filters: List[int] = [512, 512, 128, 64]
    encoder_relu_slope: float = 0.0
    generator_relu_slope: float = 0.0
    discriminator_relu_slope: float = 0.0
    use_encoder_bias: bool = True
    use_generator_bias: bool = True
    use_discriminator_bias: bool = True
    # Mean of the prior distribution
    noise_mu: float = 0.0
    # Standard deviation of the prior distribution
    noise_std: float = 1.0
    # Hyperparameters weighting different elements of the loss
    lambda_rec: float = 0.5
    lambda_gp: float = 10

    # Training settings
    # Saves embeddings every embed_interval'th epoch
    embed_interval: int = 1
    # Saves embeddings every checkpoint_interval'th epoch
    checkpoint_interval: int = 1
    # For saving and plotting embeddings. Saves len(validation_set) / sample_interval points.
    sample_interval: int = 20
    # Number of data loaders for training
    num_data_workers: int = 0
    # String which specifies from where to feed the dataset. Valid choices are `storage` and `cpu-memory`.
    dataset_location: str = "storage"


if __name__ == "__main__":
    AAEModelConfig().dump_yaml("aae_template.yaml")
