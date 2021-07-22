import itertools
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from molecules.ml.datasets import PointCloudDataset
from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
from molecules.ml.unsupervised.point_autoencoder.aae import Encoder
from aae_config import AAEModelConfig

PathLike = Union[str, Path]


def setup_mpi(comm=None) -> Tuple[int, int]:
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

    return comm_size, comm_rank


def shard_dataset(dataset: Dataset, comm_size: int, comm_rank: int) -> Dataset:

    fullsize = len(dataset)
    chunksize = fullsize // comm_size
    start = comm_rank * chunksize
    end = start + chunksize
    subset_indices = list(range(start, end))
    # deal with remainder
    for idx, i in enumerate(range(comm_size * chunksize, fullsize)):
        if idx == comm_rank:
            subset_indices.append(i)
    # split the set
    dataset = Subset(dataset, subset_indices)

    return dataset


def generate_embeddings(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int,
    comm=None,
) -> np.ndarray:

    comm_size, comm_rank = setup_mpi(comm)

    model_cfg = AAEModelConfig.from_yaml(model_cfg_path)

    model_hparams = AAE3dHyperparams(
        num_features=model_cfg.num_features,
        encoder_filters=model_cfg.encoder_filters,
        encoder_kernel_sizes=model_cfg.encoder_kernel_sizes,
        generator_filters=model_cfg.generator_filters,
        discriminator_filters=model_cfg.discriminator_filters,
        latent_dim=model_cfg.latent_dim,
        encoder_relu_slope=model_cfg.encoder_relu_slope,
        generator_relu_slope=model_cfg.generator_relu_slope,
        discriminator_relu_slope=model_cfg.discriminator_relu_slope,
        use_encoder_bias=model_cfg.use_encoder_bias,
        use_generator_bias=model_cfg.use_generator_bias,
        use_discriminator_bias=model_cfg.use_discriminator_bias,
        noise_mu=model_cfg.noise_mu,
        noise_std=model_cfg.noise_std,
        lambda_rec=model_cfg.lambda_rec,
        lambda_gp=model_cfg.lambda_gp,
    )

    encoder = Encoder(
        model_cfg.num_points,
        model_cfg.num_features,
        model_hparams,
        str(model_weights_path),
    )

    dataset = PointCloudDataset(
        str(h5_file),
        model_cfg.dataset_name,
        model_cfg.rmsd_name,
        model_cfg.rmsd_name,
        model_cfg.num_points,
        model_hparams.num_features,
        split="train",
        split_ptc=1.0,
        normalize="box",
        cms_transform=False,
    )

    # shard the dataset
    if comm_size > 1:
        dataset = shard_dataset(dataset, comm_size, comm_rank)

    # Put encoder on specified CPU/GPU
    device = torch.device(f"cuda:{encoder_gpu}")
    encoder.to(device)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    # Collect embeddings (requires shuffle=False)
    embeddings = []
    for i, (data, *_) in enumerate(data_loader):
        data = data.to(device)
        embeddings.append(encoder.encode(data).cpu().numpy())
        if (comm_rank == 0) and (i % 100 == 0):
            print(f"Batch {i}/{len(data_loader)}")

    if comm_size > 1:
        # gather results
        embeddings = comm.allgather(embeddings)
        embeddings = list(itertools.chain.from_iterable(embeddings))

    embeddings = np.concatenate(embeddings)

    return embeddings


if __name__ == "__main__":

    embeddings = generate_embeddings(
        model_cfg_path="./aae_template.yaml",
        h5_file="/scratch/06079/tg853783/ddmd/data/spike_WE_AAE.h5",
        model_weights_path="/scratch/06079/tg853783/ddmd/runs/aae_runs/run-1/checkpoint/epoch-100-20210721-203025.pt",
        inference_batch_size=512,
        encoder_gpu=0,
    )

    np.save("/homes/abrace/spike_WE_AAE_embeddings.npy", embeddings)
