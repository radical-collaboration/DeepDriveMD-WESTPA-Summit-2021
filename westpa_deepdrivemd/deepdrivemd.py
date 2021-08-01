import argparse
import h5py
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from typing import Union, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from molecules.ml.datasets import PointCloudDataset
from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
from molecules.ml.unsupervised.point_autoencoder.aae import Encoder
from aae_config import AAEModelConfig

PathLike = Union[str, Path]


def write_point_cloud(h5_file: h5py.File, point_cloud: np.ndarray):
    h5_file.create_dataset(
        "point_cloud",
        data=point_cloud,
        dtype="float32",
        fletcher32=True,
        chunks=(1,) + point_cloud.shape[1:],
    )


def write_rmsd(h5_file: h5py.File, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def write_h5(
    save_file: PathLike,
    rmsds: List[float],
    point_clouds: np.ndarray,
):
    """
    Saves data to h5 file. All data is optional, can save any
    combination of data input arrays.
    Parameters
    ----------
    save_file : PathLike
        Path of output h5 file used to save datasets.
    rmsds : List[float]
        Stores rmsd data.
    point_clouds : np.ndarray
        XYZ coordinate data in the shape: (N, 3, num_residues).
    """
    with h5py.File(save_file, "w") as f:
        write_rmsd(f, rmsds)
        write_point_cloud(f, point_clouds)


def preprocess_traj(
    topology: PathLike,
    ref_topology: PathLike,
    traj_file: PathLike,
    save_file: Optional[PathLike] = None,
    selection: str = "protein and name CA",
    skip_every: int = 1,
    verbose: bool = False,
) -> Tuple[List[float], np.ndarray]:
    r"""
    Implementation for generating machine learning datasets
    from raw molecular dynamics trajectory data. This function
    uses MDAnalysis to load the trajectory file and given a
    custom atom selection computes RMSD to reference state,
    and the point cloud (xyz coordinates) of each frame in the
    trajectory.

    Parameters
    ----------
    topology : PathLike
        Path to topology file: CHARMM/XPLOR PSF topology file,
        PDB file or Gromacs GRO file.
    ref_topology : PathLike
        Path to reference topology file for aligning trajectory:
        CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file.
    traj_file : PathLike
        Trajectory file (in CHARMM/NAMD/LAMMPS DCD, Gromacs XTC/TRR,
        or generic. Stores coordinate information for the trajectory.
    save_file : Optional[PathLike]
        Path to output h5 dataset file name.
    selection : str
        Selection set of atoms in the protein.
    skip_every : int
        Only colelct data every `skip_every` frames.
    verbose: bool
        If true, prints verbose output.

    Returns
    -------
    List[float]
        List of RMSD values ready to be written to HDF5.
    List[np.ndarray]
        List of point cloud position data (N, 3, D) ready
        to be written to HDF5, where N is the number of
        frames in the trajectory, and D is the number of
        atoms in :obj:`selection`.
    float
        The runtime of the function in seconds.
    """

    # Load simulation and reference structures
    sim = mda.Universe(str(topology), str(traj_file))
    ref = mda.Universe(str(ref_topology))

    if verbose:
        print("Traj length: ", len(sim.trajectory))

    # Align trajectory to compute accurate RMSD and point cloud
    align.AlignTraj(sim, ref, select=selection, in_memory=True).run()

    # Atom selection for reference
    atoms = sim.select_atoms(selection)
    # Get atomic coordinates of reference atoms (use first frame)
    ref_positions = ref.select_atoms(selection).positions.copy()

    rmsds, point_clouds = [], []

    for _ in sim.trajectory[::skip_every]:

        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions

        # Compute and store RMSD to reference state
        rmsds.append(
            rms.rmsd(positions, ref_positions, center=True, superposition=True)
        )

        # Store reference atoms point cloud of current frame
        point_clouds.append(positions.copy())

    point_clouds = np.transpose(point_clouds, [0, 2, 1])

    if save_file:
        write_h5(save_file, rmsds, point_clouds)

    return rmsds, point_clouds


def preprocess_pdb(
    topology: PathLike,
    ref_topology: PathLike,
    save_file: Optional[PathLike] = None,
    selection: str = "protein and name CA",
) -> Tuple[List[float], np.ndarray]:
    """Preprocess a single PDB file instead of a whole trajectory."""
    sim = mda.Universe(str(topology))
    ref = mda.Universe(str(ref_topology))

    # Align trajectory to compute accurate RMSD and point cloud
    align.AlignTraj(sim, ref, select=selection, in_memory=True).run()

    # Atom selection for reference
    atoms = sim.select_atoms(selection)
    # Get atomic coordinates of reference atoms (use first frame)
    ref_positions = ref.select_atoms(selection).positions.copy()

    rmsds, point_clouds = [], []

    # Point cloud positions of selected atoms in frame i
    positions = atoms.positions

    # Compute and store RMSD to reference state
    rmsds.append(rms.rmsd(positions, ref_positions, center=True, superposition=True))

    # Store reference atoms point cloud of current frame
    point_clouds.append(positions.copy())

    point_clouds = np.transpose(point_clouds, [0, 2, 1])

    if save_file:
        write_h5(save_file, rmsds, point_clouds)

    return rmsds, point_clouds


def generate_embeddings(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
    encoder_gpu: int,
) -> np.ndarray:

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
    for data, *_ in data_loader:
        data = data.to(device)
        embeddings.append(encoder.encode(data).cpu().numpy())
   
    embeddings = np.concatenate(embeddings)

    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # closed.prmtop
    parser.add_argument("-t", "--top", help="Topology file", type=Path, required=True)

    # Try loading this into mdanalysis
    parser.add_argument(
        "-c",
        "--coord",
        help="Trajectory file (.nc) or (.rst)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-p", "--parent", help="Parent .rst file.", type=Path, default=None
    )
    parser.add_argument(
        "-r",
        "--ref",
        help="Reference .pdb file for alignment.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Output txt file to write pcoords to.",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    return args


def parse_pdb_positions(coord: Path, selection: str) -> np.ndarray:
    u = mda.Universe(coord)
    atoms = u.select_atoms(selection)
    return atoms.positions.copy()


def parse_nc_positions(pdb: Path, coord: Path, selection: str) -> np.ndarray:
    u = mda.Universe(str(pdb), str(coord))
    atoms = u.select_atoms(selection)
    positions = []
    for _ in u.trajectory:
        positions.append(atoms.positions.copy())

    print(positions[0].shape)
    print("len positions:", len(positions))
    return np.concatenate(positions)


if __name__ == "__main__":

    args = parse_args()

    print(args.top)
    print(args.coord)

    # Need to create temporary h5 file for AI
    h5_file = args.output_path.with_suffix(".h5")

    selection = "protein and name CA"

    if args.coord.suffix == ".restrt":
        print(".restrt file detected")
        # HACK: change this to pdb file (which we wrote with amber)
        # since mda can't read the chamber file with the restrt
        args.coord = args.coord.with_suffix(".pdb")
        print("update coord:", args.coord)
        rmsds, positions = preprocess_pdb(args.coord, args.ref, selection=selection)
        write_h5(h5_file, rmsds, positions)
        # positions = parse_pdb_positions(args.coord, selection)
        print(positions.shape)
        embeddings = generate_embeddings(
            model_cfg_path=args.model_cfg,
            h5_file=h5_file,
            model_weights_path=args.model_weights,
            inference_batch_size=args.batch_size,
            encoder_gpu=0,)
        a = np.array([[1, 2]])
    else:
        print("parent:", args.parent)
        print("nc:", args.coord)
        # parent_positions = parse_pdb_positions(args.parent, selection)
        parent_rmsds, parent_positions = preprocess_pdb(
            args.parent, args.ref, selection=selection
        )
        print("parent:", parent_positions.shape)
        rmsds, positions = preprocess_traj(
            args.parent, args.ref, args.coord, selection=selection, verbose=True
        )
        # positions = parse_nc_positions(args.parent, args.coord, selection)
        print("positions:", positions.shape)
        rmsds = np.concatenate([parent_rmsds, rmsds])
        positions = np.concatenate([parent_positions, positions])
        print("rmsds shape", rmsds.shape)
        print("rmsds", rmsds)
        print("parent rmsds", parent_rmsds)
        print("positions shape", positions.shape)

        a = np.array([[1, 2], [3, 4], [5, 6]])

    # Can delete H5 file after coordinates have been computed
    h5_file.unlink()
    print("writing", args.output_path)
    np.savetxt(args.output_path, a, fmt="%.4f")
