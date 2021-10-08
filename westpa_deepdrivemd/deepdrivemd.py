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
import time

PathLike = Union[str, Path]


def write_point_cloud(h5_file: h5py.File, point_cloud: np.ndarray):
    h5_file.create_dataset(
        "point_cloud",
        data=point_cloud,
        dtype="float32",
        fletcher32=True,
        chunks=(1,) + point_cloud.shape[1:],
    )


def write_rmsd(h5_file: h5py.File, rmsd: np.ndarray):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def write_h5(
    save_file: PathLike,
    rmsds: np.ndarray,
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
    device: str = "cpu",
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
    device = torch.device(device)
    encoder.to(device)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        drop_last=False,
        shuffle=False,
        #pin_memory=True,
        #num_workers=0,
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
    parser.add_argument(
        "-s",
        "--selection",
        help="MDAnalysis selection string.",
        type=str,
        default="protein and name CA",
    )
    parser.add_argument(
        "-m",
        "--model_cfg",
        help="Model configuration file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-w",
        "--model_weights",
        help="Model weight file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Inference batch size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Torch device (cpu, cuda)",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "-n",
        "--pcoord_dim",
        help="Dimension of pcoord",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    #import sys
    #import time
    #start = time.time()

    #sys.stdout = open(args.output_path.with_suffix(".log"), "w")
    #sys.stderr = open(args.output_path.with_suffix(".err"), "w")
    #print(args.top)
    #print(args.coord)
    #sys.stdout.flush()

    # Need to create temporary h5 file for AI
    h5_file = args.output_path.with_suffix(".h5")

    s1 = s2 = s3 = 0

    if args.coord.suffix == ".restrt":
        #print(".restrt file detected")
        # HACK: change this to pdb file (which we wrote with amber)
        # since mda can't read the chamber file with the restrt
        args.coord = args.coord.with_suffix(".pdb")
        #print("update coord:", args.coord)
        rmsds, positions = preprocess_pdb(
            args.coord, args.ref, selection=args.selection
        )
        #print(positions.shape)
        #sys.stdout.flush()
        #traj_time = time.time()
        # TODO: remove temp array
        # a = np.array([[1, 2]])
    else:
        #print("parent:", args.parent)
        #print("nc:", args.coord)
        #sys.stdout.flush()
        s1 = time.time()
        print("Start preprocess_pdb,{},{}".format(s1, 0))
        parent_rmsds, parent_positions = preprocess_pdb(
            args.parent, args.ref, selection=args.selection
        )
        #parent_time = time.time()
        #print("parent preprocess time:", parent_time - start)
        #print("parent:", parent_positions.shape)
        #sys.stdout.flush()
        s2 = time.time()
        print("Start preprocess_traj,{},{}".format(s2, s2-s1))
        rmsds, positions = preprocess_traj(
            args.parent, args.ref, args.coord, selection=args.selection, verbose=False
        )

        #traj_time = time.time()
        #print("traj preprocess time:", traj_time - parent_time)

        #print("positions:", positions.shape)
        #sys.stdout.flush()
        s3 = time.time()
        print("Start concatenate,{},{}".format(s3, s3-s2))
        rmsds = np.concatenate([parent_rmsds, rmsds])
        positions = np.concatenate([parent_positions, positions])
        #print("rmsds shape", rmsds.shape)
        #print("rmsds", rmsds)
        #print("parent rmsds", parent_rmsds)
        #print("positions shape", positions.shape)
        #sys.stdout.flush()

        # TODO: remove temp array
        # a = np.array([[1, 2], [3, 4], [5, 6]])

    s4 = time.time()
    print("Start write_h5,{},{}".format(s4, s4-s3))
    # Write model input file
    write_h5(h5_file, rmsds, positions)
    #write_time = time.time()
    #print("write h5 time", write_time - traj_time)
    #print("wrote h5")
    #sys.stdout.flush()

    s5 = time.time()
    print("Start generate_embeddings,{},{}".format(s5, s5-s4))
    # Run model in inference
    embeddings = generate_embeddings(
        model_cfg_path=args.model_cfg,
        h5_file=h5_file,
        model_weights_path=args.model_weights,
        inference_batch_size=args.batch_size,
        device=args.device,
    )
    #ai_time = time.time()
    #print("AI time:", ai_time - write_time)
    #print("embeddings:", embeddings.shape)
    #sys.stdout.flush()

    # Take the first pcoord_dim embedding dims to pass to WESTPA
    # binning algorithm

    pcoord = embeddings[:, : args.pcoord_dim]

    #print("pcoord shape:", pcoord.shape)
    #print("pcoord:", pcoord)
    #sys.stdout.flush()

    s6 = time.time()
    print("Start unlink h5,{},{}".format(s6, s6-s5))
    # Can delete H5 file after coordinates have been computed
    h5_file.unlink()
    s7 = time.time()
    print("Start savetxt,{},{}".format(s7, s7-s6))
    #print("writing", args.output_path)
    np.savetxt(args.output_path, pcoord, fmt="%.4f")
    #print("total time:", time.time() - start)
    #sys.stdout.flush()
    #sys.stdout.close()
    #sys.stderr.close()
    s8 = time.time()
    print("Done,{},{}".format(s8, s8-s7))

