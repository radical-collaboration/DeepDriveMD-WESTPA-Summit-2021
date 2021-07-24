"""Preprocess the Spike protein dataset listed in README.md"""

import time
import h5py
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import MDAnalysis
from MDAnalysis.analysis import rms, align
from concurrent.futures import ProcessPoolExecutor

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


def preprocess(
    topology: PathLike,
    ref_topology: PathLike,
    traj_file: PathLike,
    save_file: Optional[PathLike] = None,
    selection: str = "protein and name CA",
    skip_every: int = 1,
    verbose: bool = False,
) -> Tuple[List[float], List[np.ndarray], float]:
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

    # start timer
    start_time = time.time()

    # Load simulation and reference structures
    sim = MDAnalysis.Universe(str(topology), str(traj_file))
    ref = MDAnalysis.Universe(str(ref_topology))

    if verbose:
        print("Traj length: ", len(sim.trajectory))

    # Align trajectory to compute accurate RMSD and point cloud
    align.AlignTraj(sim, ref, select=selection, in_memory=True).run()

    if verbose:
        print(f"Finish aligning after: {time.time() - start_time} seconds")

    # Atom selection for reference
    atoms = sim.select_atoms(selection)
    # Get atomic coordinates of reference atoms (use first frame)
    ref_positions = ref.select_atoms(selection).positions.copy()

    rmsds, point_clouds = [], []

    iterable = enumerate(sim.trajectory[::skip_every])
    if verbose:
        iterable = tqdm(iterable)
        print("Iterating over trajectory")

    for i, _ in iterable:

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
        if verbose:
            print("Writing HDF5 file")
        write_h5(save_file, rmsds, point_clouds)


    elapsed = time.time() - start_time
    if verbose:
        print(f"Duration {elapsed}s")

    return rmsds, point_clouds, elapsed


def _worker(kwargs):
    """Helper function for parallel data preprocessing."""
    return preprocess(**kwargs)


def parallel_preprocess(
    topology_files: List[PathLike],
    traj_files: List[PathLike],
    ref_topology: PathLike,
    save_files: List[PathLike],
    concatenated_h5: PathLike,
    selection: str = "protein and name CA",
    num_workers: int = 10,
) -> List[float]:

    kwargs = [
        {
            "topology": topology,
            "ref_topology": ref_topology,
            "traj_file": traj_file,
            "save_file": save_file,
            "selection": selection,
            "verbose": not bool(i % num_workers),
        }
        for i, (topology, traj_file, save_file) in enumerate(
            zip(topology_files, traj_files, save_files)
        )
    ]

    rmsds, point_clouds, run_times = [], [], []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for rmsd, point_cloud, run_time in tqdm(executor.map(_worker, kwargs)):
            rmsds.extend(rmsd)
            point_clouds.append(point_cloud)
            run_times.append(run_time)

    point_clouds = np.concatenate(point_clouds)

    write_h5(concatenated_h5, rmsds, point_clouds)
    
    return run_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--raw", help="Path to directory containing raw data", type=Path, required=True
    )
    parser.add_argument(
        "-o", "--preprocessed", help="Path to directory to write preprocessed data", type=Path, required=True
    )
    parser.add_argument(
        "-n", "--name", help="Name of concatenated output HDF5 file", type=str, required=True
    )    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    start = time.time()

    args = parse_args()

    if not args.raw.exists():
        raise ValueError(f"raw path does not exist: {args.raw}")
    if not args.preprocessed.exists():
        raise ValueError(f"preprocessed path does not exist: {args.preprocessed}")
    if len(list(args.raw.glob("*.pdb"))) > 1:
        raise ValueError("Expected a single PDB file in the raw data directory: {args.raw}") 
    if not args.name.endswith(".h5"):
        raise ValueError("Name of concatenated HDF5 file must end with '.h5'")

    # Stage the data on node local ssd
    stage_dir = Path("/tmp") / args.raw.name
    if not stage_dir.exists():
        shutil.copytree(args.raw, stage_dir)

    # Glob a single PDB file to use for alignment and RMSD computation
    ref_topology = next(stage_dir.glob("*.pdb"))
    traj_files = list(stage_dir.glob("*.dcd"))
    topology_files = [ref_topology] * len(traj_files)  # Use same PDB for each traj
    save_files = [args.preprocessed / p.with_suffix(".h5").name for p in traj_files]
    concatenated_h5 = args.preprocessed / args.name

    # Use a core per traj file. Note: MDAnalysis also uses it's own cores.
    run_times = parallel_preprocess(
        topology_files,
        traj_files,
        ref_topology,
        save_files,
        concatenated_h5,
        selection="protein and name CA",
        num_workers=len(traj_files),
    )

    print(f"Elapsed time: {round((time.time() - start), 2)}s")
    mean_time = round(np.mean(run_times), 2)
    std_time = round(np.std(run_times), 2)
    print(f"Preprocessing runtime: {mean_time}s +- {std_time}s")     
    print(run_times)
 

    # Single file preprocessing
    # preprocess(
    #     topology="/path/to/pdb/file",
    #     ref_topology="path/to/pdb/file",
    #     traj_file="/path/to/dcd/file",
    #     save_file="/path/to/h5/file",
    #     selection="protein and name CA",
    #     skip_every=1,
    #     verbose=True,
    # )

