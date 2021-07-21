"""Preprocess the Spike protein dataset listed in README.md"""

import time
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
import MDAnalysis
from MDAnalysis.analysis import rms, align
from mdtools.writers import write_rmsd, write_point_cloud

PathLike = Union[str, Path]


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


def traj_to_dset(
    topology: PathLike,
    ref_topology: PathLike,
    traj_file: PathLike,
    save_file: Optional[PathLike] = None,
    selection: str = "protein and name CA",
    skip_every: int = 1,
    verbose: bool = False,
) -> Tuple[List[float], List[np.ndarray]]:
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
    """

    # start timer
    start_time = time.time()

    # Load simulation and reference structures
    sim = MDAnalysis.Universe(str(topology), str(traj_file))
    ref = MDAnalysis.Universe(str(ref_topology), str(traj_file))

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

    if verbose:
        print(f"Duration {time.time() - start_time}s")

    return rmsds, point_clouds

if __name__ == "__main__":
    traj_to_dset(
        topology="/homes/abrace/data/spike/Longhorn-2021/spike_WE_renumbered.psf",
        ref_topology="/homes/abrace/data/spike/Longhorn-2021/spike_WE_renumbered.psf",
        traj_file="/homes/abrace/data/spike/Longhorn-2021/spike_WE.dcd",
        save_file="/homes/abrace/data/spike/Longhorn-2021/spike_WE_AAE.h5",
        selection="protein and name CA",
        skip_every=1,
        verbose=True,
    )

