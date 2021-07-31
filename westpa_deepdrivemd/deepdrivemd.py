import argparse
import numpy as np
from pathlib import Path
import MDAnalysis as mda

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # closed.prmtop
    parser.add_argument(
        "-t", "--top", help="Topology file", type=Path, required=True
    )

    # Try loading this into mdanalysis or mdtraj)
    parser.add_argument(
        "-c", "--coord", help="Trajectory file (.nc) or (.rst)", type=Path, required=True
    )
    parser.add_argument(
        "-p", "--parent", help="Parent .rst file.", type=Path, default=None
    )
    # Output file to write to
    parser.add_argument(
        "-o", "--output_path", help="Output txt file to write pcoords to.", type=Path, required=True
    )
    args = parser.parse_args()
    return args

def _parse_positions(coord: Path, top: Path) -> np.ndarray:
    traj = mdtraj.load(str(args.coord), top=str(args.top))
    print("mdtraj load rst")
    positions = traj.xyz[:, traj.topology.select(selection), :]
    return positions

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

    selection = "protein and name CA"

    if args.coord.suffix == ".restrt":
        print(".restrt file detected")
        # HACK: change this to pdb file (which we wrote with amber) since mda can't read the chamber file with the restrt
        args.coord = args.coord.with_suffix(".pdb")
        print("update coord:", args.coord)
        positions = parse_pdb_positions(args.coord, selection)
        print(positions.shape)        
        a = np.array([[1, 2]])
    else:
        print("parent:", args.parent)
        print("nc:", args.coord)
        parent_positions = parse_pdb_positions(args.parent, selection)
        print("parent:", parent_positions.shape)
        positions = parse_nc_positions(args.parent, args.coord, selection)
        print("positions:", positions.shape)
        a = np.array([[1, 2], [3,4], [5, 6]])
   
    #print("writing pcoord.txt")
    print("writing", args.output_path)
    #np.savetxt("pcoord.txt", a, fmt='%.4f')
    np.savetxt(args.output_path, a, fmt='%.4f')
 
