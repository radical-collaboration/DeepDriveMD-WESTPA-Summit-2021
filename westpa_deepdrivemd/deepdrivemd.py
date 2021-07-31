import mdtraj
import argparse
import numpy as np
from pathlib import Path

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

def parse_positions(coord: Path, top: Path) -> np.ndarray:
    traj = mdtraj.load(str(args.coord), top=str(args.top))
    print("mdtraj load rst")
    positions = traj.xyz[:, traj.topology.select(selection), :]
    return positions

if __name__ == "__main__":

    args = parse_args()

    print(args.top)
    print(args.coord)

    selection = "protein and name CA"

    if args.coord.suffix == ".rst":
        print(".rst file detected")
        positions = parse_positions(args.coord, args.top)
        print(positions.shape)        
        a = np.array([[1, 2]])
    else:
        a = np.array([[1, 2], [3,4], [5, 6]])
        print(".nc file detected")
   
    #print("writing pcoord.txt")
    print("writing", args.output_path)
    #np.savetxt("pcoord.txt", a, fmt='%.4f')
    np.savetxt(args.output_path, a, fmt='%.4f')
 
