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
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    print(args.top)
    print(args.coord)

    if args.coord.suffix == ".rst":
        a = np.array([[1, 2]])
    else:
        a = np.array([[1, 2], [3,4], [5, 6]])
    
    np.savetxt("pcoord.txt", a, fmt='%.4f')
 
