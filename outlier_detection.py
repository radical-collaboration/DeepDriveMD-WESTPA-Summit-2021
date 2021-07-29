import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple

# Helper function copied from deepdrivemd
def bestk(
    a: np.ndarray, k: int, smallest: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the best `k` values and correspdonding indices.
    Parameters
    ----------
    a : np.ndarray
        Array of dim (N,)
    k : int
        Specifies which element to partition upon.
    smallest : bool
        True if the best values are small (or most negative).
        False if the best values are most positive.
    Returns
    -------
    np.ndarray
        Of length `k` containing the `k` smallest values in `a`.
    np.ndarray
        Of length `k` containing indices of input array `a`
        coresponding to the `k` smallest values in `a`.
    """
    # If larger values are considered best, make large values the smallest
    # in order for the sort function to pick the best values.
    arr = a if smallest else -1 * a

    # Handle case where k is the full length of the array
    if k == len(arr):
        best_inds = np.argsort(arr)
        return arr[best_inds], best_inds

    # Only sorts 1 element of `arr`, namely the element that is position
    # k in the sorted array. The elements above and below the kth position
    # are partitioned but not sorted. Returns the indices of the elements
    # on the left hand side of the partition i.e. the top k.
    best_inds = np.argpartition(arr, k)[:k]
    # Get the associated values of the k-partition
    best_values = arr[best_inds]
    # Only sorts an array of size k
    sort_inds = np.argsort(best_values)
    return best_values[sort_inds], best_inds[sort_inds]


def detect_outliers(data: np.ndarray, method: str = "lof-sklearn", n_jobs: int = 8):
    if method == "lof-sklearn":
        from sklearn.neighbors import LocalOutlierFactor

        clf = LocalOutlierFactor(n_jobs=n_jobs)
        clf.fit_predict(data)

        # Collect all the scores (not just the bestk scores)
        intrinsic_scores, intrinsic_inds = bestk(
            clf.negative_outlier_factor_, k=len(data)
        )

        return intrinsic_scores, intrinsic_inds

    else:
        raise ValueError(f"Invalid method {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--embeddings_path",
        help="Path to npy embeddings file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--score_output_path",
        help="Name of output npy file containing outlier scores.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--index_output_path",
        help="Name of output npy file containing outlier indices.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--method",
        help="Outlier detection method [lof-sklearn].",
        default="lof-sklearn",
        type=str,
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        help="Number of sklearn jobs to use",
        default=8,
        type=int,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    start = time.time()

    embeddings = np.load(args.embeddings_path)

    outlier_scores, outlier_inds = detect_outliers(embeddings, args.method, args.n_jobs)

    np.save(args.score_output_path, outlier_scores)
    np.save(args.index_output_path, outlier_inds)

    print(f"Elapsed time: {round((time.time() - start), 2)}s")
