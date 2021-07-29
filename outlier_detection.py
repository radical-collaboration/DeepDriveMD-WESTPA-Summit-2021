import time
import argparse
import numpy as np
from pathlib import Path


def detect_outliers(data: np.ndarray, method: str = "lof-sklearn", n_jobs: int = 8) -> np.ndarray:
    if method == "lof-sklearn":
        from sklearn.neighbors import LocalOutlierFactor

        clf = LocalOutlierFactor(n_jobs=n_jobs)
        clf.fit_predict(data)

        return np.array(clf.negative_outlier_factor_)

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

    outlier_scores = detect_outliers(embeddings, args.method, args.n_jobs)

    np.save(args.score_output_path, outlier_scores)

    print(f"Elapsed time: {round((time.time() - start), 2)}s")
