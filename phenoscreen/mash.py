"""
Mash wrapper utilities.

Provides Python interfaces for mash sketch and mash screen operations.
"""


import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phenoscreen.utils import logger


@dataclass
class MashVersion:
    """Mash version information."""

    major: int
    minor: int
    patch: int
    full: str

    def __str__(self) -> str:
        return self.full


def check_mash() -> MashVersion:
    """
    Check if mash is available and return version info.

    Returns:
        MashVersion with version details.

    Raises:
        RuntimeError: If version cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["mash", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        version_str = result.stdout.strip() or result.stderr.strip()
        parts = version_str.split(".")
        return MashVersion(
            major=int(parts[0]) if len(parts) > 0 else 0,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            full=version_str,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get mash version: {e.stderr}")


def sketch(
        input_files: list[Path],
        output_path: Path,
        kmer_size: int,
        sketch_size: int,
        threads: int = 4,
) -> Path:
    """
    Create a mash sketch from input FASTA files.

    Args:
        input_files: List of FASTA file paths.
        output_path: Output sketch file path (without .msh extension).
        kmer_size: Kmer size to use for mash sketch.
        sketch_size: Sketch size to use for mash sketch.
        threads: Number of threads to use.

    Returns:
        Path to created sketch file (.msh).

    Raises:
        subprocess.CalledProcessError: If mash sketch fails.
    """
    cmd = [
        "mash",
        "sketch",
        "-o",
        str(output_path),
        "-k",
        str(kmer_size),
        "-s",
        str(sketch_size),
        "-p",
        str(threads),
    ]

    # Add input files
    cmd.extend(str(f) for f in input_files)

    logger.info("Running mash sketch with %d input files", len(input_files))
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    sketch_path = Path(str(output_path) + ".msh")
    logger.info("Created sketch: %s", sketch_path)
    return sketch_path


def screen(
        query_paths: list[Path],
        sketch_path: Path,
        threads: int = 4,
) -> pd.DataFrame:
    """
    Run mash screen to compare query against reference sketch.

    Args:
        query_paths: Paths to query FASTA/FASTQ files.
        sketch_path: Path to reference sketch (.msh).
        threads: Number of threads to use.

    Returns:
        DataFrame with columns: identity, shared_hashes, median_multiplicity,
        p_value, query_id, reference_id

    Raises:
        subprocess.CalledProcessError: If mash screen fails.
    """
    cmd = ["mash", "screen", "-p", str(threads)]

    cmd.extend([str(sketch_path),   " ".join(str(x) for x in query_paths)])

    logger.debug("Running mash screen: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    # Parse output
    return parse_screen_output(result.stdout)


def parse_screen_output(output: str) -> pd.DataFrame:
    """
    Parse mash screen output into a DataFrame.

    Args:
        output: Raw mash screen stdout.

    Returns:
        DataFrame with parsed results.
    """
    rows = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            # Format: identity, shared/total, median_mult, p-value, reference_path, [comment]
            identity = float(parts[0])
            shared_hashes = parts[1]  # e.g., "123/1000"
            shared, total = map(int, shared_hashes.split("/"))
            median_mult = float(parts[2])
            p_value = float(parts[3])
            reference_path = parts[4]
            query_comment = parts[5] if len(parts) > 5 else ""

            rows.append(
                {
                    "identity": identity,
                    "shared_hashes": shared,
                    "total_hashes": total,
                    "median_multiplicity": median_mult,
                    "p_value": p_value,
                    "query_comment": query_comment,
                    "reference_path": reference_path,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("shared_hashes", ascending=False).reset_index(drop=True)
    return df