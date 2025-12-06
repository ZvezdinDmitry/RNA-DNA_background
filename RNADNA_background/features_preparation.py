import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(1, "..")
from config.eda import DataConfig


def restriction_site_count(
    features: pd.DataFrame,
    genome: dict,
    restriction_site: str,
    params: DataConfig,
) -> pd.DataFrame:
    """Counts restriction sites number in DNA parts of given fragment.

    Args:


    Returns:
        pd.DataFrame: features DF with restriction_sites column.
    """
    features["restriction_sites"] = 0
    for i, row in tqdm(features.iterrows()):
        dna_chr, bin = row.dna_chr, row.bin
        bin_start = int(bin * params.bin_size)
        bin_end = bin_start + params.bin_size
        string = str(genome[dna_chr][bin_start:bin_end].seq).upper()
        sites_count = string.count(restriction_site)
        features.loc[i, "restriction_sites"] = sites_count

    return features


def GC_content_count(
    features: pd.DataFrame, genome: dict, params: DataConfig
) -> pd.DataFrame:
    """Calculates GC content.

    Args:

    Returns:
        pd.DataFrame: features DF with gc_count column.
    """
    features["gc_count"] = 0
    for i, row in tqdm(features.iterrows()):
        dna_chr, bin = row.dna_chr, row.bin
        bin_start = int(bin * params.bin_size)
        bin_end = bin_start + params.bin_size
        string_dna = str(genome[dna_chr][bin_start:bin_end].seq).upper()
        GC_count = string_dna.count("G") + string_dna.count("C")
        features.loc[i, "gc_count"] = GC_count

    return features


def interactions_bining(
    contacts: pd.DataFrame,
    params: DataConfig,
    bin_columns: list[str] = ["dna_chr", "bin"],
) -> pd.DataFrame:
    """_summary_

    Args:
        contacts (pd.DataFrame): _description_
        params (EdaParams): _description_

    Returns:
        pd.DataFrame: _description_
    """
    contacts["center"] = (contacts["dna_start"] + contacts["dna_end"]) // 2
    contacts["bin"] = contacts["center"] // params.bin_size

    contacts_binned = (
        contacts[bin_columns]
        .value_counts()
        .reset_index()
        .sort_values(bin_columns)
    )
    return contacts_binned
