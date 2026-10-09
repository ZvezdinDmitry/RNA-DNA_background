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
    """Counts occurrences of a restriction site motif within each DNA bin.

    Args:
        features (pd.DataFrame): DataFrame with columns `dna_chr` and `bin` identifying genomic bins.
        genome (dict): Mapping from chromosome name to BioPython SeqRecord (full chromosome sequences).
        restriction_site (str): Restriction motif to search for.
        params (DataConfig): Configuration object holding `bin_size`.

    Returns:
        pd.DataFrame: Input DataFrame with an added `restriction_sites` column.
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
    """Counts G and C nucleotides within each DNA bin.

    Args:
        features (pd.DataFrame): DataFrame with columns `dna_chr` and `bin` identifying genomic bins.
        genome (dict): Mapping from chromosome name to BioPython SeqRecord (full chromosome sequences).
        params (DataConfig): Configuration object holding `bin_size`.

    Returns:
        pd.DataFrame: Input DataFrame with an added `gc_count` column (absolute count of G+C bases).
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
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregates RNA-DNA contacts into genomic bins based on DNA fragment coordinates.

    The bin index is calculated as `(dna_start + dna_end) // 2 // bin_size`
    and stored in a 'bin' column.

    Args:
        contacts (pd.DataFrame): DataFrame of raw RNA-DNA contacts with columns
            `dna_chr`, `dna_start`, `dna_end`.
        params (DataConfig): Configuration object holding `bin_size`.
        group_by (list[str] | None): Optional extra columns to group by.

    Returns:
        pd.DataFrame: Aggregated DataFrame with one row per bin and a `count` column
            indicating the number of contacts falling into that bin.
    """
    group_by = group_by or []
    contacts["center"] = (contacts["dna_start"] + contacts["dna_end"]) // 2
    contacts["bin"] = contacts["center"] // params.bin_size
    agg_columns = ["dna_chr", "bin"] + group_by

    contacts_binned = (
        contacts[agg_columns]
        .value_counts()
        .reset_index()
        .sort_values(agg_columns)
    )
    return contacts_binned
