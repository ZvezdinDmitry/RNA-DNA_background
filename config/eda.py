import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for experimental data: cell line, organism, genome, and file paths.

    Populates most attributes from an .ini file (passed via `config_file_path`)
    in `__post_init__`. Supported cell lines: mESC, mOPC (mouse); hESC, K562 (human).

    Attributes:
        cell_lines: List of supported cell lines.
        experiments: List of supported RNA-DNA protocols.
        config_file_path: Path to the .ini configuration file.
        experiment: Name of the experiment (e.g. 'RADICL', 'GRID').
        cell_line: Name of the cell line (e.g. 'mESC', 'hESC').
        bin_size: Genomic bin size in bp.
        chromosomes: List of chromosome names for the organism.
        organism: 'mouse' or 'human'.
        genome: Reference genome identifier ('mm10' or 'hg38').
        sample: Name of the selected sample (or None if not specified).
        samples: List of all sample names for the experiment/cell line.
    """

    cell_lines = ["mESC", "mOPC", "hESC", "K562"]
    experiments = ["RADICL", "GRID", "Red-C"]
    config_file_path: str | Path = field(repr=False)
    experiment: str = field(init=False)
    cell_line: str = field(init=False)
    bin_size: int = field(init=False)
    chromosomes: list[str] = field(init=False, repr=False)
    organism: str = field(init=False, repr=False)
    genome: str = field(init=False, repr=False)
    sample: str | None = field(init=False)
    samples: list[str] = field(init=False, repr=False)

    def __post_init__(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file_path)
        self.cell_line = conf.get("Params", "cell_line")
        self.experiment = conf.get("Params", "experiment")
        self.bin_size = conf.getint("Params", "bin_size")
        self.samples = conf.get(
            "Samples", f"{self.experiment}_{self.cell_line}"
        ).split()
        sample = conf.get("Params", "sample")
        if sample:
            sample = int(sample)
            self.sample = self.samples[sample - 1]
        else:
            self.sample = None

        if (
            self.cell_line not in self.cell_lines
            or self.experiment not in self.experiments
        ):
            raise ValueError("Wrong cell line or experiment.")

        if self.cell_line in ["mOPC", "mESC"]:
            self.chromosomes = [*[str(i) for i in range(1, 20)], "X"]
            self.organism = "mouse"
            self.genome = "mm10"
        else:
            self.chromosomes = [*[str(i) for i in range(1, 23)], "X"]
            self.organism = "human"
            self.genome = "hg38"

        self.init_paths(conf)

    def init_paths(self, conf):
        """Builds all data paths from prefixes defined in the .ini file."""
        self.source_data_prefix = Path(conf.get("Paths", "source_data_prefix"))
        self.data_prefix = Path(conf.get("Paths", "data_prefix"))
        self.plots_prefix = Path(conf.get("Paths", "plots_prefix"))
        self.hic_prefix = Path(conf.get("Paths", "hic_prefix"))
        self.learn_prefix = Path(conf.get("Paths", "learn_prefix"))
        self.source_data_path = (
            self.source_data_prefix
            / f"{self.experiment}/{self.cell_line}/tables/tables_basic"
        )
        self.plots_path = (
            self.plots_prefix / f"{self.experiment}/{self.cell_line}"
        )
        self.genome_folder = self.data_prefix / f"genome/{self.genome}"
        self.genome_path = self.genome_folder / f"{self.genome}.fa"
        self.annotation_folder = (
            self.data_prefix / f"annotations/{self.cell_line}"
        )
        self.data_path = (
            self.data_prefix
            / f"{self.experiment}/{self.cell_line}/rnadna_data"
        )
        self.hic_folder = self.hic_prefix / self.cell_line
        self.learn_data_folder = self.learn_prefix / self.cell_line
