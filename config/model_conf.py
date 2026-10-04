import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeaturesConfig:
    """Configuration for genomic features used in model training.

    Populates attributes from the .ini file in `__post_init__`. Train/val/test
    chromosome splits are hardcoded for mouse chromosomes (even indices → train,
    odd indices → test, with chr2/8/14/18 held out as validation).

    Attributes:
        config_file_path: Path to the .ini configuration file.
        chromosomes: List of chromosome names.
        window_size: Size of the sliding window (in bins).
        shift: Shift (stride) of the sliding window in bins.
        perc_mask: Percentile threshold used to mask low-signal bins.
        num_features: List of names of continuous (numerical) features.
        cat_features: List of names of categorical features.
        features_group: Feature group name (e.g. 'all', 'sequence', 'epigenome').
        train_chromosomes: Chromosomes used for training.
        val_chromosomes: Chromosomes used for validation.
        test_chromosomes: Chromosomes used for testing.
        mask_zeros: Whether to exclude zero-contact bins from training.
    """

    config_file_path: str | Path = field(repr=False)
    chromosomes: list[str] = field(repr=False)
    window_size: int = field(init=False)
    shift: int = field(init=False)
    perc_mask: float = field(init=False)
    num_features: list[str] = field(init=False)
    cat_features: list[str] = field(init=False)
    features_group: str = field(init=False)
    train_chromosomes: list[str] = field(init=False)
    val_chromosomes: list[str] = field(init=False, repr=False)
    test_chromosomes: list[str] = field(init=False)
    mask_zeros: bool = field(init=False)

    def __post_init__(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file_path)
        self.features_group = conf.get("Features", "group")
        self.window_size = conf.getint("Features", "window_size")
        self.shift = conf.getint("Features", "shift")
        self.perc_mask = conf.getfloat("Features", "perc_mask")
        self.mask_zeros = conf.getboolean("Features", "mask_zeros")

        # hardcoded train test split, works with mouse
        self.train_chromosomes = [f"chr{i}" for i in self.chromosomes][::2]
        self.test_chromosomes = [f"chr{i}" for i in self.chromosomes][1::2]
        self.val_chromosomes = ["chr2", "chr8", "chr14", "chr18"]
        self.test_chromosomes = [
            i for i in self.test_chromosomes if i not in self.val_chromosomes
        ]

        self.num_features = conf.get("All_features", "num_features").split(" ")
        self.cat_features = conf.get("All_features", "cat_features").split(" ")
        if self.features_group != "all":
            self.num_features = [
                feature
                for feature in self.num_features
                if feature
                in conf.get(
                    "All_features", f"{self.features_group}_features"
                ).split(" ")
            ]
            self.cat_features = [
                feature
                for feature in self.cat_features
                if feature
                in conf.get(
                    "All_features", f"{self.features_group}_features"
                ).split(" ")
            ]


@dataclass
class MLPConfig:
    """Configuration for a simple MLP model.

    Hyperparameters are read from the section named `model_name` of the .ini file
    in `__post_init__`.

    Attributes:
        config_file_path: Path to the .ini configuration file.
        model_name: Name of the section in the .ini file with hyperparameters.
        batch_size: Training batch size.
        hidden: Number of hidden units per layer.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        num_epochs: Number of training epochs.
        scheduler: Learning-rate scheduler name.
        div_factor: Divisor for the one-cycle scheduler's initial LR.
        activation_func: Name of the activation function.
    """

    config_file_path: str | Path = field(repr=False)
    model_name: str
    batch_size: int = field(init=False)
    hidden: int = field(init=False)
    lr: float = field(init=False)
    weight_decay: float = field(init=False)
    num_epochs: int = field(init=False)
    scheduler: str = field(init=False)
    div_factor: float = field(init=False)
    activation_func: str = field(init=False)

    def __post_init__(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file_path)
        section = self.model_name
        self.batch_size = conf.getint(section, "batch_size")
        self.hidden = conf.getint(section, "hidden")
        self.lr = conf.getfloat(section, "lr")
        self.weight_decay = conf.getfloat(section, "weight_decay")
        self.num_epochs = conf.getint(section, "num_epochs")
        self.scheduler = conf.get(section, "scheduler")
        self.div_factor = conf.getint(section, "div_factor")
        self.activation_func = conf.get(section, "activation_func")


@dataclass
class UnetConfig:
    """Configuration for a 1D U-Net convolutional model.

    Hyperparameters are read from the section named `model_name` of the .ini file
    in `__post_init__`.

    Attributes:
        config_file_path: Path to the .ini configuration file.
        model_name: Name of the section in the .ini file with hyperparameters.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        num_epochs: Number of training epochs.
        num_conv_layers: Number of convolutional layers in each U-Net block.
        kernel_size: Kernel size of convolutions.
        channels: List of channel counts for successive U-Net blocks.
        dilations: Dilation factor for dilated convolutions.
    """

    config_file_path: str | Path = field(repr=False)
    model_name: str
    lr: float = field(init=False)
    weight_decay: float = field(init=False)
    num_epochs: int = field(init=False)
    num_conv_layers: int = field(init=False)
    kernel_size: int = field(init=False)
    channels: list[int] = field(init=False)
    dilations: int = field(init=False)

    def __post_init__(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file_path)
        section = self.model_name
        self.batch_size = conf.getint(section, "batch_size")
        self.lr = conf.getfloat(section, "lr")
        self.weight_decay = conf.getfloat(section, "weight_decay")
        self.num_epochs = conf.getint(section, "num_epochs")
        self.num_conv_layers = conf.getint(section, "num_conv_layers")
        self.kernel_size = conf.getint(section, "kernel_size")
        self.channels = list(
            map(int, (conf.get(section, "channels").split(" ")))
        )
        self.dilations = conf.getint(section, "dilations")
