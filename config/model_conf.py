import configparser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeaturesConfig:
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
