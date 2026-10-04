import torch
from torch import nn


class MLPNoiseModel(nn.Module):
    """Simple two-layer MLP for point-wise (per-bin) background contact prediction.

    Args:
        n_features (int): Number of input features per bin.
        activation: Activation class (e.g. nn.ReLU), instantiated internally.
        hidden (int): Number of hidden units. Defaults to 128.
    """

    def __init__(self, n_features, activation, hidden=128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            activation(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        return self.mlp(batch).squeeze()


class UnetBlock(nn.Module):
    """Basic 1D convolutional block: two convolutions with BatchNorm and activation.

    The second convolution uses dilated kernels to enlarge the receptive field.

    Args:
        kernel_size (int): Convolution kernel size.
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        activation: Activation class, instantiated internally.
        dilation (int): Dilation factor of the second convolution.
    """

    def __init__(
        self, kernel_size, in_ch, out_ch, activation, dilation
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm1d(out_ch),
            activation(),
            nn.Conv1d(
                out_ch, out_ch, kernel_size, padding="same", dilation=dilation
            ),
            nn.BatchNorm1d(out_ch),
            activation(),
        )

    def forward(self, batch):
        return self.block(batch)


class UnetEncoder(nn.Module):
    """U-Net encoder: three convolutional blocks with average pooling between them.

    Feature maps from each level are collected and returned as skip connections.

    Args:
        kernel_size (int): Convolution kernel size.
        features_n (int): Number of input feature channels.
        channels (list[int]): Channel counts for the three blocks.
        activation: Activation class.
        dilation (int): Dilation factor.
    """

    def __init__(
        self, kernel_size, features_n, channels, activation, dilation
    ) -> None:
        super().__init__()
        self.block1 = UnetBlock(
            kernel_size, features_n, channels[0], activation, dilation
        )
        self.block2 = UnetBlock(
            kernel_size, channels[0], channels[1], activation, dilation
        )
        self.block3 = UnetBlock(
            kernel_size, channels[1], channels[2], activation, dilation
        )
        self.pool = nn.AvgPool1d(2)

    def forward(self, batch):
        fmaps = []
        fmap = batch
        for block in (self.block1, self.block2, self.block3):
            fmap = block(fmap)
            fmaps.append(fmap)
            fmap = self.pool(fmap)

        return fmaps


class UnetDecoder(nn.Module):
    """U-Net decoder: progressive upsampling with skip connections from the encoder.

    Args:
        kernel_size (int): Convolution kernel size.
        channels (list[int]): Channel counts (reversed internally to match
            the encoder order).
        activation: Activation class.
        dilation (int): Dilation factor.
    """

    def __init__(self, kernel_size, channels, activation, dilation) -> None:
        super().__init__()
        channels = channels[::-1]
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear"),
            nn.Conv1d(channels[0], channels[1], kernel_size, padding="same"),
            nn.BatchNorm1d(channels[1]),
        )
        self.block1 = UnetBlock(
            kernel_size, channels[0], channels[1], activation, dilation
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear"),
            nn.Conv1d(channels[1], channels[2], kernel_size, padding="same"),
            nn.BatchNorm1d(channels[2]),
        )
        self.block2 = UnetBlock(
            kernel_size, channels[1], channels[2], activation, dilation
        )

    def forward(self, x, fmaps):
        upsampled = self.upsample1(x)
        x = torch.cat([upsampled, fmaps[-1]], dim=1)
        x = self.block1(x)
        upsampled = self.upsample2(x)
        x = torch.cat([upsampled, fmaps[-2]], dim=1)
        x = self.block2(x)
        return x


class UnetNoiseModel(nn.Module):
    """1D U-Net for background contact profile prediction.

    Takes a feature track of shape (batch, features_n, window_bins) and predicts
    a contact profile of shape (batch, window_bins).

    Args:
        kernel_size (int): Convolution kernel size.
        features_n (int): Number of input feature channels.
        channels (list[int]): Channel counts for encoder/decoder blocks.
        activation: Activation class.
        dilation (int): Dilation factor.
    """

    def __init__(
        self, kernel_size, features_n, channels, activation, dilation
    ) -> None:
        super().__init__()
        self.encoder = UnetEncoder(
            kernel_size, features_n, channels, activation, dilation
        )
        self.decoder = UnetDecoder(kernel_size, channels, activation, dilation)
        self.head = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(channels[0]),
            activation(),
            nn.Conv1d(in_channels=channels[0], out_channels=1, kernel_size=1),
            nn.Flatten(),
        )

    def forward(self, batch):
        fmaps = self.encoder(batch)
        x, fmaps = fmaps[-1], fmaps[:-1]
        fmap = self.decoder(x, fmaps)
        out = self.head(fmap)
        return out
