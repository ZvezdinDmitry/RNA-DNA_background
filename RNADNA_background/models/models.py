import torch
from torch import nn


class MLPNoiseModel(nn.Module):
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
