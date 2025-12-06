import sys

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(1, "..")
import bioframe as bf
from sklearn.preprocessing import StandardScaler

from config.model_conf import FeaturesConfig


def train_val_test_split(
    windows: pd.DataFrame,
    features: pd.DataFrame,
    contacts: pd.DataFrame,
    train_chromosomes: list[str],
    val_chromosomes: list[str],
    test_chromosomes: list[str],
):
    """Split tabular dataset by chromosomes to train val and test sets.

    Args:
        windows (pd.DataFrame): Windows dataframe with column: "chr".
        features (pd.DataFrame): Features dataframe with column: "chrom".
        contacts (pd.DataFrame): Contacts dataframe with column: "dna_chr".
        train_chromosomes (list[str]): Train set chromosomes list.
        val_chromosomes (list[str]): Val set chromosomes list.
        test_chromosomes (list[str]): Test set chromosomes list.

    Returns:
        tuple[tuple[DataFrame, DataFrame, DataFrame],
        tuple[DataFrame, DataFrame, DataFrame],
        tuple[DataFrame, DataFrame, DataFrame]]:
        Each tuple corresponds to windows, features and contacts splitted in 3 DFs.
    """
    assert (
        len(set(train_chromosomes) & set(val_chromosomes))
        == len(set(train_chromosomes) & set(test_chromosomes))
        == len(set(test_chromosomes) & set(val_chromosomes))
        == 0
    )

    windows["chr"] = windows["chr"].astype(str)
    features["chrom"] = features["chrom"].astype(str)
    contacts["dna_chr"] = contacts["dna_chr"].astype(str)

    train_windows = windows[
        windows["chr"].isin(train_chromosomes)
    ].reset_index(drop=True)
    val_windows = windows[windows["chr"].isin(val_chromosomes)].reset_index(
        drop=True
    )
    test_windows = windows[windows["chr"].isin(test_chromosomes)].reset_index(
        drop=True
    )

    train_features = features[features["chrom"].isin(train_chromosomes)]
    val_features = features[features["chrom"].isin(val_chromosomes)]
    test_features = features[features["chrom"].isin(test_chromosomes)]

    train_contacts = contacts[contacts["dna_chr"].isin(train_chromosomes)]
    val_contacts = contacts[contacts["dna_chr"].isin(val_chromosomes)]
    test_contacts = contacts[contacts["dna_chr"].isin(test_chromosomes)]

    return (
        (train_windows, val_windows, test_windows),
        (train_features, val_features, test_features),
        (train_contacts, val_contacts, test_contacts),
    )


def rolling_mean(
    windows: pd.DataFrame,
    features: pd.DataFrame,
    win_size: int,
    features_params: FeaturesConfig,
    win_type: str | None = None,
) -> pd.DataFrame:
    features_with_means = []
    for i, (chrom, start, end) in tqdm(windows.iterrows()):
        selected_features = features.loc[
            (features["chrom"] == chrom)
            & (features["start"] >= start)
            & (features["end"] <= end),
            [*features_params.num_features, *features_params.cat_features],
        ]

        selected_inds = features.loc[
            (features["chrom"] == chrom)
            & (features["start"] >= start)
            & (features["end"] <= end),
            ["chrom", "bin"],
        ]

        rolling_means = (
            selected_features.rolling(
                win_size, center=True, closed="both", win_type=win_type
            )
            .mean()
            .bfill()
            .ffill()
        )
        rolling_means = rolling_means.rename(
            lambda name: f"mean_{name}", axis=1
        )
        selected_features = pd.concat(
            [selected_inds, selected_features, rolling_means], axis=1
        ).reset_index(drop=True)
        features_with_means.append(selected_features)

    features_with_means = (
        pd.concat(features_with_means)
        .reset_index(drop=True)
        .sort_values(by=["chrom", "bin"])
    )

    return features_with_means


def train_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer,
    sheduler=None,
    mask_zeros=False,
):
    model.train()
    epoch_loss = 0
    batch_num = len(loader)
    for i, batch in tqdm(enumerate(loader), total=batch_num):
        features, contacts = batch[0], batch[1]

        features = features.to(device)
        contacts = contacts.to(device)
        predict = model(features)
        loss = criterion(predict, contacts)
        if mask_zeros:
            mask = batch[2]
            loss[~mask] = 0
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sheduler:
            sheduler.step()
        epoch_loss += loss.item()

    return epoch_loss / batch_num


def eval_epoch(model, loader, criterion, device, mask_zeros=False):
    model.eval()
    epoch_loss = 0
    batch_num = len(loader)
    predicts_epoch = []
    contacts_epoch = []
    epoch_loss_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            features, contacts = batch[0], batch[1]
            features = features.to(device)
            contacts = contacts.to(device)
            predict = model(features)
            loss = criterion(predict, contacts)
            if mask_zeros:
                mask = batch[2]
                loss[~mask] = 0
                loss = loss.mean()
            predict = predict.cpu().detach().numpy()
            contacts = contacts.cpu().detach().numpy()
            predict = predict.flatten()
            contacts = contacts.flatten()
            predicts_epoch.append(predict)
            contacts_epoch.append(contacts)
            epoch_loss += loss.item()
            epoch_loss_list.append(loss.item())

    predicts_epoch = np.concatenate(predicts_epoch)
    contacts_epoch = np.concatenate(contacts_epoch)
    scc = ss.spearmanr(contacts_epoch, predicts_epoch)
    return (
        epoch_loss / batch_num,
        scc,
        predicts_epoch,
        contacts_epoch,
        epoch_loss_list,
    )


def train_epoch_unet(
    model,
    loader,
    criterion,
    device,
    optimizer,
    sheduler=None,
    mask_zeros=False,
):
    model.train()
    epoch_loss = 0
    batch_num = len(loader)
    for i, batch in tqdm(enumerate(loader), total=batch_num):
        features, contacts = batch[0], batch[1]

        features = features.to(device)
        contacts = contacts.to(device)
        predict = model(features)
        loss = criterion(predict, contacts)
        if mask_zeros:
            mask = batch[2]
            loss[~mask] = 0
            loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sheduler:
            sheduler.step()
        epoch_loss += loss.item()

    return epoch_loss / batch_num


def eval_epoch_unet(model, loader, criterion, device, mask_zeros=False):
    model.eval()
    epoch_loss = 0
    batch_num = len(loader)
    predicts_epoch = []
    contacts_epoch = []
    epoch_loss_list = []
    epoch_scc = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            features, contacts = batch[0], batch[1]
            features = features.to(device)
            contacts = contacts.to(device)
            predict = model(features)
            loss = criterion(predict, contacts)
            if mask_zeros:
                mask = batch[2]
                loss[~mask] = 0
                loss = loss.mean()
            predict = predict.cpu().detach().numpy()
            contacts = contacts.cpu().detach().numpy()
            epoch_scc += [
                ss.spearmanr(predict[i, :], contacts[i, :])[0]
                for i in range(contacts.shape[0])
            ]
            predict = predict.flatten()
            contacts = contacts.flatten()
            predicts_epoch.append(predict)
            contacts_epoch.append(contacts)
            epoch_loss += loss.item()
            epoch_loss_list.append(loss.item())

    predicts_epoch = np.concatenate(predicts_epoch)
    contacts_epoch = np.concatenate(contacts_epoch)
    scc = ss.spearmanr(contacts_epoch, predicts_epoch)
    epoch_scc = np.array(epoch_scc).mean()
    return (
        epoch_loss / batch_num,
        scc,
        predicts_epoch,
        contacts_epoch,
        epoch_loss_list,
        epoch_scc,
    )


class TabularDataset(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        features: pd.DataFrame,
        contacts: pd.DataFrame,
        features_params: FeaturesConfig,
        mean_features: bool,
        win_size: int | None = None,
        win_type: str | None = None,
        mask_zeros: bool = False,
        scaler: None | StandardScaler = None,
    ):
        windows_merged = bf.merge(
            windows.rename({"chr": "chrom"}, axis=1), min_dist=0
        ).drop("n_intervals", axis=1)

        self.windows_merged = windows_merged
        features = (
            bf.overlap(
                features,
                windows_merged,
                how="left",
            )
            .dropna()
            .reset_index(drop=True)
            .drop(["chrom_", "start_", "end_"], axis=1)
            .sort_values(by=["chrom", "bin"])
        )
        contacts = (
            bf.overlap(
                contacts.rename({"dna_chr": "chrom"}, axis=1),
                windows_merged,
                how="left",
            )
            .dropna()
            .reset_index(drop=True)
            .sort_values(by=["chrom", "bin"])
        )

        self.contacts = contacts.reset_index(drop=True)
        self.features = features.reset_index(drop=True)
        if mean_features:
            self.features = rolling_mean(
                self.windows_merged,
                self.features,
                win_size,
                features_params,
                win_type,
            )
        else:
            self.features = self.features.drop(["start", "end"], axis=1)

        if mask_zeros:
            self.features = self.features.reset_index(drop=True)[
                self.contacts["count"] != 0
            ].reset_index(drop=True)
            self.contacts = self.contacts[contacts["count"] != 0].reset_index(
                drop=True
            )

        self.y = np.array(self.contacts["count"])
        if scaler is None:
            scaler = StandardScaler().fit(
                self.features.drop(["chrom", "bin"], axis=1)
            )
            self.scaler = scaler

        self.scaled_features = scaler.transform(
            self.features.drop(["chrom", "bin"], axis=1)
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.scaled_features[index, :], dtype=torch.float)
        y = torch.tensor(self.y[index], dtype=torch.float)
        return x, y


class NoiseDatasetTracks(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        binned_features: dict[str, pd.DataFrame],
        binned_contacts: dict[str, pd.DataFrame],
        num_features: list[str],
        mean: pd.Series,
        std: pd.Series,
        bin_size: int,
        mask_zeros: bool = False,
        test: bool = False,
    ) -> None:
        super().__init__()
        self.binned_features = binned_features
        self.num_features = num_features
        self.mean = mean
        self.std = std
        self.bin_size = bin_size
        self.binned_contacts = binned_contacts
        self.mask_zeros = mask_zeros
        self.test = test
        # remove duplicate windows from test & val dataset
        # this code may look much better, sorry
        if self.test:
            self.windows = []
            self.windows.append(windows.iloc[[0]])
            for i, row in windows.iterrows():
                if (
                    row.start >= self.windows[-1]["end"].item()
                    or row.chr != self.windows[-1]["chr"].item()
                ):
                    self.windows.append(windows.iloc[[i]])
            self.windows = pd.concat(self.windows)
        else:
            self.windows = windows

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        chrom, start, end = self.windows.iloc[index, :]
        features_df = self.binned_features[chrom]
        sontacts_df = self.binned_contacts[chrom]
        start //= self.bin_size
        end //= self.bin_size
        selected_features = features_df.iloc[start:end, :]
        for num_feature in self.num_features:
            selected_features[num_feature] = (
                selected_features[num_feature] - self.mean[num_feature]
            ) / self.std[num_feature]
        selected_features = torch.from_numpy(
            np.array(selected_features).astype(np.float32).T
        )
        selected_contacts = sontacts_df.iloc[start:end]
        selected_contacts = torch.from_numpy(
            np.array(selected_contacts).astype(np.float32).squeeze()
        )
        if self.mask_zeros:
            return selected_features, selected_contacts, selected_contacts > 0
        else:
            return selected_features, selected_contacts

    def get_with_coordinates(self, index):
        # get contacts and features with corresponding coordinates
        if self.mask_zeros:
            selected_features, selected_contacts, _ = self.__getitem__(index)
        else:
            selected_features, selected_contacts = self.__getitem__(index)

        chrom, start, end = self.windows.iloc[index, :]
        return chrom, start, end, selected_features, selected_contacts
