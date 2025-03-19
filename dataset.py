import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClimatePhenoDataset(Dataset):
    def __init__(
        self,
        folder,
        target_list=None,
        normalise_climate=True,
        normalise_dates=True,
        sigma_jitter=0,
        one_year_adaptive_range=True,
        start_date=None,
        end_date=None,
        nan_value_target=-1000,
        nan_value_climate=0,
        monthly=False,
        phases_as_input=None,
    ):
        """Dataset class for phenology prediction from climate time series

        Args:
            - folder (str): path to the dataset folder
            - target_list (list[str]): list of the names of the phenophases to use as target, 
            format: ["{species_name}:{phenophase_name}"]. If None, returns all available phenophases. 
            Defaults to None.
            - normalise_climate (bool, optional): If True, the values of each climate variable are standardised 
            (0 mean and unit variance). Defaults to True.
            - normalise_dates (bool, optional): If True, the values of each phenophase are standardised 
            (0 mean and unit variance). Defaults to True.
            - sigma_jitter (int, optional): Variance of the gaussian noise added to the climate inputs for augmentation.
            Defaults to 0.
            - one_year_adaptive_range (bool, optional): If True, the time range is set adaptively to the phenophases that
            were selected. For spring phenophases, the time range is set to [-101, 264] and for autumn phenophases,
            the time range is set to [1, 365]. Defaults to True.
            - start_date (int, optional): Day of year from which to start the climate time series. Defaults to None.
            - end_date (int, optional): Day of year at which to end the climate time series. Defaults to None.
            - nan_value_target (int, optional): Numerical value used to fill nans in the target data. Defaults to -1000.
            - nan_value_climate (int, optional): Numerical value used to fill nans in the input data. Defaults to 0.
            - monthly (bool, optional): If True, the climate time series is composed of monthly averages 
            instead of daily values. Defaults to False.
            - phases_as_input (list[str]): list of the names of the phenophases used as input to the model,
            format: ["{species_name}:{phenophase_name}"]. If None, returns no phenophase. Defaults to None.
        """
        self.data_dir = Path(folder)
        self.target_list = target_list
        self.nan_value_target = nan_value_target
        self.nan_value_climate = nan_value_climate
        self.normalise_climate = normalise_climate
        self.normalise_dates = normalise_dates
        self.sigma_jitter = sigma_jitter
        self.start_date = start_date
        self.end_date = end_date
        self.monthly = monthly
        self.phases_as_input = phases_as_input

        if phases_as_input is not None:
            assert len(target_list) == 1, "Only one target phenophase allowed when using phases_as_input"

        # Read normalisation values
        with open(self.data_dir / "normalisation.json") as file:
            self.norm_values = json.loads(file.read())

        # Read Data
        self.phenology_observations = pd.read_csv(self.data_dir / "phenology.csv", index_col=0)
        if self.target_list is not None:
            # remove site-years with no relevant observations
            self.phenology_observations = self.phenology_observations[
                (~self.phenology_observations[self.target_list].isna()).any(axis=1)
                ]
        else:
            self.target_list = [c for c in self.phenology_observations.columns if ":" in c]

        # Adaptive time range
        if one_year_adaptive_range:
            phases = np.unique([c.split(":")[1] for c in self.target_list]).tolist()
            if np.all([p in ['leaf_unfolding','flowering','needle_emergence']  for p in phases]):
                print("Setting spring time range : [-101, 264]")
                self.start_date = -101 
                self.end_date = 264
            elif np.all([p in ['leaf_colouring','needle_colouring']  for p in phases]):
                print("Setting autumn time range : [1, 365]")
                self.start_date = 1
                self.end_date = 365  
            else:
                print("Dataset contains mix of autumn and spring phases")
                print("Cannot use adaptive time range, set start and end dates manually. ") 
                raise 
        
        self.n_observations = (~self.phenology_observations[self.target_list].isna()).sum().sum()
        self.n_targets = len(self.target_list)

        # Read the climate data and normalise it
        self.climate_vars = {}
        for f in [
            f for f in os.listdir(self.data_dir / "climate-data") if f.endswith(".csv")
        ]:
            var_name = f.split(".")[0]

            self.climate_vars[var_name] = pd.read_csv(
                self.data_dir / "climate-data" / f, index_col=0
            )
            if self.normalise_climate:
                self.climate_vars[var_name] = (
                    self.climate_vars[var_name] - self.norm_values[var_name]["mean"]
                ) / self.norm_values[var_name]["std"]

            if self.start_date is not None:
                self.climate_vars[var_name] = self.climate_vars[var_name][
                self.climate_vars[var_name].index >= self.start_date
            ]
            if self.end_date is not None:
                self.climate_vars[var_name] = self.climate_vars[var_name][
                self.climate_vars[var_name].index <= self.end_date
            ]
            
        self.var_names = list(self.climate_vars.keys())

        # Normalise phenology observations
        if self.normalise_dates:
            means = np.nanmean(self.phenology_observations[self.target_list], 0)
            stds = np.nanstd(self.phenology_observations[self.target_list], 0)
            self.target_scaler = {
                k: (means[i], stds[i]) for i, k in enumerate(self.target_list)
            }

        # Prepare normalised phenpophase dates if given as input
        if self.phases_as_input is not None:
            imeans = np.nanmean(self.phenology_observations[self.phases_as_input], 0)
            istds = np.nanstd(self.phenology_observations[self.phases_as_input], 0)
            self.input_phase_scaler = {
                k: (imeans[i], istds[i]) for i, k in enumerate(self.phases_as_input)
            }

        # Get list of site-years for which we have all the data
        site_years = list(self.phenology_observations.index)
        available_site_years = set(site_years)
        for cv, data in self.climate_vars.items():
            available_site_years = available_site_years.intersection(set(data.columns))
        self.site_years = sorted(list(available_site_years))
        self.sites = []
        self.years = []
        for sy in self.site_years:
            self.sites.append("_".join(sy.split("_")[:2]))
            self.years.append(sy.split("_")[-1])

        # Compute normalisation values for the static features
        self.stats_elevation = (
            np.mean(self.phenology_observations.elevation),
            np.std(self.phenology_observations.elevation),
        )
        self.stats_latlon = (
            np.mean(self.phenology_observations[["lat", "long"]], 0),
            np.std(self.phenology_observations[["lat", "long"]], 0),
        )

    def __len__(self):
        """Length of the dataset (number of site-years)"""
        return len(self.site_years)

    def __getitem__(self, item, return_siteyear=False):
        """Get a sample from the dataset
        Input:
            - item (int): index of the sample
            - return_siteyear (bool): if True, return the site-year identifier
        Returns:
        A dictionary containing the following keys:
            - climate (torch.Tensor): climate time series
            - doys (torch.Tensor): day of year of each climate record
            - target (torch.Tensor): phenology observations
            - elevation (float): elevation of the observation site
            - latlon (torch.Tensor): latitude and longitude of the observation site
            - year (int): year of the observation
            - elevation_normalised (float): normalised elevation
            - latlon_normalised (torch.Tensor): normalised latitude and longitude
            - input_phases (torch.Tensor): phenology observations used as input (if any)
        """
        out = {}
        ## Get climate time series
        data = []
        for n in self.var_names:
            if self.monthly == True: # compute monthly averages
                daily_data = self.climate_vars[n][self.site_years[item]]
                year = int(daily_data.name.split("_")[-1])

                if isinstance(
                    daily_data.index[-1], (np.int64, np.int16, np.int32)
                ):  ## to
                    daily_data.index = compose_date(
                        years=[year for _ in range(daily_data.shape[0])],
                        days=daily_data.index,
                    )
                monthly_data = daily_data.resample(rule="M").mean()
                data.append(monthly_data.values)
            else:
                data.append(self.climate_vars[n][self.site_years[item]].values)
        data = np.stack(data, axis=1) # stack all climate variables together 
        data[np.isnan(data)] = self.nan_value_climate # fill nans with nan_value_climate

        ## Get day of year of each climate record
        if self.monthly:
            doys = np.round(
                (monthly_data.index - pd.to_datetime(f"{year}-01-31"))
                / np.timedelta64(1, "M")
            ).astype(int)
        else:
            doys = np.array(self.climate_vars[n].index)

        ### Input data augmentation
        if self.sigma_jitter > 0:
            data = data + np.random.randn(*data.shape) * self.sigma_jitter

        out["climate"] = data
        out["doys"] = doys

        ## Get phenology observations
        target = self.phenology_observations.loc[self.site_years[item]][self.target_list]

        if self.normalise_dates:
            target = target.fillna(self.nan_value_target).to_dict()
            for k in self.target_list:
                if target[k] != self.nan_value_target:
                    m, s = self.target_scaler[k]
                    target[k] = (target[k] - m) / s
        else:
            target = target.fillna(self.nan_value_target).to_dict()

        out["target"] = target

        ## Additional metadata
        out["elevation"] = self.phenology_observations.loc[self.site_years[item]]["elevation"]
        out["latlon"] = np.array(
            self.phenology_observations.loc[self.site_years[item]][["lat", "long"]]
        ).astype(float)
        out["year"] = self.phenology_observations.loc[self.site_years[item]]["year"]
        out["elevation_normalised"] = (
            out["elevation"] - self.stats_elevation[0]
        ) / self.stats_elevation[1]
        out["latlon_normalised"] = (
            out["latlon"] - self.stats_latlon[0]
        ) / self.stats_latlon[1]

        ## Additional phenophase as input
        if self.phases_as_input is not None:
            phases = self.phenology_observations.loc[self.site_years[item]][self.phases_as_input]
            phases = phases.fillna(self.nan_value_target).to_dict()
            for k in self.phases_as_input:
                if phases[k] != self.nan_value_target:
                    m, s = self.input_phase_scaler[k]
                    phases[k] = (phases[k] - m) / s
            out["input_phases"] = phases

        out = tensorify_dict(out)
        if return_siteyear:
            return out, self.site_years[item]
        else:
            return out


## utils
def tensorify_dict(obj):
    if not isinstance(obj, dict):
        return torch.tensor(obj).float()
    else:
        return {k: tensorify_dict(v) for k, v in obj.items()}


def to_day_of_year(x, ref_year):
    if x.year == ref_year:
        return x.dayofyear
    else:
        return x.dayofyear - 365


def compose_date(
    years,
    months=1,
    days=1,
    weeks=None,
    hours=None,
    minutes=None,
    seconds=None,
    milliseconds=None,
    microseconds=None,
    nanoseconds=None,
):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = (
        "<M8[Y]",
        "<m8[M]",
        "<m8[D]",
        "<m8[W]",
        "<m8[h]",
        "<m8[m]",
        "<m8[s]",
        "<m8[ms]",
        "<m8[us]",
        "<m8[ns]",
    )
    vals = (
        years,
        months,
        days,
        weeks,
        hours,
        minutes,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds,
    )
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals) if v is not None)


#### Splitting methods
def split_attributes(sample_attributes, ratio=0.7):
    unique_attributes = np.unique(sample_attributes)
    n_total = len(unique_attributes)
    n_train = int(ratio * n_total)
    n_val = int((n_total - n_train) / 2)
    np.random.shuffle(unique_attributes)
    train, val, test = np.split(unique_attributes, [n_train, n_train + n_val])
    return train, val, test


def random_split(n_sample, ratio=0.7):
    idxs = list(range(n_sample))
    train_idxs, val_idxs, test_idxs = split_attributes(idxs, ratio=ratio)
    return train_idxs, val_idxs, test_idxs


def get_matching_indices(sample_attributes, selected_attributes):
    return list(
        np.concatenate(
            [
                np.where(np.array(sample_attributes) == item)[0]
                for item in selected_attributes
            ]
        )
    )

def random_conditional_split(sample_attributes, ratio=0.7):
    "split randomly based on sites or years"
    train, val, test = split_attributes(sample_attributes, ratio)
    train_idxs = get_matching_indices(sample_attributes, train)
    val_idxs = get_matching_indices(sample_attributes, val)
    test_idxs = get_matching_indices(sample_attributes, test)
    return train_idxs, val_idxs, test_idxs

def merge(sites, years):
    out = []
    for s in sites:
        for y in years:
            out.append(f"{s}_{y}")
    return out

def spatio_temporal_split(dt, ratio=0.7):
    "Split based on sites AND years"
    train_sites, val_sites, test_sites = split_attributes(dt.sites, ratio)
    train_years, val_years, test_years = split_attributes(dt.years, ratio)

    train_siteyears = merge(train_sites, train_years)
    val_siteyears = merge(val_sites, val_years)
    test_siteyears = merge(test_sites, test_years)

    train_idxs = list(
        np.concatenate(
            [np.where(np.array(dt.site_years) == item)[0] for item in train_siteyears]
        )
    )
    val_idxs = list(
        np.concatenate(
            [np.where(np.array(dt.site_years) == item)[0] for item in val_siteyears]
        )
    )
    test_idxs = list(
        np.concatenate(
            [np.where(np.array(dt.site_years) == item)[0] for item in test_siteyears]
        )
    )
    return train_idxs, val_idxs, test_idxs
