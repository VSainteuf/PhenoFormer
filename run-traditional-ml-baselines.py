import datetime
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from torch.utils.data import Subset
from tqdm import tqdm

from configs.PROBLEM_CONFIG import (available_species_phase, phases_table,
                                    seeds, species_table, target_list_parser)
from configs.RUN_CONFIGS import datasplit_configs, training_configs
from dataset import ClimatePhenoDataset, get_matching_indices

data_folder = "path/to/data"
save_dir = "path/to/outputdir"

### CONFIG 1 : Random Forest Regressor with daily climate data as input
# model = "RF"
# monthly = False
# name = "rf_daily"

### CONFIG 2 : Random Forest Regressor with monthly climate data as input
model = "RF"
monthly = True
name = "rf_monthly"

### CONFIG 3 : Gradient Boosted Machine Regressor with daily climate data as input
# model = "GBM"
# monthly = False
# name = "gbm_daily"

### CONFIG 4 : Gradient Boosted Machine Regressor with monthly climate data as input
# model = "GBM"
# monthly = True
# name = "gbm_monthly"

training_config = training_configs["default"]

to_do_list = ["uniformly_rdm", "rdm_spatial", "rdm_temporal", "structured_temporal"]

print("Model : ", name)
for ds_name in to_do_list:
    print("starting split  ", ds_name)

    output_file_metric = Path(save_dir) / f"{name}-{ds_name}-metrics.json"
    os.makedirs(output_file_metric.parent, exist_ok=True)

    n_fold = 40

    split_config = datasplit_configs[ds_name]
    if split_config["split_mode"].endswith(".json"):
        with open(split_config["split_mode"]) as file:
            split = json.loads(file.read())

    results = {}
    predictions = {}

    # Prepare the structure for the output file
    for species_short, species_name in species_table.items():
        for phase_short, phase_name in phases_table.items():
            if f"{species_name}:{phase_name}" in available_species_phase:
                results[f"{species_short}:{phase_short}"] = {}
                predictions[f"{species_short}:{phase_short}"] = {}

    # Iterate over all phenophases
    for species_short, species_name in species_table.items():
        for phase_short, phase_name in phases_table.items():
            if f"{species_name}:{phase_name}" in available_species_phase:
                print(
                    f"{datetime.datetime.now().strftime(format='%Y-%m-%d_%H%M')} starting {species_short}:{phase_short} "
                )

                # Load dataset
                dt_base_args = dict(
                    folder=data_folder,
                    target_list=target_list_parser(f"{species_short}:{phase_short}"),
                    normalise_climate=True,
                    normalise_dates=True,
                    start_date=training_config["start_date"],
                    nan_value_target=training_config["nan_value_target"],
                    monthly=monthly,
                )
                dt = ClimatePhenoDataset(**dt_base_args)
                m, s = list(dt.target_scaler.values())[0]

                # Iterate over folds
                for fold in tqdm(range(1, n_fold + 1)):
                    # Set random seed
                    random.seed(seeds[fold])
                    os.environ["PYTHONHASHSEED"] = str(seeds[fold])
                    np.random.seed(seeds[fold])

                    # Load site-year split
                    if split_config["split_mode"].endswith(".json"):
                        train_idxs = get_matching_indices(
                            dt.site_years, split[str(fold)]["train"]
                        )
                        test_idxs = get_matching_indices(
                            dt.site_years, split[str(fold)]["test"]
                        )
                    elif split_config["split_mode"] == "structured":
                        train_idxs = list(
                            np.where(
                                np.array(dt.years).astype(int)
                                <= split_config["train_years_to"]
                            )[0]
                        )
                        test_idxs = list(
                            np.where(
                                np.array(dt.years).astype(int)
                                > split_config["val_years_to"]
                            )[0]
                        )
                        val_idxs = list(
                            set(range(len(dt.years))) - set(train_idxs) - set(test_idxs)
                        )
                    else:
                        raise NotImplementedError

                    dt_train = Subset(dt, train_idxs)
                    dt_test = Subset(dt, test_idxs)

                    if len(dt_train) == 0 or len(dt_test) == 0:
                        print(f"not enough samples on fold {fold}")
                        test_rmse = None
                        y_pred = None
                    else:
                        # Get data as one big chunk
                        dl_train = torch.utils.data.DataLoader(
                            dt_train, batch_size=len(dt_train), shuffle=False
                        )
                        data_train = dl_train.__iter__().__next__()
                        X = data_train["climate"].view((len(train_idxs), -1)).numpy()
                        y = list(data_train["target"].values())[0].numpy()

                        dl_test = torch.utils.data.DataLoader(
                            dt_test, batch_size=len(dt_test), shuffle=False
                        )
                        data_test = dl_test.__iter__().__next__()
                        X_test = data_test["climate"].view((len(test_idxs), -1)).numpy()
                        y_test = list(data_test["target"].values())[0].numpy()

                        # Set number of estimators
                        n_tree = max(int(np.sqrt(X_test.shape[1])), 100)

                        # Fit model and evaluate
                        if model == "RF":
                            regr = RandomForestRegressor(
                                n_estimators=n_tree,
                                criterion="squared_error",
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_features=1.0,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                bootstrap=True,
                                oob_score=False,
                                n_jobs=min(os.cpu_count() - 2, 8),
                                random_state=seeds[fold],
                                verbose=0,
                                warm_start=False,
                                ccp_alpha=0.0,
                                max_samples=None,
                            )
                        elif model == "GBM":
                            regr = GradientBoostingRegressor(
                                loss="squared_error",
                                learning_rate=0.1,
                                n_estimators=n_tree,
                                subsample=1.0,
                                criterion="friedman_mse",
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_depth=3,
                                min_impurity_decrease=0.0,
                                init=None,
                                random_state=seeds[fold],
                                max_features=None,
                                alpha=0.9,
                                verbose=0,
                                max_leaf_nodes=None,
                                warm_start=False,
                                validation_fraction=0.1,
                                n_iter_no_change=None,
                                tol=0.0001,
                                ccp_alpha=0.0,
                            )

                        regr.fit(X, y)
                        y_pred = regr.predict(X_test)
                        test_rmse = float(
                            np.sqrt((((y_pred - y_test) * s) ** 2).mean())
                        )
                        test_mae = float(np.abs((y_pred - y_test) * s).mean())

                    results[f"{species_short}:{phase_short}"][f"fold_{str(fold)}"] = (
                        dict(
                            rmse=test_rmse,
                            mae=test_mae,
                            r2=r2_score(y_true=y_test, y_pred=y_pred),
                        )
                    )

                    with open(output_file_metric, "w") as file:
                        file.write(json.dumps(results, indent=4))
