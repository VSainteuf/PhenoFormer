import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from configs.PROBLEM_CONFIG import target_list_parser, target_shorter, seeds
from dataset import ClimatePhenoDataset, get_matching_indices
from model.architecture import PhenoFormer


class LitModel(pl.LightningModule):
    """PyTorch Lighnting module used as wrapper for the PhenoFormer model
    This module implements the functions that are used for model training
    and metric computations.
    """

    def __init__(
        self,
        backbone,
        target_scaler,
        args,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.target_scaler = target_scaler
        self.target_list = backbone.target_list
        self.nan_value_target = args.nan_value_target
        self.args = args

        if args.loss == "L2":
            self.loss_func = F.mse_loss
        elif args.loss == "L1":
            self.loss_func = F.l1_loss
        else:
            raise

        self.meters = {
            "rmse": {t: MeanSquaredError(squared=False) for t in self.target_list},
            "r2": {t: R2Score() for t in self.target_list},
            "mae": {t: MeanAbsoluteError() for t in self.target_list},
        }

    def forward(self, batch):
        y_hat = self.backbone(batch)
        return y_hat
    
    def predict_unnormalised_dates(self, batch):
        y_hat = self.forward(batch=batch)
        for t in self.target_list:
            normalised_prediction = y_hat[t]
            mean, std = self.target_scaler[t]
            unnormalised_prediction = normalised_prediction * std + mean
            y_hat[t] = unnormalised_prediction
        return y_hat

    def meter_forward(self, predictions, targets, select_dim=None):
        values = {
            m: {} for m in self.meters.keys() if m != "r2"
        }  # computing r2 on batch doesn't make sense
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]
            valid = true != self.nan_value_target
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            preds = preds * scaler
            true = true * scaler
            for m in self.meters.keys():
                if m == "r2":
                    self.meters[m][t].update(preds[valid], true[valid])
                else:
                    values[m][t] = self.meters[m][t](preds[valid], true[valid])
        return values

    def meter_update(self, predictions, targets, select_dim=None):
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]
            valid = true != self.nan_value_target
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            preds = preds * scaler
            true = true * scaler
            for m in self.meters.keys():
                self.meters[m][t].update(preds[valid], true[valid])

    def meter_compute(self):
        values = {m: {} for m in self.meters.keys()}
        for t in self.target_list:
            for m in self.meters.keys():
                try:
                    values[m][t] = self.meters[m][t].compute()
                except ValueError:
                    values[m][t] = torch.tensor(torch.nan, device=self.device)
        return values

    def meter_reset(self):
        for t in self.target_list:
            for m in self.meters.keys():
                self.meters[m][t].reset()

    def multi_apply(
        self, function, predictions, targets, rescale=False, select_dim=None
    ):
        values = {}
        for t in self.target_list:
            preds = (
                predictions[t] if select_dim is None else predictions[t][:, select_dim]
            )
            true = targets[t]
            scaler = (
                self.target_scaler[t][1]
                if isinstance(self.target_scaler, dict)
                else self.target_scaler
            )
            valid = true != self.nan_value_target
            if rescale:
                values[t] = function(preds[valid], true[valid], scaler=scaler)
            else:
                values[t] = function(preds[valid], true[valid])
        return values

    def compute_batch_metrics(self, predictions, targets, prefix=""):
        out = {}
        meter_output = self.meter_forward(predictions, targets)
        for metric_name, metric_vals in meter_output.items():
            metric_vals = {
                f"{prefix}/{metric_name}_{target_shorter(k)}": v
                for k, v in metric_vals.items()
            }
            metric_vals[f"{prefix}/{metric_name}"] = torch.stack(
                [v for v in metric_vals.values() if not torch.isnan(v)]
            ).mean()
            out = {**out, **metric_vals}
        return out

    def compute_epoch_metrics(self, prefix=""):
        out = {}
        meter_output = self.meter_compute()
        for metric_name, metric_vals in meter_output.items():
            metric_vals = {
                f"{prefix}/{metric_name}_{target_shorter(k)}": v
                for k, v in metric_vals.items()
            }
            metric_vals[f"{prefix}/{metric_name}"] = torch.stack(
                [v for v in metric_vals.values() if not torch.isnan(v)]
            ).mean()
            out = {**out, **metric_vals}
        return out

    def compute_loss(self, predictions, targets, prefix=""):
        loss_vals = self.multi_apply(self.loss_func, predictions, targets)
        loss_vals = {
            f"{prefix}/loss_{target_shorter(k)}": v for k, v in loss_vals.items()
        }

        loss_vals[f"{prefix}/loss"] = torch.stack(
            [v for v in loss_vals.values() if not torch.isnan(v)]
        ).mean(0)
        return loss_vals

    def log_dictionary(self, dct, **kwargs):
        for name, value in dct.items():
            if not torch.isnan(value):
                self.log(name, value, kwargs)

    def on_fit_start(
        self,
    ):
        if self.device.type == "cuda":
            for m, d in self.meters.items():
                for t in self.target_list:
                    d[t].cuda()

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch=batch)
        y = batch["target"]
        losses = self.compute_loss(y_hat, y, prefix="train")
        metrics = self.compute_batch_metrics(y_hat, y, prefix="train")
        self.log_dictionary({**losses, **metrics}, on_step=True)
        loss = losses["train/loss"]
        if torch.isnan(loss):
            print("nan")
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch=batch)
        y = batch["target"]
        losses = self.compute_loss(y_hat, y, prefix="val")
        self.meter_update(y_hat, y)
        self.log_dictionary(losses, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch=batch)
        y = batch["target"]
        losses = self.compute_loss(y_hat, y, prefix="test")
        self.meter_update(y_hat, y)
        self.log_dictionary(losses, on_epoch=True)

    def on_validation_epoch_start(self):
        if self.global_step > 0:
            metrics = self.compute_epoch_metrics(prefix="train")
            self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

    def on_validation_epoch_end(self):
        metrics = self.compute_epoch_metrics(prefix="val")
        self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

    def on_test_epoch_end(self):
        metrics = self.compute_epoch_metrics(prefix="test")
        self.log_dictionary(metrics, on_epoch=True)
        self.meter_reset()

    def on_test_epoch_start(self):
        self.meter_reset()

    def on_test_epoch_start(self):
        self.meter_reset()

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if self.args.optim == "adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd
            )
        elif self.args.optim == "adamw":
            return torch.optim.AdamW(
                self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd
            )
        else:
            raise "Unkown optimizer"


def get_parser():
    # ------------
    # The present script is used to run all the different configurations
    # of PhenoFormer presented in the paper. The following menu enables
    # the selection of a given model and data configuration
    # ------------
    parser = ArgumentParser()

    # Dataset and data splitting args
    parser.add_argument("--data_folder", default=None, type=str, help="path to the dataset folder")
    parser.add_argument("--target", type=str, default=None, help="target to predict (see target_list_parser in configs/PROBLEM_CONFIG.py for details)")
    parser.add_argument("--nan_value_target", type=int, default=-1, help="value used to represent missing target values")
    parser.add_argument("--nan_value_climate", type=int, default=0, help="value used to represent missing climate values")
    parser.add_argument("--sigma_jitter", default=0, type=float, help="standard deviation of the gaussian noise added to the input climate data")
    parser.add_argument("--split_mode", default="structured", type=str, help="mode used to split the dataset (see datasplit_configs in configs/RUN_CONFIGS.py for details)")
    parser.add_argument("--train_years_to", default=2002, type=int, help="last year of the training set (used in `structured` split mode)")
    parser.add_argument("--val_years_to", default=2012, type=int, help="last year of the validation set (used in `structured` split mode)")
    parser.add_argument("--input_phases", default=None, type=str, help="list of phenophases used as input (used of Variant f of PhenoFormer)")

    # Model args
    parser.add_argument("--nhead", default=8, type=int, help="number of heads in the transformer layer of PhenoFormer")
    parser.add_argument("--d_model", default=64, type=int, help="dimension of the inner representations of PhenoFormer")
    parser.add_argument("--n_layers", default=1, type=int, help="number of stacked attention layers in PhenoFormer")
    parser.add_argument("--dim_feedforward", default=128, type=int, help="number of neurons in the feedforward layer of PhenoFormer")
    parser.add_argument("--T_pos_enc", default=1000, type=int, help="maximal period used in the positional encoding of PhenoFormer")
    parser.add_argument("--elevation", dest="elevation", action="store_true", help="(flag) if set the elevation of the observation site is used as additional input")
    parser.add_argument("--latlon", dest="latlon", action="store_true", help="(flag) if set the latitude and longitude of the observation site are used as additional input")

    # Training args
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--optim", type=str, default="adam", help="optimizer")
    parser.add_argument("--loss", default="L2", type=str, help="loss function (L1 or L2)")
    parser.add_argument("--grad_clip", default=0, type=float, help="gradient clipping value")
    parser.add_argument("--cross_val_id", default=None, type=str,help="cross validation id, the same id will be used for all"
    " folds of a given configuration. This enables to easily compute cross-fold average performance.") 
    parser.add_argument("--fold", default=None, type=int, help="fold number")

    # Output and logging args
    parser.add_argument("--save_dir", default="./training_logs", type=str)
    parser.add_argument("--model_tag", type=str, default="v0", help="model tag (used for logging)")
    parser.add_argument("--config_tag", type=str, default=None, help="config tag (used for logging)")
    parser.add_argument("--task_tag", type=str, default=None, help="task tag (single/multi species) (used for logging)" )
    parser.add_argument("--run_tag", type=str, default=None, help="run tag (used for logging)")
    parser.add_argument("--xp_name", default=None, type=str, help="experiment name (used for logging)")

    parser.set_defaults(
         elevation=False, latlon=False,
    )
    parser = pl.Trainer.add_argparse_args(parser)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Set the random seed (the random seed used for each fold is fixed
    #  across configurations but different between folds)
    if args.fold is not None:
        pl.seed_everything(seeds[args.fold])
    else:
        pl.seed_everything(1234)

    # ------------
    # Dataset 
    # ------------
    dt_base_args = dict(
        folder=args.data_folder,
        target_list=target_list_parser(args.target),
        normalise_climate=True,
        normalise_dates=True,
        nan_value_climate=args.nan_value_climate,
        nan_value_target=args.nan_value_target,
        phases_as_input=target_list_parser(args.input_phases),
    )
    # Dataset without augmentation
    dt = ClimatePhenoDataset(**dt_base_args)
    # Dataset with augmentation
    dt_augm = ClimatePhenoDataset(**dt_base_args, sigma_jitter=args.sigma_jitter)

    # ------------
    # Dataset splitting
    # ------------
    if args.split_mode.endswith(".json"):
        with open(args.split_mode) as file:
            split = json.loads(file.read())
        train_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["train"])
        val_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["val"])
        test_idxs = get_matching_indices(dt.site_years, split[str(args.fold)]["test"])
    elif args.split_mode == "structured":
        train_idxs = list(
            np.where(np.array(dt.years).astype(int) <= args.train_years_to)[0]
        )
        test_idxs = list(
            np.where(np.array(dt.years).astype(int) > args.val_years_to)[0]
        )
        val_idxs = list(set(range(len(dt.years))) - set(train_idxs) - set(test_idxs))
    else:
        raise "Unknown split mode"

    train_loader = DataLoader(
        Subset(dt_augm, train_idxs),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dt, val_idxs),
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(dt, test_idxs),
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )

    # ------------
    # model
    # ------------
    backbone = PhenoFormer(
        target_list=dt.target_list,
        d_in=len(dt.var_names),
        d_out=1,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        n_layers=args.n_layers,
        elevation=args.elevation,
        latlon=args.latlon,
        T_pos_enc=args.T_pos_enc,
        phases_as_input=target_list_parser(args.input_phases),
    )

    model = LitModel(backbone=backbone, target_scaler=dt.target_scaler, args=args)

    # ------------
    # training
    # ------------

    tags = [
        f"target/{args.target}",
        f"model/{args.model_tag}",
        f"run/{args.run_tag}",
        f"task/{args.task_tag}",
        f"config/{args.config_tag}",
    ]
    if args.fold is not None:
        tags.append(f"fold_{args.fold}")

    if args.cross_val_id is not None:
        name = f"{args.cross_val_id}_F{args.fold}"
    else:
        if args.xp_name is None:
            raise "If not running a N-fold cv, please specify the --xp_name argument"
        name = args.xp_name

    wandb_logger = WandbLogger(name=name, save_dir=args.save_dir, offline=True)
    logger = CSVLogger(save_dir=args.save_dir, name=name)

    monitor_metric = "val/rmse"
    monitor_mode = "min"
    early_stop = EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=30)
    ckpt = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor=monitor_metric, mode=monitor_mode
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        callbacks=[early_stop, ckpt],
        log_every_n_steps=5,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clip,
    )
    print(model)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(
        model=model, dataloaders=test_loader, ckpt_path=ckpt.best_model_path
    )
    print(result)

    metrics = wandb_logger.experiment.summary
    metrics_dict = dict(metrics)

    output = {**vars(args), **metrics_dict}

    with open(
        Path(args.save_dir) / f"run_summary_fold{args.fold}.json", "w"
    ) as json_file:
        json.dump(output, json_file, indent=4)
