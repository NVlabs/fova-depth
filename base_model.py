from collections import defaultdict
from typing import Optional

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import rank_zero_only

from models.shared.image_logger import ImageLogger


def apply_to_dict(d, f):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, list):
            new_d[k] = []
            for x in v:
                new_d[k].append(f(x))
        else:
            new_d[k] = f(v)
    return new_d


def combine_outputs(results_dict, only_unique_idx):
    full_dict = {}
    for dataset_num, results in results_dict.items():
        combined_i = combine_outputs_individual(results, only_unique_idx)
        for k, v in combined_i.items():
            full_dict[k + ("_ds%d" % dataset_num)] = v
    return full_dict


def combine_outputs_individual(l, only_unique_idx):
    # input: list of dicts with the same keys. dict values are: tensors of size (N,b) | lists of tensors of size (N,b). Where N is number of devices and b is batch_size
    # output: dict with same keys. Values are tensors (M*N*b) | lists of tensors (M*N*b) where M is the dataset size

    out_dict = {}

    keys = []
    list_keys = []
    list_keys_lens = []

    for k in l[0].keys():
        if isinstance(l[0][k], list):
            list_keys.append(k)
            list_keys_lens.append(len(l[0][k]))
        else:
            keys.append(k)

    for k in keys:
        temp = []
        for m in range(len(l)):
            temp.append(l[m][k].view(-1))
        out_dict[k] = torch.cat(temp)

    for k, k_len in zip(list_keys, list_keys_lens):
        temp = [[] for _ in range(k_len)]
        for nk in range(k_len):
            for m in range(len(l)):
                temp[nk].append(l[m][k][nk].view(-1))

        out_dict[k] = [torch.cat(x) for x in temp]

    if only_unique_idx:
        max_idx = torch.max(out_dict["idx"]) + 1
        has_appeared = torch.zeros(max_idx, dtype=torch.bool)

        idx_of_first_appearence = torch.zeros(len(out_dict["idx"]), dtype=torch.bool)

        for i in range(len(out_dict["idx"])):
            if not has_appeared[out_dict["idx"][i]]:
                idx_of_first_appearence[i] = True
                has_appeared[out_dict["idx"][i]] = True

        out_dict = apply_to_dict(out_dict, lambda t: t[idx_of_first_appearence])

    return out_dict


def write_loss_dict_to_tb(experiment, prefix, log_dict, step):
    for k, v in log_dict.items():
        if k.startswith("weight_"):
            continue
        if k[:3] == "idx":
            continue

        key = prefix + "_" + k
        weight_key = "weight_" + k
        if weight_key in log_dict:
            weight = log_dict[weight_key]
            val = torch.sum(v * weight) / torch.sum(weight)
        else:
            val = torch.mean(v)
        experiment.add_scalar(key, val, global_step=step)


class MainModel(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        optimizer_config: dict,
        step_log_freq: int,
        image_logger: ImageLogger = None,
        test_metrics_processor: Optional[nn.Module] = None,
        load_state_dict: Optional[str] = None,
    ):
        super().__init__()

        self.step_log_freq = step_log_freq
        self.image_logger = image_logger
        self.network = network
        # if compile_model:
        #     self.network = torch.compile(network)
        self.criterion = criterion
        self.optimizer_config = optimizer_config

        self.test_metrics_processor = test_metrics_processor

        self.validation_step_outputs = defaultdict(list)
        self.train_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

        if load_state_dict is not None:
            sd = torch.load(load_state_dict)
            self.load_state_dict(sd["state_dict"], strict=False)

    # don't touch these
    def on_train_epoch_end(self):
        outputs = self.all_gather(self.train_step_outputs)
        self.train_epoch_end_all_devices(outputs)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        outputs = self.all_gather(self.validation_step_outputs)
        self.validation_epoch_end_all_devices(outputs)
        self.validation_step_outputs.clear()
        # if self.trainer.is_global_zero:
        #        print('writing val loss', val_loss)
        #        self.log('val_loss', val_loss, rank_zero_only=True)

    def on_test_epoch_end(self):
        outputs = self.all_gather(self.test_step_outputs)
        self.test_epoch_end_all_devices(outputs)
        self.test_step_outputs.clear()

    def forward(self, net_in):
        net_out = self.network(net_in)
        return net_out

    # return dict with keys: loss and loss_dict
    # loss_dict has values of shape (b,) or list of shape (b,) tensors
    # loss_dict must of key idx for the idx of the sample in the dataset, to avoid repeats in parallel training
    def training_step(self, batch, batch_idx):

        net_out = self.forward(batch)

        loss, loss_dict = self.criterion(net_out, batch)
        loss_dict["idx"] = batch["idx"]

        if self.image_logger:
            self.image_logger(self, batch, batch_idx, net_out, "train")

        if (self.global_step % self.step_log_freq) == 0:
            self.logger.experiment.add_scalar("running_loss", loss, self.global_step)

        self.train_step_outputs[0].append(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        net_out = self.forward(batch)
        loss, loss_dict = self.criterion(net_out, batch)
        loss_dict["idx"] = batch["idx"]

        if self.image_logger:
            self.image_logger(self, batch, batch_idx, net_out, "val_%d" % dataloader_idx)

        self.log(
            "val_loss_monitor_%d" % dataloader_idx, loss, add_dataloader_idx=False, sync_dist=True
        )
        self.validation_step_outputs[dataloader_idx].append(loss_dict)
        return loss

    @rank_zero_only
    def train_epoch_end_all_devices(self, outputs):
        combined = combine_outputs(outputs, False)
        write_loss_dict_to_tb(self.logger.experiment, "train", combined, self.current_epoch)

    @rank_zero_only
    def validation_epoch_end_all_devices(self, outputs):
        combined = combine_outputs(outputs, True)
        write_loss_dict_to_tb(self.logger.experiment, "val", combined, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.optimizer_config["init_lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.optimizer_config["mile_stones"],
            gamma=self.optimizer_config["gamma"],
            verbose=True,
        )
        opt_sche = ([optimizer], [lr_scheduler])
        return opt_sche

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        net_out = self.forward(batch)
        loss_dict = self.test_metrics_processor.step(net_out, batch, self.logger.log_dir)
        loss_dict["idx"] = batch["idx"]

        if self.image_logger:
            self.image_logger(self, batch, batch_idx, net_out, "val_%d" % dataloader_idx)

        self.test_step_outputs[dataloader_idx].append(loss_dict)

    @rank_zero_only
    def test_epoch_end_all_devices(self, outputs):
        combined = combine_outputs(outputs, True)
        self.test_metrics_processor.final_compute(combined, self.logger.log_dir)
