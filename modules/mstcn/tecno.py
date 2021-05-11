import logging
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pytorch_lightning.metrics.utils import _input_format_classification
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
import pytorch_lightning as pl
import numpy as np
import os
import pickle

class TeCNO(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(TeCNO, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model
        self.init_metrics()

        self.criterion_reg = nn.SmoothL1Loss(reduction='mean')
        self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')

        self.val_phase_samples = []
        self.val_tool_samples = []
        self.val_phase_samples_gt = []
        self.val_tool_samples_gt = []

    def init_metrics(self):
        self.train_acc_phase = pl.metrics.regression.MeanAbsoluteError()
        self.train_acc_phase_cls = pl.metrics.Accuracy()
        self.val_acc_phase = pl.metrics.regression.MeanAbsoluteError()
        self.val_acc_phase_cls = pl.metrics.Accuracy()
        self.test_acc_phase = pl.metrics.regression.MeanAbsoluteError()
        self.test_acc_phase_cls = pl.metrics.Accuracy()

        self.train_acc_tool = pl.metrics.regression.MeanAbsoluteError()
        self.train_acc_tool_cls = pl.metrics.Accuracy()
        self.val_acc_tool = pl.metrics.regression.MeanAbsoluteError()
        self.val_acc_tool_cls = pl.metrics.Accuracy()
        self.test_acc_tool = pl.metrics.regression.MeanAbsoluteError()
        self.test_acc_tool_cls = pl.metrics.Accuracy()

    def forward(self, x, p_stem):
        x = x.transpose(2, 1)
        p_stem = p_stem.transpose(2, 1)
        tools, phases = self.model.forward(x, p_stem)
        return tools, phases

    def loss_function(self, p_phase, p_tool, labels_phase, labels_tool):
        # pre [2, d, t]
        # gt [1, t, d]
        num_stage, d, t = p_phase.shape
        pre_phase_reg = p_phase[:, -self.hparams.num_phase:, :]
        pre_phase_cls = p_phase[:, :-self.hparams.num_phase, :].reshape(num_stage, 3, self.hparams.num_phase, t)
        pre_tool_reg = p_tool[:, -self.hparams.num_ins:,:]
        pre_tool_cls = p_tool[:, :-self.hparams.num_ins,:].reshape(num_stage, 3, self.hparams.num_ins, t)

        gt_phase_reg = labels_phase[0, :, -self.hparams.num_phase:]
        gt_phase_cls = labels_phase[0, :, :-self.hparams.num_phase].long()
        gt_tool_reg = labels_tool[0, :, -self.hparams.num_ins:]
        gt_tool_cls = labels_tool[0, :, :-self.hparams.num_ins].long()
        loss_total_tool = 0
        loss_total_phase = 0
        for j in range(num_stage):
            # Phase Loss
            loss_reg_phase = self.criterion_reg(pre_phase_reg[j].transpose(1, 0).float(), gt_phase_reg.float())
            loss_cls_phase = self.criterion_cls(pre_phase_cls[j].permute(2, 0, 1), gt_phase_cls)
            loss_phase = (loss_reg_phase + 0.01 * loss_cls_phase) * self.hparams.num_phase
            # Tool Loss
            loss_reg_tool = self.criterion_reg(pre_tool_reg[j].transpose(1, 0).float(), gt_tool_reg.float())
            loss_cls_tool = self.criterion_cls(pre_tool_cls[j].permute(2, 0, 1), gt_tool_cls)
            loss_tools = (loss_reg_tool + 0.01 * loss_cls_tool) * self.hparams.num_ins
            # automatic balancing
            # precision1 = torch.exp(-self.log_vars[0])
            # loss_phase_l = precision1 * loss_phase + self.log_vars[0]
            # precision2 = torch.exp(-self.log_vars[1])
            # loss_tool_l = precision2 * loss_tools + self.log_vars[1]


            loss_total_tool += loss_tools
            loss_total_phase += loss_phase
        loss_total_tool = loss_total_tool / (num_stage * 1.0)
        loss_total_phase = loss_total_phase / (num_stage * 1.0)

        return loss_total_tool + loss_total_phase

    def log_mae_metrics(self, horizon):
        wMAE = []
        out_MAE = []
        in_MAE = []
        pMAE = []
        eMAE = []
        metric_dict = {}

        for y, t in zip(self.val_tool_samples, self.val_tool_samples_gt):
            y = y.cpu().numpy()*horizon
            t = t.cpu().numpy()*horizon

            outside_horizon = (t == horizon)
            inside_horizon = (t < horizon) & (t > 0)
            anticipating = (y > horizon * .1) & (y < horizon * .9)

            e_anticipating = (t < horizon * .1) & (t > 0)

            wMAE_ins = np.mean([
                np.abs(y[outside_horizon] - t[outside_horizon]).mean(),
                np.abs(y[inside_horizon] - t[inside_horizon]).mean()
            ])
            out_MAE_ins = np.mean([np.abs(y[outside_horizon] - t[outside_horizon]).mean()])
            in_MAE_ins = np.mean([np.abs(y[inside_horizon] - t[inside_horizon]).mean()])
            pMAE_ins = np.abs(y[anticipating] - t[anticipating]).mean()
            eMAE_ins = np.abs(y[e_anticipating] - t[e_anticipating]).mean()

            wMAE.append(wMAE_ins)
            out_MAE.append(out_MAE_ins)
            in_MAE.append(in_MAE_ins)
            pMAE.append(pMAE_ins)
            eMAE.append(eMAE_ins)

        metric_dict["val_WMAE_tool"] = np.mean(wMAE)
        metric_dict["val_pMAE_tool"] = np.mean(pMAE)
        metric_dict["val_out_MAE_tool"] = np.mean(out_MAE)
        metric_dict["val_in_MAE_tool"] = np.mean(in_MAE)
        metric_dict["val_eMAE_tool"] = np.mean(eMAE)

        #################### Phase #######################
        wMAE = []
        out_MAE = []
        in_MAE = []
        pMAE = []
        eMAE = []
        for y, t in zip(self.val_phase_samples, self.val_phase_samples_gt):
            y = y.cpu().numpy()*horizon
            t = t.cpu().numpy()*horizon

            outside_horizon = (t == horizon)
            inside_horizon = (t < horizon) & (t > 0)
            anticipating = (y > horizon * .1) & (y < horizon * .9)

            e_anticipating = (t < horizon * .1) & (t > 0)

            wMAE_ins = np.mean([
                np.abs(y[outside_horizon] - t[outside_horizon]).mean(),
                np.abs(y[inside_horizon] - t[inside_horizon]).mean()
            ])
            out_MAE_ins = np.mean([np.abs(y[outside_horizon] - t[outside_horizon]).mean()])
            in_MAE_ins = np.mean([np.abs(y[inside_horizon] - t[inside_horizon]).mean()])
            pMAE_ins = np.abs(y[anticipating] - t[anticipating]).mean()
            eMAE_ins = np.abs(y[e_anticipating] - t[e_anticipating]).mean()

            wMAE.append(wMAE_ins)
            out_MAE.append(out_MAE_ins)
            in_MAE.append(in_MAE_ins)
            pMAE.append(pMAE_ins)
            eMAE.append(eMAE_ins)
        metric_dict["val_WMAE_phase"] = np.mean(wMAE)
        metric_dict["val_pMAE_phase"] = np.mean(pMAE)
        metric_dict["val_out_MAE_phase"] = np.mean(out_MAE)
        metric_dict["val_in_MAE_phase"] = np.mean(in_MAE)
        metric_dict["val_eMAE_phase"] = np.mean(eMAE)

        self.val_phase_samples = []
        self.val_tool_samples = []
        self.val_phase_samples_gt = []
        self.val_tool_samples_gt = []

        self.log_dict(metric_dict, on_epoch=True, on_step=False)

    def training_step(self, batch, batch_idx):
        stem_tool, stem_phase, tool_label, phase_label, vid, tool_sig, phase_sig = batch
        p_tool, p_phase = self.forward(stem_tool, stem_phase)
        loss_total = self.loss_function(p_phase, p_tool, phase_label, tool_label)

        num_stage, d, t = p_phase.shape
        pre_phase_reg = p_phase[-1, -self.hparams.num_phase:, :].transpose(1, 0)
        pre_phase_cls = p_phase[-1, :-self.hparams.num_phase, :].reshape(3, self.hparams.num_phase, t).permute(2, 0, 1)
        pre_tool_reg = p_tool[-1, -self.hparams.num_ins:, :].transpose(1, 0)
        pre_tool_cls = p_tool[-1, :-self.hparams.num_ins, :].reshape(3, self.hparams.num_ins, t).permute(2, 0, 1)

        gt_phase_reg = phase_label[0, :, -self.hparams.num_phase:]
        gt_phase_cls = phase_label[0, :, :-self.hparams.num_phase]
        gt_tool_reg = tool_label[0, :, -self.hparams.num_ins:]
        gt_tool_cls = tool_label[0, :, :-self.hparams.num_ins]

        self.train_acc_phase(pre_phase_reg, gt_phase_reg)
        self.log("train_acc_phase", self.train_acc_phase, on_epoch=True, on_step=True)

        self.train_acc_phase_cls(pre_phase_cls, gt_phase_cls)
        self.log("train_acc_phase_cls", self.train_acc_phase_cls, on_epoch=True, on_step=True)

        self.train_acc_tool(pre_tool_reg, gt_tool_reg)
        self.log("train_acc_tool", self.train_acc_tool, on_epoch=True, on_step=True)

        self.train_acc_tool_cls(pre_tool_cls, gt_tool_cls)
        self.log("train_acc_tool_cls", self.train_acc_tool_cls, on_epoch=True, on_step=True)

        self.log("train_loss_total", loss_total, prog_bar=True, logger=True, on_epoch=True, on_step=True)

        return loss_total

    def validation_step(self, batch, batch_idx):

        stem_tool, stem_phase, tool_label, phase_label, vid, tool_sig, phase_sig = batch
        p_tool, p_phase = self.forward(stem_tool, stem_phase)
        loss_total = self.loss_function(p_phase, p_tool, phase_label, tool_label)

        num_stage, d, t = p_phase.shape
        pre_phase_reg = p_phase[-1, -self.hparams.num_phase:, :].transpose(1, 0)
        pre_phase_cls = p_phase[-1, :-self.hparams.num_phase, :].reshape(3, self.hparams.num_phase, t).permute(2, 0, 1)
        pre_tool_reg = p_tool[-1, -self.hparams.num_ins:, :].transpose(1, 0)
        pre_tool_cls = p_tool[-1, :-self.hparams.num_ins, :].reshape(3, self.hparams.num_ins, t).permute(2, 0, 1)

        gt_phase_reg = phase_label[0, :, -self.hparams.num_phase:]
        gt_phase_cls = phase_label[0, :, :-self.hparams.num_phase]
        gt_tool_reg = tool_label[0, :, -self.hparams.num_ins:]
        gt_tool_cls = tool_label[0, :, :-self.hparams.num_ins]

        self.val_phase_samples.append(pre_phase_reg)
        self.val_tool_samples.append(pre_tool_reg)
        self.val_phase_samples_gt.append(gt_phase_reg)
        self.val_tool_samples_gt.append(gt_tool_reg)

        self.val_acc_phase(pre_phase_reg, gt_phase_reg)
        self.log("val_acc_phase", self.val_acc_phase, on_epoch=True, on_step=False)

        self.val_acc_phase_cls(pre_phase_cls, gt_phase_cls)
        self.log("val_acc_phase_cls", self.val_acc_phase_cls, on_epoch=True, on_step=False)

        self.val_acc_tool(pre_tool_reg, gt_tool_reg)
        self.log("val_acc_tool", self.val_acc_tool, on_epoch=True, on_step=False)

        self.val_acc_tool_cls(pre_tool_cls, gt_tool_cls)
        self.log("val_acc_tool_cls", self.val_acc_tool_cls, on_epoch=True, on_step=False)

        self.log("val_loss_total", loss_total, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss_total


    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        self.max_acc_last_stage = {"epoch": 0, "acc": 0}
        self.max_acc_global = {"epoch": 0, "acc": 0 , "stage": 0}
        """
        pass
        self.log_mae_metrics(self.hparams.horizon)


    def test_step(self, batch, batch_idx):
        '''
        save the test samples
        :param batch:
        :param batch_idx:
        :return:
        '''
        stem_tool, stem_phase, tool_label, phase_label, vid, tool_sig, phase_sig = batch
        p_tool, p_phase = self.forward(stem_tool, stem_phase)

        pre_phase_reg = p_phase[-1].transpose(1, 0)
        pre_tool_reg = p_tool[-1].transpose(1, 0)

        gt_phase_reg = phase_label[0]
        gt_tool_reg = tool_label[0]

        pre_phase_reg = pre_phase_reg.cpu().numpy()
        pre_tool_reg = pre_tool_reg.cpu().numpy()
        gt_phase_reg = gt_phase_reg.cpu().numpy()
        gt_tool_reg = gt_tool_reg.cpu().numpy()


        save_path = self.pickle_path + "/"
        os.makedirs(save_path, exist_ok=True)
        save_path_vid = save_path + "/" + f"video_{int(vid[0])}.pkl"
        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                pre_phase_reg,
                gt_phase_reg,
                pre_tool_reg,
                gt_tool_reg
            ], f)

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=19, gamma=0.5)
        return [optimizer]#, [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = False
        if split == "train":
            should_shuffle = True
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)
            should_shuffle = False
        print(f"split: {split} - shuffle: {should_shuffle}")
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def set_export_pickle_path(self):
        self.pickle_path = str(self.hparams.output_path) + "/cholec80_pickle_export_"+str(self.hparams.horizon)
        os.makedirs(self.pickle_path, exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(len(dataloader.dataset)))
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        regressiontcn = parser.add_argument_group(
            title='regression tcn specific args options')
        regressiontcn.add_argument("--learning_rate",
                                   default=0.001,
                                   type=float)
        regressiontcn.add_argument("--optimizer_name",
                                   default="adam",
                                   type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)

        return parser
