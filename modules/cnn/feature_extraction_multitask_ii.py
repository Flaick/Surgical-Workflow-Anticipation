import logging
import torch
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from pycm import ConfusionMatrix
import numpy as np
import pickle
import os
# from pytorch_lightning.metrics.regression import MeanAbsoluteError

class FeatureExtraction(LightningModule):
    def __init__(self, hparams, model, dataset):
        super(FeatureExtraction, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.num_tasks = self.hparams.num_tasks  # output stem 0, output phase 1 , output phase and tool 2
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.criterion_reg = nn.SmoothL1Loss(reduction='mean')
        self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')
        self.sig_f = nn.Sigmoid()
        self.current_video_idx = self.dataset.df["test"].video_idx.min()
        self.init_metrics()

        self.horizon = hparams.horizon
        self.num_ins = hparams.num_ins
        self.num_phase = hparams.num_phase
        self.num_class = 3

        # store model
        self.current_stems_tool = []
        self.current_stems_phase = []
        self.current_phase_labels = []
        self.current_p_phases = []
        self.phase_sigs = []
        self.current_tool_labels = []
        self.current_p_tools = []
        self.tool_sigs = []

        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.best_metrics_high = {"val_acc_tool": 2} #
        self.test_acc_per_video = {}
        self.pickle_path = None

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



    def set_export_pickle_path(self):
        self.pickle_path = str(self.hparams.output_path) + "/cholec80_pickle_export_"+str(self.horizon)
        os.makedirs(self.pickle_path, exist_ok=True)
        print(f"setting export_pickle_path: {self.pickle_path}")

    # ---------------------
    # TRAINING
    # ---------------------

    def forward(self, x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg):
        stem_tool, stem_phase, tool, phase = self.model.forward(x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg)
        return stem_tool, stem_phase, tool, phase

    def loss_phase_tool(self, p_phase, p_tool, labels_phase, labels_tool):

        bs = p_phase.shape[0]
        pre_phase_reg = p_phase[:, -self.num_phase:]
        pre_phase_cls = p_phase[:, :-self.num_phase].reshape(bs, self.num_class, self.num_phase)
        pre_tool_reg = p_tool[:, -self.num_ins:]
        pre_tool_cls = p_tool[:, :-self.num_ins].reshape(bs, self.num_class, self.num_ins)

        gt_phase_reg = labels_phase[:, -self.num_phase:]
        gt_phase_cls = labels_phase[:, :-self.num_phase].long()
        gt_tool_reg = labels_tool[:, -self.num_ins:]
        gt_tool_cls = labels_tool[:, :-self.num_ins].long()
        # Phase Loss
        loss_reg_phase = self.criterion_reg(pre_phase_reg, gt_phase_reg)
        loss_cls_phase = self.criterion_cls(pre_phase_cls, gt_phase_cls)
        loss_phase = (loss_reg_phase + 0.01*loss_cls_phase) * self.num_phase
        # Tool Loss
        loss_reg_tool = self.criterion_reg(pre_tool_reg, gt_tool_reg)
        loss_cls_tool = self.criterion_cls(pre_tool_cls, gt_tool_cls)
        loss_tools = (loss_reg_tool + 0.01*loss_cls_tool) * self.num_ins
        # automatic balancing
        precision1 = torch.exp(-self.log_vars[0])
        loss_phase_l = precision1 * loss_phase + self.log_vars[0]
        precision2 = torch.exp(-self.log_vars[1])
        loss_tool_l = precision2 * loss_tools + self.log_vars[1]
        loss = loss_phase_l + loss_tool_l
        return loss


    def training_step(self, batch, batch_idx):
        tool_sig, phase_sig, x, y_tool, y_phase, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg = batch
        other_boxes, other_boxes_class, other_boxes_mask = other_boxes.float(), other_boxes_class.float(), other_boxes_mask.float()
        y_phase = y_phase[:, 2:]

        _, _, p_tool, p_phase = self.forward(x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg)

        loss = self.loss_phase_tool(p_phase, p_tool, y_phase, y_tool)

        bs = p_phase.shape[0]
        pre_phase_reg = p_phase[:, -self.num_phase:]
        pre_phase_cls = p_phase[:, :-self.num_phase].reshape(bs, self.num_class, self.num_phase)
        pre_tool_reg = p_tool[:, -self.num_ins:]
        pre_tool_cls = p_tool[:, :-self.num_ins].reshape(bs, self.num_class, self.num_ins)

        gt_phase_reg = y_phase[:, -self.num_phase:]
        gt_phase_cls = y_phase[:, :-self.num_phase]
        gt_tool_reg = y_tool[:, -self.num_ins:]
        gt_tool_cls = y_tool[:, :-self.num_ins]

        self.train_acc_phase(pre_phase_reg, gt_phase_reg)
        self.log("train_acc_phase", self.train_acc_phase, on_epoch=True, on_step=True)

        self.train_acc_phase_cls(pre_phase_cls, gt_phase_cls)
        self.log("train_acc_phase_cls", self.train_acc_phase_cls, on_epoch=True, on_step=True)

        self.train_acc_tool(pre_tool_reg, gt_tool_reg)
        self.log("train_acc_tool", self.train_acc_tool, on_epoch=True, on_step=True)

        self.train_acc_tool_cls(pre_tool_cls, gt_tool_cls)
        self.log("train_acc_tool_cls", self.train_acc_tool_cls, on_epoch=True, on_step=True)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        tool_sig, phase_sig, x, y_tool, y_phase, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg = batch
        other_boxes, other_boxes_class, other_boxes_mask = other_boxes.float(), other_boxes_class.float(), other_boxes_mask.float()
        y_phase = y_phase[:, 2:]

        _, _, p_tool, p_phase = self.forward(x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg)
        p_phase, p_tool, y_phase, y_tool = p_phase.detach(), p_tool.detach(), y_phase.detach(), y_tool.detach()
        loss = self.loss_phase_tool(p_phase, p_tool, y_phase, y_tool)

        bs = p_phase.shape[0]
        pre_phase_reg = p_phase[:, -self.num_phase:]
        pre_phase_cls = p_phase[:, :-self.num_phase].reshape(bs, self.num_class, self.num_phase)
        pre_tool_reg = p_tool[:, -self.num_ins:]
        pre_tool_cls = p_tool[:, :-self.num_ins].reshape(bs, self.num_class, self.num_ins)

        gt_phase_reg = y_phase[:, -self.num_phase:]
        gt_phase_cls = y_phase[:, :-self.num_phase]
        gt_tool_reg = y_tool[:, -self.num_ins:]
        gt_tool_cls = y_tool[:, :-self.num_ins]

        self.val_acc_phase(pre_phase_reg, gt_phase_reg)
        self.log("val_acc_phase", self.val_acc_phase, on_epoch=True, on_step=False)

        self.val_acc_phase_cls(pre_phase_cls, gt_phase_cls)
        self.log("val_acc_phase_cls", self.val_acc_phase_cls, on_epoch=True, on_step=False)

        self.val_acc_tool(pre_tool_reg, gt_tool_reg)
        self.log("val_acc_tool", self.val_acc_tool, on_epoch=True, on_step=False)

        self.val_acc_tool_cls(pre_tool_cls, gt_tool_cls)
        self.log("val_acc_tool_cls", self.val_acc_tool_cls, on_epoch=True, on_step=False)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def save_to_drive(self, vid_index):
        save_path = self.pickle_path + "/" + f"{self.hparams.fps_sampling_test}fps"
        os.makedirs(save_path, exist_ok=True)

        save_path_vid = save_path + "/" + f"video_{vid_index}_{self.hparams.fps_sampling_test}fps.pkl"
        with open(save_path_vid, 'wb') as f:
            pickle.dump([
                np.asarray(self.current_stems_tool),
                np.asarray(self.current_stems_phase),
                np.asarray(self.current_p_phases),
                np.asarray(self.current_phase_labels),
                np.asarray(self.phase_sigs),
                np.asarray(self.current_p_tools),
                np.asarray(self.current_tool_labels),
                np.asarray(self.tool_sigs)
            ], f)

    def test_step(self, batch, batch_idx):
        tool_sig, phase_sig, x, y_tool, y_phase, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg = batch
        other_boxes, other_boxes_class, other_boxes_mask = other_boxes.float(), other_boxes_class.float(), other_boxes_mask.float()
        vid_idx = y_phase[:,0:1]
        img_index = y_phase[:,1:2]
        y_phase = y_phase[:,2:]
        # y_tool [bs, 5]  y_phase [bs, 7]

        vid_idx_raw = vid_idx.cpu().numpy()
        with torch.no_grad():
            stem_tool, stem_phase, p_tool, p_phase = self.forward(x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg) # [bs, 2048] [bs, 7] [bs, 5]

        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(x) for x in vid_idxs]
        index_next = len(vid_idx) if len(vid_idxs) == 1 else indexes[1]
        for i in range(len(vid_idxs)): # 80 videos
            vid_idx = vid_idxs[i]
            index = indexes[i] # Starting index of this video
            if vid_idx != self.current_video_idx:
                self.save_to_drive(self.current_video_idx)

                self.current_stems_tool = []
                self.current_stems_phase = []

                # Regression for Phase
                self.current_phase_labels = []
                self.current_p_phases = []
                self.phase_sigs = []

                # Regression for Tool
                self.current_tool_labels = []
                self.current_p_tools = []
                self.tool_sigs = []

                if len(vid_idxs) <= i + 1:
                    index_next = len(vid_idx_raw)
                else:
                    index_next = indexes[i+1]  # for the unlikely case that we have 3 phases in one batch
                self.current_video_idx = vid_idx


            self.current_stems_tool.extend(stem_tool[index:index_next, :].cpu().detach().numpy().tolist())
            self.current_stems_phase.extend(stem_phase[index:index_next, :].cpu().detach().numpy().tolist())


            # Regression for Phase
            p_phase_ = np.asarray(p_phase.cpu()).squeeze()
            self.current_p_phases.extend(
                np.asarray(p_phase_[index:index_next, :]).tolist())

            y_phase_ = y_phase.cpu().numpy()
            self.current_phase_labels.extend(
                np.asarray(y_phase_[index:index_next]).tolist())
            phase_sig_ = phase_sig.cpu().numpy()
            self.phase_sigs.extend(
                np.asarray(phase_sig_[index:index_next]).tolist())


            # Regression for Tool
            p_tool_ = np.asarray(p_tool.cpu()).squeeze()
            self.current_p_tools.extend(
                np.asarray(p_tool_[index:index_next, :]).tolist())

            y_tool_ = y_tool.cpu().numpy()
            self.current_tool_labels.extend(
                np.asarray(y_tool_[index:index_next]).tolist())

            tool_sig_ = tool_sig.cpu().numpy()
            self.tool_sigs.extend(
                np.asarray(tool_sig_[index:index_next]).tolist())

        if (batch_idx + 1) * self.hparams.batch_size >= self.len_test_data:
            self.save_to_drive(vid_idx)
            print(f"Finished extracting all videos...")



    def test_epoch_end(self, outputs):
        pass
        # self.log("test_acc_train", np.mean(np.asarray([self.test_acc_per_video[x]for x in
        #                                                self.dataset.vids_for_training])))
        # self.log("test_acc_val", np.mean(np.asarray([self.test_acc_per_video[x]for x in
        #                                              self.dataset.vids_for_val])))
        # self.log("test_acc_test", np.mean(np.asarray([self.test_acc_per_video[x] for x in
        #                                               self.dataset.vids_for_test])))
        # self.log("test_acc", float(self.test_acc_phase.compute()))
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
        return [optimizer], [scheduler]

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        if self.hparams.batch_size > self.hparams.model_specific_batch_size_max:
            print(
                f"The choosen batchsize is too large for this model."
                f" It got automatically reduced from: {self.hparams.batch_size} to {self.hparams.model_specific_batch_size_max}"
            )
            self.hparams.batch_size = self.hparams.model_specific_batch_size_max

        if split == "val" or split == "test":
            should_shuffle = False
        else:
            should_shuffle = True
        print(f"split: {split} - shuffle: {should_shuffle}")
        worker = self.hparams.num_workers
        if split == "test":
            print(
                "worker set to 0 due to test"
            )  # otherwise for extraction the order in which data is loaded is not sorted e.g. 1,2,3,4, --> 1,5,3,2
            worker = 0

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            num_workers=worker,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        """
        Intialize train dataloader
        :return: train loader
        """
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        """
        Initialize val loader
        :return: validation loader
        """
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(len(dataloader.dataset)))
        return dataloader


    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(len(dataloader.dataset)))
        print(f"starting video idx for testing: {self.current_video_idx}")
        self.set_export_pickle_path()
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        cholec_fe_module = parser.add_argument_group(
            title='cholec_fe_module specific args options')
        cholec_fe_module.add_argument("--learning_rate",
                                      default=0.001,
                                      type=float)
        cholec_fe_module.add_argument("--num_tasks",
                                      default=2,
                                      type=int,
                                      choices=[1, 2])
        cholec_fe_module.add_argument("--optimizer_name",
                                      default="adam",
                                      type=str)
        cholec_fe_module.add_argument("--batch_size", default=32, type=int)
        return parser
