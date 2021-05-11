import pandas as pd
from torch.utils.data import Dataset
import pprint, pickle
from pathlib import Path
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
import torch
import json
import random
import cv2
import torch.nn.functional as F
class Cholec80FeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.fps_sampling = hparams.fps_sampling
        self.fps_sampling_test = hparams.fps_sampling_test
        self.cholec_root_dir = Path(self.hparams.data_root)  # videos splitted in images
        self.transformations = self.__get_transformations()
        self.horizon = hparams.horizon

        self.label_col = [
            "Bipolar_cls", "Scissors_cls", "Clipper_cls", "Irrigator_cls", "SpecimenBag_cls",
            "Bipolar_reg", "Scissors_reg", "Clipper_reg", "Irrigator_reg", "SpecimenBag_reg"
        ]
        self.phase_label_col = ["video_idx", "index",
            "P1_cls", "P2_cls", "P3_cls", "P4_cls", "P5_cls", "P6_cls", "P7_cls",
            "P1_reg", "P2_reg", "P3_reg", "P4_reg", "P5_reg", "P6_reg", "P7_reg"
        ]
        self.df = {}
        print(str(self.cholec_root_dir) + "/dataframes/cholec_split_250px_25fps_anni_"+str(self.horizon)+".pkl")

        # For Windows python3.6
        import pickle5 as pickle
        with open(str(self.cholec_root_dir) + "/dataframes/cholec_split_250px_25fps_anni_"+str(self.horizon)+".pkl", "rb") as fh:
            self.df["all"] = pickle.load(fh)
        # For Linux python 3.8
        # self.df["all"] = pd.read_pickle(str(self.cholec_root_dir) + "/dataframes/cholec_split_250px_25fps_anni_"+str(self.horizon)+".pkl")

        #print("Drop nan rows from df manually")
        ## Manually remove these indices as they are nan in the DF which causes issues
        index_nan = [1983913, 900090]
        #self.df["all"][self.df["all"].isna().any(axis=1)]
        self.df["all"] = self.df["all"].drop(index_nan)
        # print(self.df["all"].isnull())
        assert self.df["all"].isnull().sum().sum() == 0, "Dataframe contains nan Elements"
        self.df["all"] = self.df["all"].reset_index()

        sub1 = ['02','04','06','12','24','29','34','37','38','39','44','58','60','61','64','66','75','78','79','80']
        sub2 = ['01','03','05','09','13','16','18','21','22','25','31','36','45','46','48','50','62','71','72','73']
        sub3 = ['10','15','17','20','32','41','42','43','47','49','51','52','53','55','56','69','70','74','76','77']
        sub4 = ['07','08','11','14','19','23','26','27','28','30','33','35','40','54','57','59','63','65','67','68']
        train_ids = sub1 + sub2 + sub3
        val_test_ids = sub4
        self.vids_for_training = [int(i) for i in train_ids]
        self.vids_for_val = [int(i) for i in val_test_ids]
        self.vids_for_test = self.vids_for_val

        self.df["train"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_training)]
        self.df["val"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_val)]
        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.vids_for_test = [i for i in range(1, 81)]
            self.df["test"] = self.df["all"]
        else:
            self.df["test"] = self.df["all"][self.df["all"]["video_idx"].isin(self.vids_for_test)]

        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"])
        }
        if self.fps_sampling < 25 and self.fps_sampling > 0:
            factor = int(25 / self.fps_sampling)
            print(
                f"Subsampling(factor: {factor}) data: 25fps > {self.fps_sampling}fps"
            )
            self.df["train"] = self.df["train"].iloc[::factor]
            self.df["val"] = self.df["val"].iloc[::factor]
            self.df["all"] = self.df["all"].iloc[::factor]
            for split in ["train", "val"]:
                print(
                    f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")
        if hparams.fps_sampling_test < 25 and self.fps_sampling_test > 0:
            factor = int(25 / self.fps_sampling_test)
            print(
                f"Subsampling(factor: {factor}) data: 25fps > {self.fps_sampling}fps"
            )
            self.df["test"] = self.df["test"].iloc[::factor]
            split = "test"
            print(f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")

        self.data = {}

        if self.dataset_mode == "img_multilabel":
            for split in ["train", "val"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                    image_path_col="image_path",
                    add_label_cols=self.phase_label_col)
            # here we want to extract all features
            self.df["test"] = self.df["test"].reset_index()
            self.data["test"] = Dataset_from_Dataframe(
                self.df["test"],
                self.transformations["test"],
                self.label_col,
                img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                image_path_col="image_path",
                add_label_cols=self.phase_label_col)
        else:
            print('Not Implement dataset_mode, only img_multilabel supported')
            exit()
    def __get_transformations(self):
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    def median_frequency_weights(
            self, file_list):  ## do only once and define weights in class
        frequency = [0, 0, 0, 0, 0, 0, 0]
        for i in file_list:
            frequency[int(i[1])] += 1
        median = np.median(frequency)
        weights = [median / j for j in frequency]
        return weights

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 specific args options')
        cholec80_specific_args.add_argument("--fps_sampling",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument("--fps_sampling_test",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument(
            "--dataset_mode",
            default='video',
            choices=[
                'vid_multilabel', 'img', 'img_multilabel',
                'img_multilabel_feature_extract'
            ])
        cholec80_specific_args.add_argument("--test_extract",
                                            action="store_true")
        return parser


class Dataset_from_Dataframe(Dataset):
    def __init__(self,
                 df,
                 transform,
                 label_col=[],
                 img_root="",
                 image_path_col="path",
                 add_label_cols=[]):
        self.annotation_path_geo = "/home/kun/Desktop/miccai21/data/annotations/other_box_seq_list.json"  # Geo feature _5_5
        self.annotation_path_geo_class = "/home/kun/Desktop/miccai21/data/annotations/other_box_class_seq_list.json"  # Geo feature _5_5


        self.df = df
        self.transform = transform

        self.label_col = label_col

        self.image_path_col = image_path_col
        self.img_root = img_root

        self.add_label_cols = add_label_cols

        with open(self.annotation_path_geo) as f:
            self.geo_list = json.load(f)
        with open(self.annotation_path_geo_class) as f:
            self.geo_class_list = json.load(f)

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = self.img_root / img_path_df
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        X_array, p_id = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]

        label = []
        for l in self.label_col:
            label.append(self.df[l][index])
        label = torch.Tensor(label)

        add_label = []
        for add_l in self.add_label_cols:
            add_label.append(self.df[add_l][index])
        add_label = torch.Tensor(add_label)

        # print(add_label.shape, label.shape)
        X = X.type(torch.FloatTensor)

        '''
        Tool sig [N, 7]
        '''
        tool_sig = []
        tool_cols = ["tool_Grasper_yolo", "tool_Bipolar_yolo", "tool_Hook_yolo", "tool_Scissors_yolo", "tool_Clipper_yolo", "tool_Irrigator_yolo", "tool_SpecimenBag_yolo"]
        # tool_cols = ["tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"]
        for t in tool_cols:
            tool_sig.append(self.df[t][index])
        tool_sig = torch.LongTensor(tool_sig)

        '''
        Phase sig [N, 7]
        '''
        phase_sig = []
        phase_cols = ["P1_tecno", "P2_tecno", "P3_tecno", "P4_tecno", "P5_tecno", "P6_tecno", "P7_tecno"]
        for p in phase_cols:
            phase_sig.append(self.df[p][index])
        phase_sig = torch.LongTensor(phase_sig)


        '''
        Instrument-Instrument & Instrument-Surrounding Features
        '''
        v_id = str(p_id).split('/')[-1].split('_')[0]
        f_id = int(str(p_id).split('_')[-1].split('.png')[0])//25
        this_geo_list = self.geo_list[v_id]  # (seq_len, K, 4) K is variable length
        this_geo_class_list = self.geo_class_list[v_id]  # (seq_len, K) K is variable length
        # print(len(this_geo_list), v_id)
        # Top K Features
        K = 2  # max_other boxes
        other_boxes_mask = np.zeros(shape=(K,), dtype=np.bool)
        other_boxes = np.zeros(shape=(K, 4), dtype=np.float)
        other_boxes_class = np.zeros(shape=(K, 7), dtype=np.float)

        this_other_boxes = this_geo_list[f_id]
        this_other_boxes_class = this_geo_class_list[f_id]

        other_box_idxs = list(range(len(this_other_boxes)))

        random.shuffle(other_box_idxs)

        other_box_idxs = other_box_idxs[:K]

        for k, idx in enumerate(other_box_idxs):
            other_boxes_mask[k] = True
            other_boxes[k, :] = torch.Tensor(this_other_boxes[idx])

            # one-hot representation
            box_class = this_other_boxes_class[idx]
            other_boxes_class[k, int(box_class)] = 1
        assert np.max(other_boxes_class[:, 0]) == 0

        '''
        Segmentation Map
        '''
        seg_name = '/home/kun/Desktop/miccai21/TeCNO/Videos/cholec_split_250px_25fps_masks/'+v_id+'/'+'%08d.png'%f_id
        seg_img = torch.from_numpy(cv2.imread(seg_name, 0))

        one_hot_seg = torch.zeros((7, seg_img.shape[0], seg_img.shape[1]))
        for i, unique_value in enumerate(torch.unique(seg_img)):
            one_hot_seg[i, :, :][seg_img == unique_value] = 1
        one_hot_seg = F.interpolate(one_hot_seg.unsqueeze(0), size=(224, 224), mode='nearest').squeeze(0)
        assert one_hot_seg.shape[1] == X.shape[1]
        assert one_hot_seg.shape[2] == X.shape[2]
        # other_boxes, other_boxes_class, other_boxes_mask = other_boxes, other_boxes_class, other_boxes_mask
        # print(other_boxes.shape, other_boxes_class.shape, other_boxes_mask.shape)

        return tool_sig, phase_sig, X, label, add_label, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg




