import pandas as pd
from torch.utils.data import Dataset
import pprint, pickle
from pathlib import Path
import numpy as np
# from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
# from albumentations import (
#     Compose,
#     Resize,
#     Normalize,
#     ShiftScaleRotate,
# )
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
tool_label_col = [
    "Bipolar_cls", "Scissors_cls", "Clipper_cls", "Irrigator_cls", "SpecimenBag_cls"
    # "Bipolar_reg", "Scissors_reg", "Clipper_reg", "Irrigator_reg", "SpecimenBag_reg"
]
phase_label_col = ["video_idx", "index",
    "P1_cls", "P2_cls", "P3_cls", "P4_cls", "P5_cls", "P6_cls", "P7_cls"
    # "P1_reg", "P2_reg", "P3_reg", "P4_reg", "P5_reg", "P6_reg", "P7_reg"
]
df = {}

# For Windows python3.6
# import pickle5 as pickle
# with open("/home/kun/Desktop/miccai21/TeCNO/Videos/dataframes/cholec_split_250px_25fps_anni_3.pkl", "rb") as fh:
#     df["all"] = pickle.load(fh)
# For Linux python 3.8
df["all"] = pd.read_pickle("./data/dataframes/cholec_split_250px_25fps_anni_5.pkl")
print(df["all"].keys())

#print("Drop nan rows from df manually")
## Manually remove these indices as they are nan in the DF which causes issues
index_nan = [1983913, 900090]
#df["all"][df["all"].isna().any(axis=1)]
df["all"] = df["all"].drop(index_nan)
assert df["all"].isnull().sum().sum() == 0, "Dataframe contains nan Elements"
df["all"] = df["all"].reset_index()
# Tool sig [N, 7]
tool_sig = []
tool_gt_sig = []
ids = []
# tool_cols = ["video_idx", "index", "tool_Grasper_yolo", "tool_Bipolar_yolo", "tool_Hook_yolo", "tool_Scissors_yolo", "tool_Clipper_yolo", "tool_Irrigator_yolo", "tool_SpecimenBag_yolo"]
# tool_gt_cols = ["video_idx", "index", "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"]
is_phase = False

tool_cols = ["video_idx", "index", "P1_tecno", "P2_tecno", "P3_tecno", "P4_tecno", "P5_tecno", "P6_tecno", "P7_tecno"]
tool_gt_cols = ["video_idx", "index", "class"]
is_phase = True

for t in tool_cols[2:]:
    tool_sig.append(df["all"][t])
for t in tool_gt_cols[2:]:
    tool_gt_sig.append(df["all"][t])
for t in tool_cols[0:1]:
    ids.append(df["all"][t])
tool_sig = np.asarray(tool_sig).transpose(1, 0).tolist()
tool_gt_sig = np.asarray(tool_gt_sig).transpose(1, 0).tolist()
print(tool_gt_sig.shape)
exit()
ids = np.asarray(ids).transpose(1, 0).tolist()


plt.rcParams["figure.figsize"] = (40, 20)
sample = []
sample_gt = []
current_id = 1
for idx, (t, t_gt) in enumerate(zip(tool_sig, tool_gt_sig)):
    if current_id != ids[idx][0]:
        sample = np.asarray(sample).transpose(1, 0)
        sample_gt = np.asarray(sample_gt).transpose(1, 0)
        print(sample.shape, sample_gt.shape)
        for row in range(7):
            plt.subplot(7, 2, row*2+1)
            x = np.arange(sample[row].shape[0], dtype=np.float)
            print(x.shape, sample.shape)
            plt.plot(x, sample[row], c='green', label='Prediction')
            plt.ylim(-0.5, 1.5)
            plt.legend(fontsize=15)

            plt.subplot(7, 2, row*2+2)
            plt.plot(x, sample_gt[row], c='black', label='Ground truth')
            plt.ylim(-0.5, 1.5)
            plt.legend(fontsize=15)



        # fig, axes = plt.subplots(sample.shape[0])
        # for i, (ax, signal) in enumerate(zip(axes, sample)):
        #     x = np.arange(signal.shape[0], dtype=np.float)
        #     ax.plot(x, signal, c='green', label='Prediction')
        #     ax.set_ylim(-0.5, 1.5)
        #     ax.legend(fontsize=15)


        plt.savefig('./sigs_visual/'+str(current_id)+'_tool.png')
        plt.close()
        sample = []
        sample_gt = []
    current_id = ids[idx][0]
    sample.append(t)
    sample_gt.append(t_gt)



tool_cls_sig = []
for t in tool_label_col:
    tool_cls_sig.append(df["all"][t])
    # (N, 3)

phase_cls_sig = []
for t in phase_label_col:
    phase_cls_sig.append(df["all"][t])


