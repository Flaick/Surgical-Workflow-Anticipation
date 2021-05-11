import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle
import torch
import random
import json
import cv2
import math

def write_pkl(root_dir, horizon=5):
    print("writing_pkl...")
    root_dir = Path(root_dir)
    img_base_path = root_dir / "/home/kun/Desktop/miccai21/TeCNO/Videos/cholec_split_250px_25fps/"
    seg_base_path = root_dir / "/home/kun/Desktop/miccai21/TeCNO/Videos/cholec_split_250px_25fps_masks/"
    annot_tool_path = root_dir / "tool_annotations" # GT tool presence
    annot_timephase_path = root_dir / "time_stamp" # GT phase presence
    out_path = root_dir / "dataframes"
    annot_tool_anni_path = str(root_dir) + "/anticipation_annotations/tool/h" + str(horizon)
    annot_phase_anni_path = str(root_dir) + "/anticipation_annotations/phase/h" + str(horizon)
    phase_signal_path = str(root_dir) + "/phase_annotations_tecno" # Detected phase presence
    tool_signal_path = '/home/kun/Desktop/miccai21/data/annotations/tool_annotations_yolo_5_5/' # Detected tool presence
    # tool_signal_path = str(root_dir) + "/tool_annotations_yolo" # Previously detected tool presence

    class_labels = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningCoagulation",
        "GallbladderRetraction",
    ]

    cholec_df = pd.DataFrame(columns=[
        "image_path", "seg_image_path", "class", "time", "video_idx",
        "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag",
        "Bipolar_reg", "Scissors_reg", "Clipper_reg", "Irrigator_reg", "SpecimenBag_reg",
        "Bipolar_cls", "Scissors_cls", "Clipper_cls", "Irrigator_cls", "SpecimenBag_cls",
        "P1_reg", "P2_reg", "P3_reg", "P4_reg", "P5_reg", "P6_reg", "P7_reg",
        "P1_cls", "P2_cls", "P3_cls", "P4_cls", "P5_cls", "P6_cls", "P7_cls",
        "tool_Grasper_yolo", "tool_Bipolar_yolo", "tool_Hook_yolo", "tool_Scissors_yolo", "tool_Clipper_yolo", "tool_Irrigator_yolo", "tool_SpecimenBag_yolo",
        "P1_tecno", "P2_tecno", "P3_tecno", "P4_tecno", "P5_tecno", "P6_tecno", "P7_tecno",
    ])
    for id in tqdm(range(1, 81)):
        vid_df = pd.DataFrame()

        img_path_for_vid = img_base_path / f"video{id:02d}"
        img_list = sorted(img_path_for_vid.glob('*.png'))
        img_list = [str(i.relative_to(img_base_path)) for i in img_list]
        vid_df["image_path"] = img_list

        vid_df["video_idx"] = [id] * len(img_list)

        seg_path_for_vid = seg_base_path / f"video{id:02d}"
        seg_list = sorted(seg_path_for_vid.glob('*.png'))
        seg_list = [str(i.relative_to(seg_base_path)) for i in seg_list]
        numbers_of_repetitions = 25
        seg_list = [val for val in seg_list for i in range(numbers_of_repetitions)]
        vid_df["seg_image_path"] = seg_list[:len(img_list)]
        assert len(vid_df["seg_image_path"]) == len(vid_df["image_path"])

        '''
        Adding Anticipation Labels for Tool & Phase
        '''
        # add regression and classification label for Tool anticipation
        vid_tar_reg = annot_tool_anni_path + f"/video{id:02d}-reg.txt"
        reg_short = pd.read_csv(vid_tar_reg, sep='\t')
        regs_df_tool = []
        for row in reg_short.itertuples(index=False):
            numbers_of_repetitions = 25
            regs_df_tool.extend([list(row)] * numbers_of_repetitions)
        regs_df_tool.append(regs_df_tool[-1])
        regs_df_tool = np.array(regs_df_tool)
        regs_df_tool = pd.DataFrame(regs_df_tool[:, 1:], columns=[
                                    "Bipolar_reg", "Scissors_reg",
                                    "Clipper_reg", "Irrigator_reg",
                                    "SpecimenBag_reg"])

        vid_cls_reg = annot_tool_anni_path + f"/video{id:02d}-cls.txt"
        cls_short = pd.read_csv(vid_cls_reg, sep='\t')
        clss_df_tool = []
        for row in cls_short.itertuples(index=False):
            numbers_of_repetitions = 25
            clss_df_tool.extend([list(row)] * numbers_of_repetitions)
        clss_df_tool.append(clss_df_tool[-1])
        clss_df_tool = np.array(clss_df_tool)
        clss_df_tool = pd.DataFrame(clss_df_tool[:, 1:], columns=[
                                    "Bipolar_cls", "Scissors_cls",
                                    "Clipper_cls", "Irrigator_cls",
                                    "SpecimenBag_cls"])

        # add regression and classification label for Phase anticipation
        vid_tar_reg = annot_phase_anni_path + f"/video{id:02d}-reg.txt"
        reg_short = pd.read_csv(vid_tar_reg, sep='\t')
        regs_df_phase = []
        for row in reg_short.itertuples(index=False):
            numbers_of_repetitions = 25
            regs_df_phase.extend([list(row)] * numbers_of_repetitions)
        regs_df_phase.append(regs_df_phase[-1])
        regs_df_phase = np.array(regs_df_phase)
        regs_df_phase = pd.DataFrame(regs_df_phase[:, 1:], columns=[
                                    "P1_reg", "P2_reg","P3_reg", "P4_reg",
                                    "P5_reg", "P6_reg", "P7_reg"])

        vid_cls_reg = annot_phase_anni_path + f"/video{id:02d}-cls.txt"
        cls_short = pd.read_csv(vid_cls_reg, sep='\t')
        clss_df_phase = []
        for row in cls_short.itertuples(index=False):
            numbers_of_repetitions = 25
            clss_df_phase.extend([list(row)] * numbers_of_repetitions)
        clss_df_phase.append(clss_df_phase[-1])
        clss_df_phase = np.array(clss_df_phase)
        clss_df_phase = pd.DataFrame(clss_df_phase[:, 1:], columns=[
                                    "P1_cls", "P2_cls","P3_cls", "P4_cls",
                                    "P5_cls", "P6_cls", "P7_cls"])

        anni_anno = pd.concat([regs_df_tool, clss_df_tool, regs_df_phase, clss_df_phase], axis=1)

        '''
        Adding Tool Signal and Phase Signal
        '''
        #### Adding Tool YOLO Signal ####
        vid_tools_yolo = tool_signal_path + f"/video{id:02d}-tool.txt"
        tools_short_yolo = pd.read_csv(vid_tools_yolo, sep='\t')[:-1]
        # tools_short_yolo = pd.read_csv(vid_tools_yolo, sep='\t') # Previous Detect tool signal
        tools_df_yolo = []
        for row in tools_short_yolo.itertuples(index=False):
            numbers_of_repetitions = 25
            tools_df_yolo.extend([list(row)] * numbers_of_repetitions)
        tools_df_yolo.append(tools_df_yolo[-1])
        tools_df_yolo = np.array(tools_df_yolo)
        tools_df_yolo = pd.DataFrame(tools_df_yolo[:, 1:],
                                columns=[
                                    "tool_Grasper_yolo", "tool_Bipolar_yolo",
                                    "tool_Hook_yolo", "tool_Scissors_yolo",
                                    "tool_Clipper_yolo", "tool_Irrigator_yolo",
                                    "tool_SpecimenBag_yolo"
                                ])

        #### Adding Phase TECNO Signal ####
        vid_phase_tecno = phase_signal_path + f"/video_{id:d}.pkl"
        df = open(vid_phase_tecno, 'rb')
        data = pickle.load(df)
        df.close()

        pre_ = data[0][-1, 0, ...]
        pre = np.argmax(pre_, 0) # t,

        phase_df_tecno = []
        for i in pre[:-1]:
            pre_onehot = np.zeros(shape=(7, ))
            pre_onehot[i] = 1
            phase_df_tecno.append(np.expand_dims(pre_onehot, 0).repeat(25, axis=0)) # [25, 7]

        pre_onehot = np.zeros(shape=(7,))
        pre_onehot[pre[-1]] = 1
        phase_df_tecno.append(np.expand_dims(pre_onehot, 0).repeat(100, axis=0))  # [25, 7]

        phase_df_tecno = np.concatenate(phase_df_tecno)
        phase_df_tecno = phase_df_tecno[:len(img_list), :]
        phase_df_tecno = pd.DataFrame(phase_df_tecno, columns=["P1_tecno", "P2_tecno", "P3_tecno", "P4_tecno", "P5_tecno", "P6_tecno","P7_tecno"])

        # add image class
        vid_time_and_phase = annot_timephase_path / f"video{id:02d}-timestamp.txt"
        phases = pd.read_csv(vid_time_and_phase, sep='\t')
        for j, p in enumerate(class_labels):
            phases["Phase"] = phases.Phase.replace({p: j})

        # Add tool gt sig
        vid_tools = annot_tool_path / f"video{id:02d}-tool.txt"
        tools_short = pd.read_csv(vid_tools, sep='\t')
        tools_df = []
        for row in tools_short.itertuples(index=False):
            numbers_of_repetitions = 25
            tools_df.extend([list(row)] * numbers_of_repetitions)
        tools_df.append(tools_df[-1])
        tools_df = np.array(tools_df)

        tools_df = pd.DataFrame(tools_df[:, 1:],
                                columns=[
                                    "tool_Grasper", "tool_Bipolar",
                                    "tool_Hook", "tool_Scissors",
                                    "tool_Clipper", "tool_Irrigator",
                                    "tool_SpecimenBag"
                                ])

        vid_df = pd.concat([vid_df, phases], axis=1)
        vid_df = pd.concat([vid_df, tools_df], axis=1)
        vid_df = pd.concat([vid_df, anni_anno], axis=1) # Annotation for Anticipation
        vid_df = pd.concat([vid_df, tools_df_yolo], axis=1) # Tool Signal from Yolo
        vid_df = pd.concat([vid_df, phase_df_tecno], axis=1) # Phase Signal from TeCNO

        if not vid_df.shape[0] == len(img_list) == len(tools_df) == len(phases) == len(anni_anno) == len(tools_df_yolo) == len(tools_df_yolo):
            print(
                f"vid_df.shape[0]:{vid_df.shape[0]} - len(img_list): {len(img_list)} - len(tools_df):{len(tools_df)} - len(phases):{len(phases)} "
                f"- len(anticipation annotation):{len(anni_anno)} - len(yolo tool signal):{len(tools_df_yolo)} - len(tecno phase signal):{len(tools_df_yolo)}"
            )
        # assert vid_df.shape[0]==len(img_list)==len(tools_df)==len(phases)==len(anni_anno)==len(tools_df_yolo)==len(tools_df_yolo)
        vid_df = vid_df.rename(columns={
            "Phase": "class",
            "Frame": "time",
        })
        cholec_df = cholec_df.append(vid_df, ignore_index=True, sort=False)

    print("DONE")
    print(cholec_df.shape)
    print(cholec_df.columns)
    cholec_df.to_pickle(str(out_path) + "/cholec_split_250px_25fps_anni_"+str(horizon)+".pkl")


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    root_dir = "/home/kun/Desktop/miccai21/TeCNO/Videos/"
    write_pkl(root_dir, horizon=3)
