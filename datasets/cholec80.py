from pathlib import Path
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class Cholec80Helper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert hparams.data_root != ""
        self.data_root = Path(hparams.data_root)
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = hparams.factor_sampling

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        # Dict
        # np.asarray(self.current_stems_tool),
        # np.asarray(self.current_stems_phase),
        # np.asarray(self.current_p_phases),
        # np.asarray(self.current_phase_labels),
        # np.asarray(self.phase_sigs),
        # np.asarray(self.current_p_tools),
        # np.asarray(self.current_tool_labels),
        # np.asarray(self.tool_sigs)
        p = self.data_root / self.data_p[index]
        vid = self.data_p[index].split('/')[-1].split('_')[1]
        unpickled_x = pd.read_pickle(p)
        stem_tool = np.asarray(unpickled_x[0], dtype=np.float32)[::self.factor_sampling]
        stem_phase = np.asarray(unpickled_x[1], dtype=np.float32)[::self.factor_sampling]
        phase_label = np.asarray(unpickled_x[3], dtype=np.float32)[::self.factor_sampling]
        tool_label = np.asarray(unpickled_x[6])[::self.factor_sampling]

        phase_sig = np.asarray(unpickled_x[4], dtype=np.float32)[::self.factor_sampling]
        tool_sig = np.asarray(unpickled_x[7], dtype=np.float32)[::self.factor_sampling]

        return stem_tool, stem_phase, tool_label, phase_label, vid, tool_sig, phase_sig



class Cholec80():
    def __init__(self, hparams):
        self.name = "Cholec80Pickle"
        self.hparams = hparams
        self.out_features = self.hparams.out_features
        self.features_per_seconds = hparams.features_per_seconds
        hparams.factor_sampling = (int(25 / hparams.features_subsampling))
        print(
            f"Subsampling features: 25features_ps --> {hparams.features_subsampling}features_ps (factor: {hparams.factor_sampling})"
        )

        sub1 = ['02', '04', '06', '12', '24', '29', '34', '37', '38', '39', '44', '58', '60', '61', '64', '66', '75',
                '78', '79', '80']
        sub2 = ['01', '03', '05', '09', '13', '16', '18', '21', '22', '25', '31', '36', '45', '46', '48', '50', '62',
                '71', '72', '73']
        sub3 = ['10', '15', '17', '20', '32', '41', '42', '43', '47', '49', '51', '52', '53', '55', '56', '69', '70',
                '74', '76', '77']
        sub4 = ['07', '08', '11', '14', '19', '23', '26', '27', '28', '30', '33', '35', '40', '54', '57', '59', '63',
                '65', '67', '68']
        train_ids = sub1 + sub2 + sub3
        val_test_ids = sub4
        self.vids_for_training = [int(i) for i in train_ids]
        self.vids_for_val = [int(i) for i in val_test_ids]
        self.vids_for_test = self.vids_for_val

        self.data_p = {}
        self.data_p["train"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in self.vids_for_training]
        self.data_p["val"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in self.vids_for_val]
        self.data_p["test"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i:d}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in self.vids_for_test]

        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = Cholec80Helper(hparams,
                                              self.data_p[split],
                                              dataset_split=split)

        print(f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 dataset specific args options')
        cholec80_specific_args.add_argument("--features_per_seconds",
                                                  default=25,
                                                  type=float)
        cholec80_specific_args.add_argument("--features_subsampling",
                                                  default=5,
                                                  type=float)

        return parser
