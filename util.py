import numpy as np
import os

instruments = [
    'Bipolar',
    'Scissors',
    'Clipper',
    'Irrigator',
    'SpecBag'
]

phases = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7'
]

colors = [
    'teal',
    'darkorange',
    'maroon',
    'forestgreen',
    'indigo',
    'red',
    'blue'
]
uncert_names = {
    'epistemic_reg': 'Epistemic uncertainty (reg.)',
    'epistemic_cls': 'Epistemic uncertainty (cls.)',
    'aleatoric_cls': 'Aleatoric uncertainty (cls.)',
    'entropy_cls': 'Entropy (cls.)'
}
classes = {
    'instrument_present': 0,
    'outside_horizon': 1,
    'inside_horizon': 2
}

softmax = lambda x, axis: np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)
epistemic_reg = lambda pred, dim_samples: pred.std(axis=dim_samples)
epistemic_cls = lambda p, dim_samples, dim_classes: np.sqrt(
    ((p - p.mean(axis=dim_samples, keepdims=True)) ** 2).mean(axis=(dim_classes, dim_samples)))
aleatoric_cls = lambda p, dim_samples, dim_classes: np.sqrt(np.mean(p * (1 - p), axis=(dim_classes, dim_samples)))
entropy_cls = lambda p, dim_samples, dim_classes: -(p.mean(axis=dim_samples) * np.log(p.mean(axis=dim_samples))).sum(
    axis=dim_classes)

import pickle



def load_samples_ins(opts):
    # Dict
    # np.asarray(self.current_stems_tool),
    # np.asarray(self.current_stems_phase),
    # np.asarray(self.current_p_phases),
    # np.asarray(self.current_phase_labels),
    # np.asarray(self.phase_sigs),
    # np.asarray(self.current_p_tools),
    # np.asarray(self.current_tool_labels),
    # np.asarray(self.tool_sigs)

    horizon = opts.horizon
    val_fold = ['7', '8', '11', '14', '19', '23', '26', '27', '28', '30', '33', '35', '40', '54', '57', '59', '63','65', '67', '68']

    prediction_reg_tool, target_reg_tool = [], []
    prediction_reg_phase, target_reg_phase = [], []

    for f in sorted(os.listdir(opts.sample_path)):
        vid_id = f.split('_1.0')[0].split('_')[-1]
        for kk in val_fold:
            if int(kk) == int(vid_id):
                df = open(os.path.join(opts.sample_path, f), 'rb')
                data = pickle.load(df)

                y_reg_phase = data[2] * horizon
                t_reg_phase = data[3] * horizon
                y_reg_tool = data[5] * horizon
                t_reg_tool = data[6] * horizon

                y_reg_phase = y_reg_phase[:, -7:]
                t_reg_phase = t_reg_phase[:, -7:]
                y_reg_tool = y_reg_tool[:, -5:]
                t_reg_tool = t_reg_tool[:, -5:]


                prediction_reg_tool.append(y_reg_tool)
                target_reg_tool.append(t_reg_tool)
                prediction_reg_phase.append(y_reg_phase)
                target_reg_phase.append(t_reg_phase)

    return prediction_reg_tool, target_reg_tool, prediction_reg_phase, target_reg_phase

def load_samples_ins_mstcn(opts):
    horizon = opts.horizon

    prediction_reg_tool, target_reg_tool = [], []
    prediction_reg_phase, target_reg_phase = [], []

    for f in sorted(os.listdir(opts.sample_path)):
        df = open(os.path.join(opts.sample_path, f), 'rb')
        data = pickle.load(df)

        y_reg_phase = data[0] * horizon
        t_reg_phase = data[1] * horizon
        y_reg_tool = data[2] * horizon
        t_reg_tool = data[3] * horizon

        y_reg_phase = y_reg_phase[:, -7:]
        t_reg_phase = t_reg_phase[:, -7:]
        y_reg_tool = y_reg_tool[:, -5:]
        t_reg_tool = t_reg_tool[:, -5:]


        prediction_reg_tool.append(y_reg_tool)
        target_reg_tool.append(t_reg_tool)
        prediction_reg_phase.append(y_reg_phase)
        target_reg_phase.append(t_reg_phase)

    return prediction_reg_tool, target_reg_tool, prediction_reg_phase, target_reg_phase

def print_scores(instruments, scores, header):
    print('\n{}'.format(header))

    for i, s in zip(instruments, scores):
        num_spaces = max([1, len(header) - (len(i) + 5)])
        print('{}:{}{:.2f}'.format(i, ' ' * num_spaces, s))

    print('-' * len(header))

    num_spaces = max([1, len(header) - 9])
    print('Mean:{}{:.2f}\n'.format(' ' * num_spaces, np.mean(scores)))
