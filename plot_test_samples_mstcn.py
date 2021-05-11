import numpy as np
import matplotlib.pyplot as plt
import os
import util
from options import parser
import pickle

opts = parser.parse_args()

horizon = opts.horizon

if opts.save and not os.path.exists(opts.out_dir):
    os.mkdir(opts.out_dir)

for f in sorted(os.listdir(opts.sample_path)):
    df = open(os.path.join(opts.sample_path, f), 'rb')
    data = pickle.load(df)

    y_reg_phase = data[0] * horizon
    t_reg_phase = data[1] * horizon
    y_reg_tool = data[2] * horizon
    t_reg_tool = data[3] * horizon

    y_reg_phase = y_reg_phase[:, -7:].transpose(1, 0)
    t_reg_phase = t_reg_phase[:, -7:].transpose(1, 0)
    y_reg_tool = y_reg_tool[:, -5:].transpose(1, 0)
    t_reg_tool = t_reg_tool[:, -5:].transpose(1, 0)

    ############################### Figure for Tool ########################################
    plt.rcParams["figure.figsize"] = (40, 20)
    fig, axes = plt.subplots(t_reg_tool.shape[0])
    fig.suptitle(f)
    for i, (ax, pred, gt) in enumerate(zip(axes, y_reg_tool, t_reg_tool)):
        x = np.arange(gt.shape[0], dtype=np.float)

        mean = pred

        ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0.1 * horizon) & (gt < horizon),
                        facecolor='gray', alpha=.2)
        ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0) & (gt < 0.1 * horizon),
                        facecolor='red', alpha=.2)
        ax.plot(x, gt, c='black', label='Ground truth')
        ax.plot(x, mean, c=util.colors[i], label='Prediction')

        ax.set_xlabel('Time [sec.]', size=15)
        ax.set_ylabel('{}\n[min.]'.format(util.instruments[i]), size=15)

        ax.set_yticks([0, horizon / 2, horizon])
        ax.set_yticklabels([0, horizon / 2, '>' + str(horizon)], size=12)
        ax.set_ylim(-.5, horizon + .5)

        ax.legend(fontsize=15)

    if not opts.save:
        plt.show()
    else:
        print('saving', opts.out_dir + f.split('.')[0] + '_tool.png')
        plt.savefig(opts.out_dir + f.split('.')[0] + '_tool.png')
    plt.close()

    ############################### Figure for Phase ########################################

    plt.rcParams["figure.figsize"] = (40, 20)
    fig, axes = plt.subplots(t_reg_phase.shape[0])
    fig.suptitle(f)
    for i, (ax, pred, gt) in enumerate(zip(axes, y_reg_phase, t_reg_phase)):
        x = np.arange(gt.shape[0], dtype=np.float)

        mean = pred

        ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0.1 * horizon) & (gt < horizon),
                        facecolor='gray', alpha=.2)
        ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0) & (gt < 0.1 * horizon),
                        facecolor='red', alpha=.2)
        ax.plot(x, gt, c='black', label='Ground truth')
        ax.plot(x, mean, c=util.colors[i], label='Prediction')

        ax.set_xlabel('Time [sec.]', size=15)
        ax.set_ylabel('{}\n[min.]'.format(util.phases[i]), size=15)

        ax.set_yticks([0, horizon / 2, horizon])
        ax.set_yticklabels([0, horizon / 2, '>' + str(horizon)], size=12)
        ax.set_ylim(-.5, horizon + .5)

        ax.legend(fontsize=15)

    if not opts.save:
        plt.show()
    else:
        print('saving', opts.out_dir + f.split('.')[0] + '_phase.png')
        plt.savefig(opts.out_dir + f.split('.')[0] + '_phase.png')
    plt.close()