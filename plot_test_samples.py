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

val_fold = ['7','8','11','14','19','23','26','27','28','30','33','35','40','54','57','59','63','65','67','68']

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

            phase_sig = data[4].transpose(1, 0)
            tool_sig = data[7].transpose(1, 0)

            y_reg_phase = y_reg_phase[:, -7:].transpose(1, 0)
            t_reg_phase = t_reg_phase[:, -7:].transpose(1, 0)
            y_reg_tool = y_reg_tool[:, -5:].transpose(1, 0)
            t_reg_tool = t_reg_tool[:, -5:].transpose(1, 0)

            ############################### Figure for Recognition Phase Signals ########################################
            plt.rcParams["figure.figsize"] = (40, 20)
            fig, axes = plt.subplots(phase_sig.shape[0])
            fig.suptitle(f)
            for i, (ax, signal) in enumerate(zip(axes, phase_sig)):
                x = np.arange(signal.shape[0], dtype=np.float)

                ax.plot(x, signal, c='black', label='Ground truth')
                ax.set_ylim(0, 1)

                ax.legend(fontsize=15)

            if not opts.save:
                plt.show()
            else:
                print('saving', opts.out_dir + f.split('.')[0] + '_phase_sig.png')
                plt.savefig(opts.out_dir + f.split('.')[0] + '_phase_sig.png')
            plt.close()

            ############################### Figure for Recognition Tool Signals ########################################
            plt.rcParams["figure.figsize"] = (40, 20)
            fig, axes = plt.subplots(tool_sig.shape[0])
            fig.suptitle(f)
            for i, (ax, signal) in enumerate(zip(axes, tool_sig)):
                x = np.arange(signal.shape[0], dtype=np.float)

                ax.plot(x, signal, c='black', label='Ground truth')
                ax.set_ylim(0, 1)

                ax.legend(fontsize=15)

            if not opts.save:
                plt.show()
            else:
                print('saving', opts.out_dir + f.split('.')[0] + '_tool_sig.png')
                plt.savefig(opts.out_dir + f.split('.')[0] + '_tool_sig.png')
            plt.close()

            ############################### Figure for Tool ########################################
            plt.rcParams["figure.figsize"] = (40, 20)
            fig, axes = plt.subplots(t_reg_tool.shape[0])
            fig.suptitle(f)
            for i, (ax, pred, gt) in enumerate(zip(axes, y_reg_tool, t_reg_tool)):
                x = np.arange(gt.shape[0], dtype=np.float)

                mean = pred

                ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0.1*horizon) & (gt < horizon),
                                facecolor='gray', alpha=.2)
                ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0) & (gt < 0.1*horizon),
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

                ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0.1*horizon) & (gt < horizon),
                                facecolor='gray', alpha=.2)
                ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0) & (gt < 0.1*horizon),
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