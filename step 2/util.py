import datetime
import os
from shutil import copy2
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import math


def prepare_output_folders(opts):
    output_folder = "{}{}_horizon{}_{}/".format(opts.output_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M"),
                                                opts.horizon, opts.trial_name)
    print('Output directory: ' + output_folder)
    result_folder = os.path.join(output_folder, 'results')
    script_folder = os.path.join(output_folder, 'scripts')
    model_folder = os.path.join(output_folder, 'models')
    log_path = os.path.join(output_folder, 'log.txt')
    log_path_csv = os.path.join(output_folder, 'log.csv')


    os.makedirs(output_folder)
    os.makedirs(result_folder)
    os.makedirs(script_folder)
    os.makedirs(model_folder)

    for f in os.listdir():
        if '.py' in f:
            copy2(f, script_folder)

    return result_folder, model_folder, log_path, log_path_csv


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def encode_other_boxes(person_box, other_box):
    """Encoder other boxes."""
    # get relative geometric feature
    x1, y1, x2, y2 = person_box
    xx1, yy1, xx2, yy2 = other_box

    x_m = x1
    y_m = y1
    w_m = x2 - x1
    h_m = y2 - y1

    x_n = xx1
    y_n = yy1
    w_n = xx2 - xx1
    h_n = yy2 - yy1

    return [
        math.log(max((x_m - x_n), 1e-3) / w_m),
        math.log(max((y_m - y_n), 1e-3) / h_m),
        math.log(w_n / w_m),
        math.log(h_n / h_m),
    ]
