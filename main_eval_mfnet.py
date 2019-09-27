# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2019

main eval mfnet multitask

@author: Georgios Kapidis
"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from src.models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from src.utils.argparse_utils import parse_args, make_log_file_name, parse_tasks_str, parse_tasks_per_dataset
from src.utils.file_utils import print_and_save
from src.utils.dataset_loader import MultitaskDatasetLoader
from src.utils.dataset_loader_utils import Resize, RandomCrop, ToTensorVid, Normalize, CenterCrop
from src.utils.calc_utils import eval_final_print_mt
from src.utils.video_sampler import RandomSampling, MiddleSampling
from src.utils.train_utils import validate_mfnet_mo
from src.constants import *

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)

def main():
    args = parse_args('mfnet', val=True)
    tasks_per_dataset = parse_tasks_str(args.tasks)
    objectives_text, num_objectives, num_classes, num_coords = parse_tasks_per_dataset(tasks_per_dataset)
    output_dir = os.path.dirname(args.ckpt_path)
    log_file = make_log_file_name(output_dir, args)
    print_and_save(args, log_file)
    cudnn.benchmark = True

    mfnet_3d = MFNET_3D_MO
    validate = validate_mfnet_mo

    kwargs = dict()
    kwargs['num_coords'] = num_coords

    model_ft = mfnet_3d(num_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    model_ft.load_state_dict(checkpoint['state_dict'])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    overall_top1 = [0]*num_objectives[0]
    overall_mean_cls_acc = [0]*num_objectives[0]
    for i in range(args.mfnet_eval):
        crop_type = CenterCrop((224, 224)) if args.eval_crop == 'center' else RandomCrop((224, 224))
        if args.eval_sampler == 'middle':
            val_sampler = MiddleSampling(num=args.clip_length)
        else:
            val_sampler = RandomSampling(num=args.clip_length,
                                         interval=args.frame_interval,
                                         speed=[1.0, 1.0], seed=i)

        val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                             ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

        val_loader = MultitaskDatasetLoader(val_sampler, args.val_lists, args.dataset, tasks_per_dataset,
                                            batch_transform=val_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                            hand_list_prefix=args.hand_list_prefix[:], validation=True)
        val_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)

        # evaluate dataset
        top1, outputs = validate(model_ft, ce_loss, val_iter, num_objectives, tasks_per_dataset,
                                 checkpoint['epoch'], args.dataset, log_file)

        # calculate statistics
        for ind in range(len(num_classes)):
            video_preds = [x[0] for x in outputs[ind]]
            video_labels = [x[1] for x in outputs[ind]]
            task_type = ''
            for td in tasks_per_dataset:
                for key, value in td.items():
                    if value == num_classes[ind]:
                        task_type = key
            mean_cls_acc, top1_acc = eval_final_print_mt(video_preds, video_labels, args.dataset, ind,
                                                         num_classes[ind], log_file, args.annotations_path,
                                                         args.val_list, task_type=task_type,
                                                         actions_file=args.epic_actions_path)
            overall_mean_cls_acc[ind] += mean_cls_acc
            overall_top1[ind] += top1_acc

    print_and_save("", log_file)
    text_mean_cls_acc = "Mean Cls Acc ({} times)".format(args.mfnet_eval)
    text_dataset_acc = "Dataset Acc ({} times)".format(args.mfnet_eval)
    for ind in range(len(num_classes)):
        text_mean_cls_acc += ", T{}::{} ".format(ind, (overall_mean_cls_acc[ind] / args.mfnet_eval))
        text_dataset_acc += ", T{}::{} ".format(ind, (overall_top1[ind] / args.mfnet_eval))
    print_and_save(text_mean_cls_acc, log_file)
    print_and_save(text_dataset_acc, log_file)


if __name__ == '__main__':
    main()
