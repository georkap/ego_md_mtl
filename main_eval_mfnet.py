# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 2019

main eval mfnet multitask
Notes for evaluation on multidataset models.
Evaluation is not to be done on multiple datasets together as it will most probably complicate everything when calculating metrics.
Instead I intend to run the evaluation script for one dataset only.

The smartest thing to do would be to choose which weights to load into the model depending on the dataset I have.
args.tasks has all the model tasks, as it was originally trained.
Adding args.eval_tasks for the tasks to be evaluated. I will compare eval tasks with tasks and load only the appropriate output layers.
Dataloader loads only the dataset to be evaluated with the tasks assigned from args.eval_tasks as if loading a single-dataset model.

When designing the model structure and loading the weights it chooses only the appropriate output layers for the tasks
of the dataset under evaluation.

So I will create a single-dataset model from a multi-dataset one.

@author: Georgios Kapidis
"""

import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from src.models.mfnet_3d_slowfast import MFNET_3D_SF as MFNET_3D_SF
from src.models.mfnet_3d_mo_mm import MFNET_3D_MO_MM as MFNET_3D_MO_MM
from src.utils.argparse_utils import parse_args, make_log_file_name, parse_tasks_str, parse_tasks_per_dataset, compare_tasks_per_dataset
from src.utils.file_utils import print_and_save
from src.utils.dataset.dataset_loader import MultitaskDatasetLoader
from src.utils.dataset.dataset_loader_transforms import Resize, RandomCrop, ToTensorVid, Normalize, CenterCrop
from src.utils.calc_utils import eval_final_print_mt
from src.utils.video_sampler import RandomSampling, MiddleSampling
from src.utils.train_utils import validate_mfnet_mo
from src.constants import *

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)

def main():
    args = parse_args('mfnet', val=True)
    tasks_per_dataset = parse_tasks_str(args.tasks, args.dataset)
    if args.eval_tasks is not None: # trained multi-dataset eval single-dataset
        eval_tasks_per_dataset = parse_tasks_str(args.eval_tasks, [args.eval_dataset])
        starting_cls_id, starting_g_id, starting_h_id = compare_tasks_per_dataset(tasks_per_dataset,
                                                                                  eval_tasks_per_dataset)
        train_tasks_per_dataset = tasks_per_dataset
        tasks_per_dataset = eval_tasks_per_dataset
        args.dataset = [args.eval_dataset]

    objectives_text, num_objectives, num_classes, num_coords, num_objects = parse_tasks_per_dataset(tasks_per_dataset)

    output_dir = os.path.dirname(args.ckpt_path)
    log_file = make_log_file_name(output_dir, args)
    print_and_save(args, log_file)
    cudnn.benchmark = True

    kwargs = dict()
    if args.sf:
        mfnet_3d = MFNET_3D_SF
    elif args.flow:
        mfnet_3d = MFNET_3D_MO_MM
        kwargs['modalities'] = {'RGB':3, 'Flow':2}
    else:
        mfnet_3d = MFNET_3D_MO
    validate = validate_mfnet_mo

    kwargs['num_coords'] = num_coords
    kwargs["num_objects"] = num_objects
    if args.long:
        kwargs["k_sec"] = {2: 3, 3: 4, 4: 11, 5: 3}
    model_ft = mfnet_3d(num_classes, **kwargs)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:1': 'cuda:0'})
    if args.eval_tasks is not None:
        cls_task_base_name = 'module.classifier_list.classifier_list.{}.{}'
        for i in range(len(num_classes)):
            for layer in ['weight', 'bias']:
                checkpoint['state_dict'][cls_task_base_name.format(i, layer)] = \
                    checkpoint['state_dict'][cls_task_base_name.format(i + starting_cls_id, layer)]
        starting_coord_id = starting_g_id + 2 * starting_h_id
        if num_coords > 0:
            checkpoint['state_dict']['module.coord_layers.hm_conv.weight'] = \
                checkpoint['state_dict']['module.coord_layers.hm_conv.weight'][starting_coord_id:
                                                                               starting_coord_id+num_coords]

    model_ft.load_state_dict(checkpoint['state_dict'], strict=(args.eval_tasks is None))
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    ce_loss = torch.nn.CrossEntropyLoss().cuda()

    overall_top1 = [0]*num_objectives[0]
    overall_mean_cls_acc = [0]*num_objectives[0]
    for i in range(args.mfnet_eval):
        crop_type = CenterCrop((224, 224)) if args.eval_crop == 'center' else RandomCrop((224, 224))
        if args.eval_sampler == 'middle':
            val_sampler = MiddleSampling(num=args.clip_length, window=args.eval_window)
        else:
            val_sampler = RandomSampling(num=args.clip_length, interval=args.frame_interval, speed=[1.0, 1.0], seed=i)

        val_transforms = transforms.Compose([Resize((256, 256), False), crop_type, ToTensorVid(),
                                             Normalize(mean=mean_3d, std=std_3d)])
        val_transforms_flow = transforms.Compose(
            [Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(dim=2),
             Normalize(mean=mean_1d, std=std_1d)])
        val_loader = MultitaskDatasetLoader(val_sampler, args.val_lists, args.dataset, tasks_per_dataset,
                                            batch_transform=val_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                            hand_list_prefix=args.hand_list_prefix[:],
                                            object_list_prefix=args.object_list_prefix[:],
                                            validation=True,
                                            use_flow=args.flow, flow_transforms=val_transforms_flow)
        val_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)

        # evaluate dataset
        top1, outputs = validate(model_ft, ce_loss, val_iter, num_objectives, checkpoint['epoch'], args.dataset,
                                 log_file, args.flow)

        # calculate statistics
        for ind in range(len(num_classes)):
            video_preds = [x[0] for x in outputs[ind]]
            video_labels = [x[1] for x in outputs[ind]]
            task_type = ''
            # dataset_index = -1
            # find the name/type of the current task's predictions, since I do not mix the tasks between the datasets
            # i.e. a task layer produces outputs for one dataset only, I can find and use the dataset index
            # for dataset specific evaluation
            for dt_ind, td in enumerate(tasks_per_dataset):
                for key, value in td.items():
                    if value == num_classes[ind]:
                        task_type = key
                        # dataset_index = dt_ind
            mean_cls_acc, top1_acc = eval_final_print_mt(video_preds, video_labels, args.dataset[0], ind,
                                                         num_classes[ind], log_file, args.annotations_path,
                                                         args.val_lists[0], task_type=task_type,
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
