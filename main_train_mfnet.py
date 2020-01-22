# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:04:16 2018

main train mfnet

@author: Γιώργος
"""

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from src.models.mfnet_3d_mo import MFNET_3D_MO
from src.models.mfnet_3d_mo_comb import MFNET_3D_MO_COMB
from src.models.mfnet_3d_slowfast import MFNET_3D_SF
from src.models.mfnet_3d_mo_mm import MFNET_3D_MO_MM
from src.models.mfnet_3d_mo_dfb import MFNET_3D_DFB
from src.models.mfnet_3d_mo_lstm import MFNET_3D_LSTM
from src.models.mfnet_3d_mo_tdn import MFNET_3D_TDN
from src.models.mfnet_3d_mo_weighted import MFNET_3D_MO_WEIGHTED
from src.models.mfnet_3d_mo_t_attn import MFNET_3D_MO_T_ATTN
from src.utils.argparse_utils import parse_args, parse_tasks_str, parse_tasks_per_dataset
from src.utils.file_utils import print_and_save, save_mt_checkpoints, init_folders, resume_checkpoint, load_pretrained_weights
from src.utils.dataset.dataset_loader import MultitaskDatasetLoader, MultitaskDatasetLoaderVideoLevel
from src.utils.video_sampler import prepare_sampler
from src.utils.dataset.dataset_loader_transforms import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS_2, ToTensorVid,\
    Normalize, Resize, CenterCrop, PredefinedHorizontalFlip
from src.utils.train_utils import train_mfnet_mo, test_mfnet_mo, test_mfnet_mo_map, train_mfnet_mo_comb, test_mfnet_mo_comb
from src.utils.lr_utils import load_lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.constants import *


def main():
    args, model_name = parse_args('mfnet', val=False)
    tasks_per_dataset = parse_tasks_str(args.tasks, args.dataset, args.interpolate_coordinates)
    objectives_text, objectives, task_sizes = parse_tasks_per_dataset(tasks_per_dataset)
    num_classes, num_coords, num_objects, num_obj_cat = task_sizes
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    print_and_save("Training for {} objective(s)".format(sum(objectives)), log_file)
    print_and_save(objectives_text, log_file)
    cudnn.benchmark = True
    multioutput_loss = 0

    kwargs = dict()
    if args.sf:
        mfnet_3d = MFNET_3D_SF
    elif args.flow:
        mfnet_3d = MFNET_3D_MO_MM
        kwargs['modalities'] = {'RGB': 3, 'Flow': 2}
    elif args.dfb:
        mfnet_3d = MFNET_3D_DFB
        multioutput_loss = 4
    elif args.lstm:
        mfnet_3d = MFNET_3D_LSTM
        kwargs['attn'] = args.attn
        kwargs['mtl'] = args.mtl
        if args.mtl:
            multioutput_loss = 3
    elif args.attn:
        mfnet_3d = MFNET_3D_MO_WEIGHTED
    elif args.t_attn:
        mfnet_3d = MFNET_3D_MO_T_ATTN
    elif args.tdn:
        mfnet_3d = MFNET_3D_TDN
        multioutput_loss = 3
    elif args.map_tasks:
        mfnet_3d = MFNET_3D_MO_COMB
    else:
        mfnet_3d = MFNET_3D_MO
        if args.only_flow:
            kwargs['input_channels'] = 2

    kwargs["num_coords"] = num_coords
    kwargs["num_objects"] = num_objects
    kwargs["num_obj_cat"] = num_obj_cat
    kwargs["one_object_layer"] = args.one_object_layer
    kwargs["interpolate_coordinates"] = args.interpolate_coordinates
    if args.long:
        kwargs["k_sec"] = {2: 3, 3: 4, 4: 11, 5: 3}
    model_ft = mfnet_3d(num_classes, dropout=args.dropout, **kwargs)
    if args.pretrained:
        model_ft = load_pretrained_weights(model_ft, args)
    model_ft.cuda(device=args.gpus[0])
    model_ft = torch.nn.DataParallel(model_ft, device_ids=args.gpus, output_device=args.gpus[0])
    print_and_save("Model loaded on gpu {} devices".format(args.gpus), log_file)

    # config optimizer
    param_base_layers = []
    param_new_layers = []
    name_base_layers = []
    for name, param in model_ft.named_parameters():
        if args.pretrained:
            if 'classifier' in name or 'coord_layers' in name:
                param_new_layers.append(param)
            else:
                param_base_layers.append(param)
                name_base_layers.append(name)
        else:
            param_new_layers.append(param)

    optimizer = torch.optim.SGD([{'params': param_base_layers, 'lr_mult': args.lr_mult_base},
                                 {'params': param_new_layers, 'lr_mult': args.lr_mult_new}],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.decay, nesterov=True)

    if args.resume:
        (model_ft, optimizer), ckpt_path = resume_checkpoint(model_ft, optimizer, output_dir, model_name, args.resume_from)
        print_and_save("Resuming training from: {}".format(ckpt_path), log_file)

    # load dataset and train and validation iterators
    train_sampler = prepare_sampler("random", args.clip_length, args.frame_interval, speed=[0.5, 1.5])
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]), RandomCrop((224, 224)),
        RandomHorizontalFlip(), RandomHLS_2(vars=[15, 35, 25]), ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    if args.only_flow:
        train_transforms_flow = transforms.Compose([
            RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288]), RandomCrop((224, 224)),
            RandomHorizontalFlip(), ToTensorVid(dim=2), Normalize(mean=mean_1d, std=std_1d)])
    else:
        train_transforms_flow = transforms.Compose([
            RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288]), RandomCrop((224, 224)),
            PredefinedHorizontalFlip(), ToTensorVid(dim=2), Normalize(mean=mean_1d, std=std_1d)])
    train_loader = MultitaskDatasetLoader(train_sampler, args.train_lists, args.dataset, tasks_per_dataset,
                                          batch_transform=train_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                          hand_list_prefix=args.hand_list_prefix[:],
                                          object_list_prefix=args.object_list_prefix[:],
                                          object_categories=args.object_cats[:],
                                          use_flow=args.flow, flow_transforms=train_transforms_flow,
                                          only_flow=args.only_flow,
                                          map_to_epic=args.map_tasks,
                                          interpolate_coords=args.interpolate_coordinates)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers, pin_memory=True)

    test_sampler = prepare_sampler(args.eval_sampler, args.clip_length, args.frame_interval, speed=[1.0, 1.0],
                                   window=args.eval_window)
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(),
                                          Normalize(mean=mean_3d, std=std_3d)])
    test_transforms_flow = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(dim=2),
                                              Normalize(mean=mean_1d, std=std_1d)])
    test_loader = MultitaskDatasetLoader(test_sampler, args.test_lists, args.dataset, tasks_per_dataset,
                                         batch_transform=test_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                         hand_list_prefix=args.hand_list_prefix[:],
                                         object_list_prefix=args.object_list_prefix[:],
                                         object_categories=args.object_cats[:],
                                         use_flow=args.flow, flow_transforms=test_transforms_flow,
                                         only_flow=args.only_flow,
                                         map_to_epic=args.map_tasks,
                                         interpolate_coords=args.interpolate_coordinates)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)

    if args.eval_map_vl:
        map_sampler = prepare_sampler('middle', 16, None, speed=None, window=args.eval_window)
        map_loader = MultitaskDatasetLoaderVideoLevel(map_sampler, args.eval_lists_vl, args.dataset, tasks_per_dataset,
                                                      test_transforms, vis_data=False)
        map_iterator = torch.utils.data.DataLoader(map_loader, batch_size=25, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=True)

    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    if args.map_tasks:
        train = train_mfnet_mo_comb
        test = test_mfnet_mo_comb
    else:
        train = train_mfnet_mo
        test = test_mfnet_mo
    num_cls_tasks = objectives[0]
    top1 = [0.0] * num_cls_tasks
    if args.eval_map_vl:
        mAP = [0.0] * num_cls_tasks
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, train_iterator, tasks_per_dataset, epoch, log_file, args.gpus,
              lr_scheduler=lr_scheduler, moo=args.moo, use_flow=args.flow, one_obj_layer=args.one_object_layer,
              grad_acc_batches=args.grad_acc_batches, multioutput_loss=multioutput_loss, t_attn=args.t_attn)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, train_iterator, tasks_per_dataset, epoch, "Train", log_file, args.gpus,
                     use_flow=args.flow, one_obj_layer=args.one_object_layer, multioutput_loss=multioutput_loss,
                     t_attn=args.t_attn)
            new_top1 = test(model_ft, test_iterator, tasks_per_dataset, epoch, "Test", log_file, args.gpus,
                            use_flow=args.flow, one_obj_layer=args.one_object_layer, multioutput_loss=multioutput_loss,
                            t_attn=args.t_attn)
            top1 = save_mt_checkpoints(model_ft, optimizer, top1, new_top1, args.save_all_weights, output_dir,
                                       model_name, epoch)
            if args.eval_map_vl:
                new_mAP = test_mfnet_mo_map(model_ft, map_iterator, tasks_per_dataset, epoch, "Video level test", log_file,
                                            args.gpus)
                mAP = save_mt_checkpoints(model_ft, optimizer, mAP, new_mAP, args.save_all_weights, output_dir,
                                          model_name, epoch, mAP=True)

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(new_top1[0])


if __name__ == '__main__':
    main()
