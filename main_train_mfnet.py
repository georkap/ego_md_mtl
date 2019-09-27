# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:04:16 2018

main train mfnet

@author: Γιώργος
"""

import torch
from torch.optim import SGD
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from src.models.mfnet_3d_mo import MFNET_3D as MFNET_3D_MO
from src.utils.argparse_utils import parse_args, parse_tasks_str, parse_tasks_per_dataset
from src.utils.file_utils import print_and_save, save_mt_checkpoints, init_folders, resume_checkpoint
from src.utils.dataset_loader import MultitaskDatasetLoader, prepare_sampler
from src.utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid,\
    Normalize, Resize, CenterCrop
from src.utils.train_utils import load_lr_scheduler, train_mfnet_mo, test_mfnet_mo
from src.constants import *


def main():
    args, model_name = parse_args('mfnet', val=False)
    tasks_per_dataset = parse_tasks_str(args.tasks)
    objectives_text, num_objectives, num_classes, num_coords = parse_tasks_per_dataset(tasks_per_dataset)
    output_dir, log_file = init_folders(args.base_output_dir, model_name, args.resume, args.logging)
    print_and_save(args, log_file)
    print_and_save("Model name: {}".format(model_name), log_file)
    print_and_save("Training for {} objective(s)".format(sum(num_objectives)), log_file)
    print_and_save(objectives_text, log_file)
    cudnn.benchmark = True

    mfnet_3d = MFNET_3D_MO  # mfnet 3d multi output
    kwargs = dict()
    kwargs["num_coords"] = num_coords
    model_ft = mfnet_3d(num_classes, dropout=args.dropout, **kwargs)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained_model_path)
        # below line is needed if network is trained with DataParallel
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        base_dict = {k: v for k, v in list(base_dict.items()) if 'classifier' not in k}
        model_ft.load_state_dict(base_dict, strict=False)  # model.load_state_dict(checkpoint['state_dict'])
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
        model_ft, optimizer, ckpt_path = resume_checkpoint(model_ft, optimizer, output_dir, model_name,
                                                           args.resume_from)
        print_and_save("Resuming training from: {}".format(ckpt_path), log_file)

    # load dataset and train and validation iterators
    train_sampler = prepare_sampler("train", args.clip_length, args.frame_interval)
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1./0.8], slen=[224, 288]), RandomCrop((224, 224)),
        RandomHorizontalFlip(), RandomHLS(vars=[15, 35, 25]), ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    train_loader = MultitaskDatasetLoader(train_sampler, args.train_lists, args.dataset, tasks_per_dataset,
                                          batch_transform=train_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                          hand_list_prefix=args.hand_list_prefix[:])
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.num_workers, pin_memory=True)

    test_sampler = prepare_sampler("val", args.clip_length, args.frame_interval)
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(),
                                          Normalize(mean=mean_3d, std=std_3d)])
    test_loader = MultitaskDatasetLoader(test_sampler, args.test_lists, args.dataset, tasks_per_dataset,
                                         batch_transform=test_transforms, gaze_list_prefix=args.gaze_list_prefix[:],
                                         hand_list_prefix=args.hand_list_prefix[:])
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)

    ce_loss = torch.nn.CrossEntropyLoss().cuda(device=args.gpus[0])
    lr_scheduler = load_lr_scheduler(args.lr_type, args.lr_steps, optimizer, len(train_iterator))

    train = train_mfnet_mo
    test = test_mfnet_mo
    num_cls_tasks = num_objectives[0]
    new_top1, top1 = [0.0] * num_cls_tasks, [0.0] * num_cls_tasks
    for epoch in range(args.max_epochs):
        train(model_ft, optimizer, ce_loss, train_iterator, num_objectives, tasks_per_dataset, epoch, log_file,
              args.gpus, lr_scheduler)
        if (epoch+1) % args.eval_freq == 0:
            if args.eval_on_train:
                test(model_ft, ce_loss, train_iterator, num_objectives, tasks_per_dataset, epoch, "Train", log_file,
                     args.gpus)
            new_top1 = test(model_ft, ce_loss, test_iterator, num_objectives, tasks_per_dataset, epoch, "Test",
                            log_file, args.gpus)
            top1 = save_mt_checkpoints(model_ft, optimizer, top1, new_top1, args.save_all_weights, output_dir,
                                       model_name, epoch, log_file)


if __name__ == '__main__':
    main()
