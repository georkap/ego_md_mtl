# -*- coding: utf-8 -*-
"""
argparse utils

@author: GEO
"""

import os, re, argparse, sys


def make_base_parser(val):
    parser = argparse.ArgumentParser(description='Hand activity recognition')

    parser.add_argument('dataset', type=str, nargs='*', choices=['epick', 'egtea', 'somv1', 'adl', 'adl18', 'charego1', 'charego3'])
    # Load the necessary paths    
    if not val:
        parser.add_argument('--train_lists', type=str, nargs='*')
        parser.add_argument('--test_lists', type=str, nargs='*')
        parser.add_argument('--base_output_dir', type=str, default=r'outputs/')
        parser.add_argument('--model_name', type=str, default=None, help='if left to None it will be automatically created from the args')
    else:
        parser.add_argument('--ckpt_path', type=str)
        parser.add_argument('--val_lists', type=str, nargs='*')
        parser.add_argument('--annotations_path', type=str, default=None)
        parser.add_argument('--epic_actions_path', type=str, default=None)
    parser.add_argument('--append_to_model_name', type=str, default='')
    
    return parser


def parse_args_dataset(parser, net_type):
    # Dataset parameters
    parser.add_argument('--gaze_list_prefix', type=str, default='', nargs='*')
    parser.add_argument('--hand_list_prefix', type=str, default='', nargs='*')
    parser.add_argument('--object_list_prefix', type=str, default='', nargs='*')
    parser.add_argument('--object_cats', type=str, default='', nargs='*')
    parser.add_argument('--interpolate_coordinates', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    if net_type in ['mfnet']:
        parser.add_argument('--clip_gradient', action='store_true')
        parser.add_argument('--clip_length', type=int, default=16, help="define the length of each input sample.")
        parser.add_argument('--frame_interval', type=int, default=2, help="define the sampling interval between frames.")
        #parser.add_argument('--img_tmpl', type=str)
    return parser


def parse_args_network(parser, net_type):
    # Network configuration
    if net_type in ['mfnet']:
        parser.add_argument('--long', default=False, action='store_true')
        parser.add_argument('--sf', default=False, action='store_true')
        parser.add_argument('--flow', default=False, action='store_true')
        parser.add_argument('--dfb', default=False, action='store_true')
        parser.add_argument('--lstm', default=False, action='store_true')
        parser.add_argument('--attn', default=False, action='store_true')
        parser.add_argument('--mtl', default=False, action='store_true')
        parser.add_argument('--tdn', default=False, action='store_true')
        parser.add_argument('--t_attn', default=False, action='store_true')
        parser.add_argument('--only_flow', default=False, action='store_true')
        parser.add_argument('--one_object_layer', default=False, action='store_true')
        parser.add_argument('--epic_pt_gtea', default=False, action='store_true')
        parser.add_argument('--pretrained', default=False, action='store_true')
        parser.add_argument('--pretrained_model_path', type=str, default=r"data/pretrained_models/MFNet3D_Kinetics-400_72.8.pth")
        parser.add_argument('--map_tasks', default=False, action='store_true',
                            help='works only to map VNH tasks of EGTEA to EPIC, effectively reducing the classification'
                                 ' tasks from 6 to 4 and the hand tasks from 2 to 1,'
                                 ' doesnt work for anything else as it is a very elaborate process to make versatile.')
        parser.add_argument('--tasks', type=str, default='A106',
                            help="e.g. A106V19N53GH for all EGTEA tasks with all their classes. "
                                 "If '+' is in the string it will assume multitask for multiple datasets, the order of"
                                 "which will be determined by the value of args.dataset")
        # parser.add_argument('--use_gaze', default=False, action='store_true')  # only applies to gtea
        # parser.add_argument('--use_hands', default=False, action='store_true')  # applies to epick or gtea
    return parser


def parse_args_training(parser):
    # Training parameters
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--mixup_a', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_mult_base', type=float, default=1.0)
    parser.add_argument('--lr_mult_new', type=float, default=1.0)
    parser.add_argument('--lr_type', type=str, default='step',
                        choices=['step', 'multistep', 'clr', 'groupmultistep', 'reduceplat'])
    parser.add_argument('--lr_steps', nargs='+', type=str, default=[7],
                        help="The value of lr_steps depends on lr_type. If lr_type is:"\
                            +"'step' then lr_steps is a list of size 2 that contains the number of epochs needed to reduce the lr at lr_steps[0] and the gamma to reduce by, at lr_steps[1]."\
                            +"'multistep' then lr_steps is a list of size n+1 for n number of learning rate decreases and the gamma to reduce by at lr_steps[-1]."\
                            +"'clr' then lr_steps is a list of size 6: [base_lr, max_lr, num_epochs_up, num_epochs_down, mode, gamma]. In the clr case, argument 'lr' is ignored."\
                            +"'groupmultistep' then the arguments are used like 'multistep' but internally different learning rate is applied to different parameter groups." \
                            +"'reduceplat' args (list of 3): ['min', 'max'] (str), factor (float), patience (int)"
                        )
    parser.add_argument('--moo', default=False, action='store_true')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)  # decay for mfnet is 0.0001
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--grad_acc_batches', type=int, default=None)
    parser.add_argument('--batch_strategy', type=str, default='mixed', choices=['mixed','interleaved','full'],
                        help="mixed: to compose a batch it uses samples from all datasets. "\
                        +"interleaved: each batch is composed from a distinct dataset, but batches from all datasets are used during training at random"\
                        +"full: uses a dataset in full before switching to the next one")
    return parser


def parse_args_eval(parser):
    # Parameters for evaluation during training
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_on_train', default=False, action='store_true')
    # Parameters for evaluation during testing
    parser.add_argument('--mfnet_eval', type=int, default=1)
    # sampler and window apply to evaluation during training as well
    parser.add_argument('--eval_sampler', type=str, default='random',
                        choices=['middle', 'random', 'sequential', 'full', 'doublefull', 'uniform'])
    parser.add_argument('--eval_crop', type=str, default='random', choices=['center', 'random'])
    parser.add_argument('--eval_ensemble', default=False, action='store_true')
    parser.add_argument('--eval_gaze', default=False, action='store_true')

    parser.add_argument('--eval_map_vl', default=False, action='store_true',
                        help='evaluate mean average precision on the video level as per charades dataset')
    parser.add_argument('--eval_lists_vl', type=str, nargs='*')
    parser.add_argument('--eval_map_vid_splits', type=int, default=25)

    # it is a good design choice during evaluation to use a temporal window for the video
    # that lasts approx. 1 second, i.e. 32 frames for EGTEA and 64 for EPIC KITCHENS
    parser.add_argument('--eval_window', type=int, default=32)
    parser.add_argument('--eval_tasks', type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, default=None)
    parser.add_argument('--eval_branch', type=int, default=None, help='choose from 0-2 depending on the model, for the sum use None')

    return parser


def parse_args_program(parser):
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_all_weights', default=False, action='store_true')
    parser.add_argument('--logging', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_from', type=str, default="", help="specify where to resume from otherwise resume from last checkpoint")
    
    return parser


def parse_args(net_type, val=False):
    parser = make_base_parser(val)
    parser = parse_args_dataset(parser, net_type)
    parser = parse_args_network(parser, net_type)
    parser = parse_args_training(parser)
    parser = parse_args_eval(parser)
    parser = parse_args_program(parser)
    
    args = parser.parse_args()
    if not val:
        if args.model_name is None:
            model_name = make_model_name(args, net_type)
            return args, model_name
        return args, args.model_name
    return args


def make_model_name(args, net_type):
    model_name = ''
    for dat_name in args.dataset:
        model_name += '{}_'.format(dat_name)
    if net_type == 'mfnet':
        model_name += "{}_{}_{}_{}_cl{}".format(net_type, args.batch_size,
                                                str(args.dropout).split('.')[0]+str(args.dropout).split('.')[1],
                                                args.max_epochs, args.clip_length)
        if args.pretrained:
            model_name = model_name + "_pt"

    model_name = model_name + "_{}".format(args.lr_type)
    if args.lr_type == "clr":
        clr_type = "tri" if args.lr_steps[4] == "triangular" else "tri2" if args.lr_steps[4] == "triangular2" else "exp"
        model_name = model_name + "_{}".format(clr_type)
        model_name = model_name + str(args.lr_steps[0]).split('.')[0] + str(args.lr_steps[0]).split('.')[1] + '-' + str(args.lr_steps[1]).split('.')[0] + str(args.lr_steps[1]).split('.')[1]

    model_name += "_{}".format(args.tasks)
    if args.mixup_a != 1.:
        model_name = model_name + "_mixup"
    if args.moo:
        model_name = model_name + "_moo"
    if args.long:
        model_name = model_name + "_long"
    if args.sf:
        model_name = model_name + "_sf"
    if args.flow:
        model_name = model_name + "_flow"
    if args.only_flow:
        model_name = model_name + "_only_flow"
    if args.batch_strategy != "mixed":
        model_name = model_name + "_{}".format(args.batches)

    model_name = model_name + args.append_to_model_name
    
    return model_name


def make_log_file_name(output_dir, args):
    # creates the file name for the evaluation log file or None if no logging is required
    if args.logging:
        log_file = os.path.join(output_dir, "results-accuracy-validation")
        log_file += args.append_to_model_name
        log_file += ".txt"
    else:
        log_file = None
    return log_file

def parse_tasks_str(task_str, dataset_names, interpolate_coordinates):
    """ Parser for task string. '+' will split the string and will parse each part for a dataset. It will return a list
     with dictionaries. The length of the list is equal to the number of datasets in the multidataset training scheme.
     Each list entry is a dictionary where the key is the task starting letter and the value is an Int or None. Int
     means the number of classes for the task (i.e. it is a classification task) and None means a coordinate regression
      task which depending on the letter will mean a specific thing."""
    task_str = task_str.split('+')
    tasks_per_dataset = []
    for dataset_tasks, dataset_name in zip(task_str, dataset_names):
        num_classes = {}
        if not re.match(r'[A-Z]', dataset_tasks):
            sys.exit("Badly written task pattern, read the docs. Exit status -1.")

        # get the task starting letters
        tasks = re.findall(r'[A-Z]', dataset_tasks)
        # get per class numbers (if there are any) but skip the first one because the patter always starts with a letter
        classes = re.split(r'[A-Z]', dataset_tasks)[1:]
        assert len(tasks) == len(classes)
        num_cls_tasks = 0 # classification tasks
        num_g_tasks = 0 # gaze prediction tasks
        num_h_tasks = 0 # hand detection tasks
        num_o_tasks = 0 # object vector prediction tasks
        num_c_tasks = 0 # object category prediction tasks
        max_target_size = 0
        for t, cls in zip(tasks, classes):
            num_classes[t] = int(cls) if cls is not '' else None
            # current classification tasks A, V, N, L ->update as necessary
            if t not in ['G', 'H', 'O', 'C']: # expand with other non classification tasks as necessary
                num_cls_tasks += 1
                max_target_size += 1
            if t == 'G':
                num_g_tasks += 1
                max_target_size += 16 * interpolate_coordinates
            if t == 'H':
                num_h_tasks += 1
                max_target_size += 32 * interpolate_coordinates
            if t == 'O':
                num_o_tasks += 1
                max_target_size += int(cls)
            if t == 'C':
                num_c_tasks += 1
                max_target_size += int(cls)

        num_classes['num_cls_tasks'] = num_cls_tasks
        num_classes['num_g_tasks'] = num_g_tasks
        num_classes['num_h_tasks'] = num_h_tasks
        num_classes['num_o_tasks'] = num_o_tasks
        num_classes['num_c_tasks'] = num_c_tasks
        num_classes['interpolate_coordinates'] = interpolate_coordinates
        num_classes['max_target_size'] = max_target_size
        num_classes['dataset'] = dataset_name
        tasks_per_dataset.append(num_classes)
    return tasks_per_dataset

def parse_tasks_per_dataset(tasks_per_dataset):
    objectives_text = "Objectives: "
    num_coords = 0
    num_objects = []
    num_classes = []
    num_obj_cat = []
    num_cls_objectives = 0
    num_g_objectives = 0
    num_h_objectives = 0
    num_o_objectives = 0
    num_c_objectives = 0
    for i, td in enumerate(tasks_per_dataset):
        # parse the dictionary with the tasks
        objectives_text += "\nDataset {}\n".format(i)
        for key, value in td.items():
            if key == 'A':
                objectives_text += " actions {}, ".format(value)
                num_classes.append(value)
                num_cls_objectives += 1
            elif key == 'V':
                objectives_text += " verbs {}, ".format(value)
                num_classes.append(value)
                num_cls_objectives += 1
            elif key == 'N':
                objectives_text += " nouns {}, ".format(value)
                num_classes.append(value)
                num_cls_objectives += 1
            elif key == 'L':
                objectives_text += " locations {}, ".format(value)
                num_classes.append(value)
                num_cls_objectives += 1
            elif key == 'G':
                objectives_text += " gaze, "
                num_coords += 1
                num_g_objectives += 1
            elif key == 'H':
                objectives_text += " hands, "
                num_coords += 2
                num_h_objectives += 1
            elif key == 'O':
                objectives_text += "objects {}, ".format(value)
                num_o_objectives += 1
                num_objects.append(value)
            elif key == 'C':
                objectives_text += "object categories {}, ".format(value)
                num_c_objectives += 1
                num_obj_cat.append(value)
            else:
                pass
            # and an if clause for every new type of task
    objectives = (num_cls_objectives, num_g_objectives, num_h_objectives, num_o_objectives, num_c_objectives)
    task_sizes = (num_classes, num_coords, num_objects, num_obj_cat)
    return objectives_text, objectives, task_sizes

def compare_tasks_per_dataset(train_td, eval_td):
    eval_dataset = eval_td[0]['dataset']

    starting_cls_id = 0
    starting_g_id = 0
    starting_h_id = 0
    dataset_id = -1
    for i, td in enumerate(train_td):
        if td['dataset'] == eval_dataset:
            dataset_id = i
            break
        else:
            starting_cls_id += td['num_cls_tasks']
            starting_g_id += td['num_g_tasks']
            starting_h_id += td['num_h_tasks']
    if dataset_id == -1:
        raise Exception("Can't find eval dataset in trained model.")

    return starting_cls_id, starting_g_id, starting_h_id
