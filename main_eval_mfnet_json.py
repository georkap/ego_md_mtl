import os
import numpy as np
import json

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models.mfnet_3d_mo import MFNET_3D_MO as MFNET_3D_MO
from src.models.mfnet_3d_mo_comb import MFNET_3D_MO_COMB
from src.utils.argparse_utils import parse_args, make_log_file_name, parse_tasks_str, compare_tasks_per_dataset, parse_tasks_per_dataset
from src.utils.file_utils import print_and_save
from src.utils.dataset.dataset_loader import create_dataset_loader
from src.utils.dataset.dataset_loader_transforms import Resize, ToTensorVid, Normalize, CenterCrop, RandomCrop
from src.utils.video_sampler import prepare_sampler
from src.utils.train_utils import validate_mfnet_mo_json
from src.constants import *

np.set_printoptions(linewidth=np.inf, threshold=np.inf)
torch.set_printoptions(linewidth=1000000, threshold=1000000)

def get_task_type_epic(action_classes, verb_classes, noun_classes):
    """
    This snippet to decided what type of task is given for evaluation. This is really experiment specific and needs to be
    updated if things change. The only use for the task types is to make the evaluation on the classes with more than 100
    samples at training for the epic evaluation.
    If actions are trained explicitly then they are task0
    if verbs are trained with actions they they are task1 else they are task0
    if nouns are trained they are always verbtask+1, so either task2 or task1
    if hands are trained they are always the last task so they do not change the above order.
    :return: a list of task names that follows the size of 'num_valid_classes'
    """
    task_types = []
    if action_classes > 0:
        task_types.append("EpicActions")
    if verb_classes > 0:
        task_types.append("EpicVerbs")
    if noun_classes > 0:
        task_types.append("EpicNouns")
    return task_types


EPIC_CLASSES = [2513, 125, 322]
def main():
    args = parse_args('mfnet', val=True)
    tasks_per_dataset = parse_tasks_str(args.tasks, args.dataset, args.interpolate_coordinates)
    gtea_map = (args.eval_dataset == 'egtea' and args.map_tasks)
    epic_map = (args.eval_dataset == 'epick' and args.map_tasks)
    if args.eval_tasks is not None: # trained multi-dataset eval single-dataset
        eval_tasks_per_dataset = parse_tasks_str(args.eval_tasks, [args.eval_dataset], args.interpolate_coordinates)
        starting_cls_id, starting_g_id, starting_h_id = compare_tasks_per_dataset(tasks_per_dataset,
                                                                                  eval_tasks_per_dataset)
        if gtea_map:
            starting_cls_id -= 2
            starting_h_id -= 1
        elif epic_map:
            starting_g_id += 1
        train_tasks_per_dataset = tasks_per_dataset
        tasks_per_dataset = eval_tasks_per_dataset
        args.dataset = [args.eval_dataset]

    objectives_text, objectives, task_sizes = parse_tasks_per_dataset(tasks_per_dataset)
    num_classes, num_coords, num_objects, num_obj_cat = task_sizes

    output_dir = os.path.dirname(args.ckpt_path)
    log_file = make_log_file_name(output_dir, args)
    print_and_save(args, log_file)
    cudnn.benchmark = True

    if args.map_tasks:
        mfnet_3d = MFNET_3D_MO_COMB
    else:
        mfnet_3d = MFNET_3D_MO

    kwargs = dict()
    kwargs["num_coords"] = num_coords + (2 if args.map_tasks else 0)
    kwargs["num_objects"] = num_objects
    kwargs["num_obj_cat"] = num_obj_cat
    kwargs["one_object_layer"] = args.one_object_layer
    kwargs["interpolate_coordinates"] = args.interpolate_coordinates
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

    crop_type = CenterCrop((224, 224)) if args.eval_crop == 'center' else RandomCrop((224, 224))
    val_sampler = prepare_sampler(args.eval_sampler, args.clip_length, args.frame_interval, speed=[1.0, 1.0],
                                  window=args.eval_window, seed=None)
    val_transforms = transforms.Compose([Resize((256, 256), False), crop_type,
                                         ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    val_transforms_flow = transforms.Compose([
        Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(dim=2), Normalize(mean=mean_1d, std=std_1d)])
    val_loader = create_dataset_loader(val_sampler, args.val_lists, val_transforms, val_transforms_flow, True,
                                       tasks_per_dataset, args)
    val_iter = torch.utils.data.DataLoader(val_loader, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)

    outputs = validate_mfnet_mo_json(model_ft, val_iter, args.val_lists[0].split("\\")[-1],
                                     action_file=args.epic_actions_path)

    eval_mode = 'seen' if 's1' in args.val_lists[0] else 'unseen' if 's2' in args.val_lists[0] else 'unknown'
    json_file = "{}.json".format(os.path.join(output_dir, eval_mode))
    with open(json_file, 'w') as jf:
        json.dump(outputs, jf)


if __name__ == '__main__':
    main()
