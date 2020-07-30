# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:01:53 2019

Make splits on the train set of epic kitchens for hand track activity recognition

Similar to build_file_list_epic_smart_points.py with the difference that the track
files are now based on the uid so I can make the train/test lists from the annotatation file
of epic kitchens

@author: Γιώργος
"""
import os, pandas, argparse


def parse_args():
    parser = argparse.ArgumentParser("How to split your dataset")
    parser.add_argument("data_dir_path", type=str)
    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("--epic_annot_file", type=str,
                        default=r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_train_action_labels_new.csv")
    parser.add_argument("--selected_classes", type=int, nargs="*", default=None)
    parser.add_argument("--split_type", type=str, default="validation", choices=['75', 'val', 'brd'])
    parser.add_argument("--nd", default=False, action='store_true')
    parser.add_argument("--act", default=False, action='store_true')
    parser.add_argument("--action_file", type=str,
                        default=r"D:\Datasets\egocentric\EPIC_KITCHENS\EPIC_action_classes_new.csv")
    parser.add_argument("--add_rgb_frames_path", default=False, action='store_true')

    return parser.parse_args()


args = parse_args()
selected_classes = args.selected_classes
base_dir = args.data_dir_path

bad_uids = [126, 961, 5099, 12599, 21740, 25710, 26811, 28585, 33647, 37431]
unavailable = [9, 11, 18]  # 32
available_pids = ["P{:02d}".format(i) for i in range(1, 32) if i not in unavailable]

if args.split_type == "75":  # val PID = [25, 26, 27, 28, 29, 30, 31]
    split_size = 21
elif args.split_type == "brd":  # val PID = [26, 27, 28, 29, 30, 31]
    split_size = 22
else:  # elif args.split_type == "val":  # val PID = [30, 31]
    split_size = 26

split_1 = {}
for i in range(28):
    split_1[available_pids[i]] = "train" if i < split_size else "val"
split_dicts = [split_1]


splits_dir = os.path.join(r"..\splits\EPIC_KITCHENS", args.dataset_identifier)
if args.nd:
    splits_dir += '_nd'
splits_dir += '_{}'.format(args.split_type)
if args.act:
    splits_dir += '_act'
    all_action_ids = pandas.read_csv(args.action_file)
os.makedirs(splits_dir, exist_ok=True)

train_names = os.path.join(splits_dir, "{}_train_{}.txt".format(args.dataset_identifier, 1))
val_names = os.path.join(splits_dir, "{}_val_{}.txt".format(args.dataset_identifier, 1))

train_files = []
val_files = []
train_files.append(open(train_names, 'a'))
val_files.append(open(val_names, 'a'))

annotations = pandas.read_csv(args.epic_annot_file)
for index, row in annotations.iterrows():
    start_frame = row.start_frame
    stop_frame = row.stop_frame
    num_frames = stop_frame - start_frame
    verb_class = row.verb_class
    if selected_classes:
        if verb_class not in selected_classes:
            continue
    noun_class = row.noun_class
    pid = row.participant_id
    uid = row.uid
    if args.nd and uid in bad_uids:
        continue
    videoid = row.video_id
    if args.add_rgb_frames_path:
        segment_dir = os.path.join(args.data_dir_path, pid, "rgb_frames", videoid)
    else:
        segment_dir = os.path.join(args.data_dir_path, pid, videoid)
    if args.act:
        action_id = all_action_ids[all_action_ids.class_key == '{}_{}'.format(verb_class, noun_class)].action_id.item()
        line = "{} {} {} {} {} {} {}\n".format(segment_dir, num_frames, verb_class, noun_class, uid, start_frame, action_id)
    else:
        line = "{} {} {} {} {} {}\n".format(segment_dir, num_frames, verb_class, noun_class, uid, start_frame)
    split = split_dicts[0][pid]
    if split == 'train':
        train_files[0].write(line)
    else:
        val_files[0].write(line)

train_files[0].close()
val_files[0].close()
