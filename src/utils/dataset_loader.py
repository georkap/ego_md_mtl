# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:55:53 2018

Image dataset loader for a .txt file with a sample per line in the format
'path of image start_frame verb_id noun_id'

@author: Γιώργος
"""

import os
import sys
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset as torchDataset
from src.utils.video_sampler import RandomSampling, SequentialSampling, MiddleSampling, DoubleFullSampling, FullSampling
from src.constants import *

def make_class_mapping_generic(samples_list, attribute):
    classes = []
    for sample in samples_list:
        label = getattr(sample, attribute)
        if label not in classes:
            classes.append(label)
    classes = np.sort(classes)
    mapping_dict = {}
    for i, c in enumerate(classes):
        mapping_dict[c] = i
    return mapping_dict


def load_pickle(tracks_path):
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    return tracks


def substitute_prefix(tracks_path, secondary_prefix):
    obj_path = secondary_prefix
    for p in tracks_path.split('\\')[1:]:
        obj_path = os.path.join(obj_path, p)
    return obj_path

def load_two_pickle(tracks_path, secondary_prefix):
    obj_path = substitute_prefix(tracks_path, secondary_prefix)
    return load_pickle(tracks_path), load_pickle(obj_path)


# from PIL import Image
def load_images(data_path, frame_indices, image_tmpl):
    images = []
    # images = np.zeros((len(frame_indices), 640, 480, 3))
    for f_ind in frame_indices:
        im_name = os.path.join(data_path, image_tmpl.format(f_ind))
        # next_image = np.array(Image.open(im_name).convert('RGB'))
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        images.append(next_image)
        # images[i] = next_image
    return images


def prepare_sampler(sampler_type, clip_length, frame_interval):
    if sampler_type == "train":
        train_sampler = RandomSampling(num=clip_length,
                                       interval=frame_interval,
                                       speed=[0.5, 1.5], seed=None)
        out_sampler = train_sampler
    else:
        val_sampler = SequentialSampling(num=clip_length,
                                         interval=frame_interval,
                                         fix_cursor=True,
                                         shuffle=True, seed=None)
        out_sampler = val_sampler
    return out_sampler


class EPICDataLine(object):
    def __init__(self, row):
        self.data = row

    @property
    def data_path(self):
        return self.data[0]

    @property
    def num_frames(self):  # sto palio format ayto einai to start_frame
        return int(self.data[1])

    @property
    def label_verb(self):
        return int(self.data[2])

    @property
    def label_noun(self):
        return int(self.data[3])

    @property
    def uid(self):
        return int(self.data[4] if len(self.data) > 4 else -1)

    @property
    def start_frame(self):
        return int(self.data[5] if len(self.data) > 5 else -1)

    @property
    def label_action(self):
        return int(self.data[6] if len(self.data) > 6 else -1)

class GTEADataLine(object):
    def __init__(self, row):
        self.data = row
        self.data_len = len(row)

    @property
    def data_path(self):
        return self.data[0]

    @property
    def frames_path(self):
        path_parts = os.path.normpath(self.data[0]).split(os.sep)
        session_parts = path_parts[-1].split('-')
        session = session_parts[0] + '-' + session_parts[1] + '-' + session_parts[2]
        return os.path.join(path_parts[-4], path_parts[-3]), os.path.join(path_parts[-4], path_parts[-3], path_parts[-2], session, path_parts[-1])

    @property
    def instance_name(self):
        return os.path.normpath(self.data[0]).split(os.sep)[-1]

    @property
    def label_action(self): # to zero based labels
        return int(self.data[1]) - 1 

    @property
    def label_verb(self):
        return int(self.data[2]) - 1 

    @property
    def label_noun(self):
        return int(self.data[3]) - 1

    @property
    def extra_nouns(self):
        extra_nouns = list()
        if self.data_len > 4:
            for noun in self.data[4:]:
                extra_nouns.append(int(noun) - 1)
        return extra_nouns


def parse_samples_list(list_file, datatype):
    return [datatype(x.strip().split(' ')) for x in open(list_file)]

class DatasetInfo(object):
    def __init__(self, dataset_id, dataset_name, data_line, img_tmpl, norm_val, tasks_for_dataset, cls_tasks, max_num_classes,
                 gaze_list_prefix, hand_list_prefix, video_list):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.data_line = data_line
        self.img_tmpl = img_tmpl
        self.norm_val = norm_val
        self.gaze_list_prefix = gaze_list_prefix
        self.hand_list_prefix = hand_list_prefix
        self.td = tasks_for_dataset
        self.cls_tasks = cls_tasks
        self.max_num_classes = max_num_classes

        self.num_classes = list()
        self.mappings = list()

        for task_short, task_long in zip(cls_tasks, list(max_num_classes.keys())): # works in python 3.7 because dicts are ordered
            if task_short in tasks_for_dataset:
                value = tasks_for_dataset[task_short]
                self.num_classes.append(value)
                if value != max_num_classes[task_long]:
                    self.mappings.append(make_class_mapping_generic(video_list, task_long))
                else:
                    self.mappings.append(None)
            else:
                self.mappings.append(None)


def apply_transform_to_track(track, scale_x, scale_y, tl_x, tl_y, norm_val, is_flipped):
    track *= [scale_x, scale_y]
    track -= [tl_x, tl_y]
    if is_flipped:
        track[:, 0] = norm_val[0] - track[:, 0]  # norm_val[0] = img width after transform
    return track

def make_dsnt_track(track, norm_val):
    track = (track * 2 + 1) / norm_val[:2] - 1
    return track

def vis_with_circle(img, left_point, right_point, winname):
    k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255, 0, 0), 4)
    k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0, 0, 255), 4)
    cv2.imshow(winname, k)

def vis_with_circle_gaze(img, gaze_point, winname):
    k = cv2.circle(img.copy(), (int(gaze_point[0]), int(gaze_point[1])), 10, (0, 255, 0), 4)  # green is gaze
    cv2.imshow(winname, k)


class MultitaskDatasetLoader(torchDataset):

    def __init__(self, sampler, split_files, datasets, tasks_per_dataset, batch_transform,
                 gaze_list_prefix, hand_list_prefix, validation=False, eval_gaze=False, vis_data=False):
        self.sampler = sampler
        assert len(datasets) == len(tasks_per_dataset) # 1-1 association between dataset name, split file and resp tasks
        self.video_list = list()
        self.dataset_infos = dict()
        self.maximum_target_size = 0
        for i, (dataset_name, split_file, td) in enumerate(zip(datasets, split_files, tasks_per_dataset)):
            glp = gaze_list_prefix.pop(0) if 'G' in td else None
            hlp = hand_list_prefix.pop(0) if 'H' in td else None
            if dataset_name == 'epick':
                data_line = EPICDataLine
                img_tmpl = 'frame_{:010d}.jpg'
                norm_val = [456., 256., 456., 256.]
                video_list = parse_samples_list(split_file, data_line)
                cls_tasks = EPIC_CLS_TASKS
                max_num_classes = LABELS_EPIC
            elif dataset_name == 'egtea':
                data_line = GTEADataLine
                img_tmpl = 'img_{:05d}.jpg'
                norm_val = [640., 480., 640., 480.]
                video_list = parse_samples_list(split_file, data_line)
                cls_tasks = GTEA_CLS_TASKS
                max_num_classes = LABELS_GTEA
            else:
                # undeveloped dataset yet e.g. something something or whatever
                pass
            dat_info = DatasetInfo(i, dataset_name, data_line, img_tmpl, norm_val, td, cls_tasks, max_num_classes, glp,
                                   hlp, video_list)
            self.maximum_target_size = td['max_target_size'] if td['max_target_size'] > self.maximum_target_size else self.maximum_target_size
            self.dataset_infos[dataset_name] = dat_info
            self.video_list += video_list

        self.transform = batch_transform
        self.validation = validation
        self.eval_gaze = eval_gaze
        self.vis_data = vis_data

    def __len__(self):
        return len(self.video_list)

    def get_scale_and_shift(self, orig_width, orig_height):
        is_flipped = False
        if 'RandomScale' in self.transform.transforms[0].__repr__():
            # means we are in training so get the transformations
            sc_w, sc_h = self.transform.transforms[0].get_new_size()
            tl_y, tl_x = self.transform.transforms[1].get_tl()
            if 'RandomHorizontalFlip' in self.transform.transforms[2].__repr__():
                is_flipped = self.transform.transforms[2].is_flipped()
        elif 'Resize' in self.transform.transforms[0].__repr__():  # means we are in testing
            sc_h, sc_w, _ = self.transform.transforms[0].get_new_shape()
            tl_y, tl_x = self.transform.transforms[1].get_tl()
        else:
            sc_w = orig_width
            sc_h = orig_height
            tl_x = 0
            tl_y = 0
        return sc_w / orig_width, sc_h / orig_height, tl_x, tl_y, is_flipped

    def __getitem__(self, index):
        data_line = self.video_list[index]
        use_hands = False
        hand_track_path = None
        use_gaze = False
        gaze_track_path = None
        if isinstance(data_line, EPICDataLine): # parse EPIC line
            dataset_name = 'epick'
            path = data_line.data_path
            validation_id = data_line.uid
            frame_count = data_line.num_frames
            label_verb = data_line.label_verb
            label_noun = data_line.label_noun
            start_frame = data_line.start_frame if data_line.start_frame != -1 else 0
            if 'H' in self.dataset_infos[dataset_name].td:
                use_hands = True
                path_d, path_ds, a, b, c, pid, vid_id = path.split("\\")
                hand_track_path = os.path.join(path_d, path_ds, self.dataset_infos[dataset_name].hand_list_prefix,
                                               pid, vid_id, "{}_{}_{}.pkl".format(start_frame, label_verb, label_noun))
        elif isinstance(data_line, GTEADataLine): # parse EGTEA line
            dataset_name = 'egtea'
            base_path, path = self.video_list[index].frames_path
            instance_name = self.video_list[index].instance_name
            validation_id = instance_name
            frame_count = len(os.listdir(path))
            assert frame_count > 0
            start_frame = 0
            if 'H' in self.dataset_infos[dataset_name].td:
                use_hands = True
                hand_track_path = os.path.join(base_path, self.dataset_infos[dataset_name].hand_list_prefix,
                                               instance_name + '.pkl')
            if 'G' in self.dataset_infos[dataset_name].td:
                use_gaze = True
                gaze_track_path = os.path.join(base_path, self.dataset_infos[dataset_name].gaze_list_prefix,
                                               instance_name + '.pkl')
        else:
            # unknown type
            sys.exit("Wrong data_line in dataloader.__getitem__. Exit code -2")

        dataset_id = self.dataset_infos[dataset_name].dataset_id
        norm_val = self.dataset_infos[dataset_name].norm_val
        img_tmpl = self.dataset_infos[dataset_name].img_tmpl
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=start_frame)
        sampled_frames = load_images(path, sampled_idxs, img_tmpl)

        clip_input = np.concatenate(sampled_frames, axis=2)
        or_h, or_w, _ = clip_input.shape

        # hands points is the final output, hand tracks is pickle, left and right track are intermediate versions
        hand_points, hand_tracks, left_track, right_track = None, None, None, None
        if use_hands:
            hand_tracks = load_pickle(hand_track_path)
            left_track = np.array(hand_tracks['left'], dtype=np.float32)
            right_track = np.array(hand_tracks['right'], dtype=np.float32)

            sampled_idxs = (np.array(sampled_idxs) - start_frame).astype(np.int)
            left_track = left_track[sampled_idxs]  # keep the points for the sampled frames
            right_track = right_track[sampled_idxs]
            if self.vis_data:
                left_track_vis = left_track
                right_track_vis = right_track
            left_track = left_track[::2]
            right_track = right_track[::2]

        # gaze points is the final output, gaze data is the pickle data, gaze track is intermediate versions
        gaze_points, gaze_data, gaze_track = None, None, None
        if use_gaze:
            gaze_data = load_pickle(gaze_track_path)
            gaze_track = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)
            if 'DoubleFullSampling' not in self.sampler.__repr__():
                gaze_track = gaze_track[sampled_idxs]
            if self.vis_data:
                gaze_track_vis = gaze_track
            if 'DoubleFullSampling' not in self.sampler.__repr__():
                gaze_track = gaze_track[::2]
            gaze_track *= norm_val[:2]

        if self.transform is not None:
            # transform the frames
            clip_input = self.transform(clip_input)
            # apply transforms to tracks
            if use_hands or use_gaze:
                scale_x, scale_y, tl_x, tl_y, is_flipped = self.get_scale_and_shift(or_w, or_h)
                _, _, max_h, max_w = clip_input.shape
                norm_val = [max_w, max_h, max_w, max_h]
                if use_hands:
                    left_track = apply_transform_to_track(left_track, scale_x, scale_y, tl_x, tl_y, norm_val, is_flipped)
                    right_track = apply_transform_to_track(right_track, scale_x, scale_y, tl_x, tl_y, norm_val, is_flipped)
                if use_gaze:
                    gaze_track = apply_transform_to_track(gaze_track, scale_x, scale_y, tl_x, tl_y, norm_val, is_flipped)
        # regardless of the transfoms the tracks should be normalized to [-1,1] for the dsnt layer
        if use_hands:
            left_track = make_dsnt_track(left_track, norm_val)
            right_track = make_dsnt_track(right_track, norm_val)
        if use_gaze:
            gaze_track = make_dsnt_track(gaze_track, norm_val)

        if self.vis_data:
            # for i in range(len(sampled_frames)):
            #     cv2.imshow('orig_img', sampled_frames[i])
            #     cv2.imshow('transform', clip_input[:, i, :, :].numpy().transpose(1, 2, 0))
            #     cv2.waitKey(0)
            if use_hands:
                orig_left = np.array(hand_tracks['left'], dtype=np.float32)
                orig_left = orig_left[sampled_idxs]
                orig_right = np.array(hand_tracks['right'], dtype=np.float32)
                orig_right = orig_right[sampled_idxs]

                for i in range(len(sampled_frames)):
                    vis_with_circle(sampled_frames[i], orig_left[i], orig_right[i], 'hands no aug')
                    vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), left_track_vis[i],
                                    right_track_vis[i], 'hands transformed')
                    vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_left[i], orig_right[i],
                                    'hands trans. img not coords')
                    cv2.waitKey(0)
            if use_gaze:
                orig_gaze = np.array([[value[0], value[1]] for key, value in gaze_data.items()],
                                     dtype=np.float32)[sampled_idxs]
                for i in range(len(sampled_frames)):
                    vis_with_circle_gaze(sampled_frames[i], orig_gaze[i]*[or_w, or_h], 'gaze no aug')
                    vis_with_circle_gaze(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), gaze_track_vis[i],
                                         'gaze transformed')
                    vis_with_circle_gaze(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_gaze[i]*norm_val[:2],
                                         'gaze trans. img not coords')
                    cv2.waitKey(0)

        # get the classification task labels
        labels = list()
        all_cls_tasks = self.dataset_infos[dataset_name].cls_tasks
        all_cls_tasks_names = list(self.dataset_infos[dataset_name].max_num_classes.keys())
        tasks_for_dataset = self.dataset_infos[dataset_name].td
        mappings = self.dataset_infos[dataset_name].mappings
        for i, cls_task in enumerate(all_cls_tasks):
            if cls_task in tasks_for_dataset:
                label_num = getattr(data_line, all_cls_tasks_names[i])
                label_num = mappings[i][label_num] if mappings[i] is not None else label_num
                labels.append(label_num)

        labels_dtype = np.float32 if use_gaze or use_hands else np.int64
        labels = np.array(labels, dtype=labels_dtype)

        # get the coordinate task labels
        if use_gaze:
            gaze_points = gaze_track.astype(np.float32).flatten()
            labels = np.concatenate((labels, gaze_points))
        if use_hands:
            hand_points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(
                np.float32)
            hand_points = hand_points.flatten()
            labels = np.concatenate((labels, hand_points))

        if len(labels) < self.maximum_target_size:
            labels = np.concatenate((labels, [0.0]*(self.maximum_target_size-len(labels)))).astype(np.float32)
        if self.validation:
            return clip_input, labels, validation_id
        elif self.eval_gaze and use_gaze:
            orig_gaze = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32).flatten()
            return clip_input, labels, dataset_id, orig_gaze, validation_id
        else:
            return clip_input, labels, dataset_id


if __name__ == '__main__':
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\vis_utils\21247.txt"

    # video_list_file = r"D:\Code\hand_track_classification\splits\gtea_rgb\fake_split2.txt"
    # video_list_file = r"splits\gtea_rgb_frames\fake_split3.txt"

    import torchvision.transforms as transforms
    from src.utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, \
        Normalize, Resize, CenterCrop
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]

    from src.utils.argparse_utils import parse_tasks_str

    seed = 0
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), RandomHorizontalFlip(seed=seed), RandomHLS(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(),
                                          Normalize(mean=mean_3d, std=std_3d)])

    # test_sampler = MiddleSampling(num=16)
    # test_sampler = FullSampling()
    test_sampler = RandomSampling(num=16, interval=2, speed=[1.0, 1.0], seed=seed)

    #### tests
    # Note: before running tests put working directory as the "main" files
    # 1 test dataloader for epic kitchens
    # tasks_str = "V125N352" # "A2513N352H" ok # "A2513V125N352H" ok # "V125N351" ok # "V125H" ok # "A2513V125N351H" ok  # "A2513V125N351GH" crashes due to 'G'
    # tpd = parse_tasks_str(tasks_str)
    # video_list_file = r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1.txt"
    # _hlp = 'hand_detection_tracks_lr005'
    # loader = MultitaskDatasetLoader(test_sampler, [video_list_file], ['epick'], tasks_per_dataset=tpd,
    #                                 batch_transform=train_transforms, gaze_list_prefix=[None], hand_list_prefix=[_hlp],
    #                                 validation=True, eval_gaze=False, vis_data=True)
    # 2 test dataloader for egtea
    # tasks_str = "A106N53" ok # "N53GH" ok # "A106V19N53GH" ok
    # tpd = parse_tasks_str(tasks_str)
    # video_list_file = r"other\splits\EGTEA\fake_split3.txt"
    # _hlp = 'hand_detection_tracks_lr005'
    # _glp = 'gaze_tracks'
    # loader = MultitaskDatasetLoader(test_sampler, [video_list_file], ['egtea'], tasks_per_dataset=tpd,
    #                                 batch_transform=train_transforms, gaze_list_prefix=[_glp], hand_list_prefix=[_hlp],
    #                                 validation=True, eval_gaze=False, vis_data=True)

    # 3 test dataloader for epic + gtea
    # tasks_str = "A2513V125N352H+A106V19N53GH" ok
    # tpd = parse_tasks_str(tasks_str)
    # video_list_file = [r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt", r"other\splits\EGTEA\fake_split3.txt"]
    # _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']
    # loader = MultitaskDatasetLoader(test_sampler, video_list_file, ['epick', 'egtea'], tasks_per_dataset=tpd,
    #                                 batch_transform=train_transforms, gaze_list_prefix=_glp, hand_list_prefix=_hlp,
    #                                 validation=True, eval_gaze=False, vis_data=True)
    # 4 test dataloader for gtea + epic
    tasks_str = "A106V19N53GH+A2513V125N352H"
    tpd = parse_tasks_str(tasks_str)
    video_list_file = [r"other\splits\EGTEA\fake_split1.txt", r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt"]
    _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    _glp = ['gaze_tracks']
    loader = MultitaskDatasetLoader(test_sampler, video_list_file, ['egtea', 'epick'], tasks_per_dataset=tpd,
                                    batch_transform=train_transforms, gaze_list_prefix=_glp, hand_list_prefix=_hlp,
                                    validation=True, eval_gaze=False, vis_data=True)

    for ind in range(len(loader)):
        _clip_input, _labels, _dataset_id, _validation_id = loader.__getitem__(ind)
        print("\rItem {}: {}: {}".format(ind, _validation_id, _labels))