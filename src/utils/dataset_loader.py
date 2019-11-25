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

def load_flow_images(data_path, frame_indices, image_tmpl):
    flow = [] # the will go uvuv...
    for f_ind in frame_indices:
        u_name = os.path.join(data_path, 'u', image_tmpl.format(f_ind))
        v_name = os.path.join(data_path, 'v', image_tmpl.format(f_ind))
        _u = cv2.imread(u_name, cv2.IMREAD_GRAYSCALE)
        _v = cv2.imread(v_name, cv2.IMREAD_GRAYSCALE)
        flow.append(np.concatenate((_u[:,:,np.newaxis], _v[:,:,np.newaxis]), axis=2))
    return flow

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
    def __init__(self, dataset_id, dataset_name, data_line, img_tmpl, norm_val, tasks_for_dataset, cls_tasks,
                 max_num_classes, gaze_list_prefix, hand_list_prefix, object_list_prefix, video_list, sub_with_flow):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.data_line = data_line
        self.img_tmpl = img_tmpl
        self.norm_val = norm_val
        self.gaze_list_prefix = gaze_list_prefix
        self.hand_list_prefix = hand_list_prefix
        self.object_list_prefix = object_list_prefix
        self.td = tasks_for_dataset
        self.cls_tasks = cls_tasks
        self.max_num_classes = max_num_classes
        self.sub_with_flow = sub_with_flow

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

def mask_coord_bounds(track):
    mask_lower = track >= [-1, -1]
    mask_lower = mask_lower[:, 0] & mask_lower[:, 1]
    mask_upper = track <= [1, 1]
    mask_upper = mask_upper[:, 0] & mask_upper[:, 1]
    mask = mask_lower & mask_upper
    return mask

def object_list_to_bpv(detections, num_noun_classes):
    bpv = np.zeros(num_noun_classes, dtype=np.float32)
    for i, dets in enumerate(detections):
        for obj in dets:
            bpv[obj] = 1 # set 1 to the objects that are seen in any of the frames, not really important when
    return bpv

class MultitaskDatasetLoader(torchDataset):
    def __init__(self, sampler, split_files, datasets, tasks_per_dataset, batch_transform,
                 gaze_list_prefix, hand_list_prefix, object_list_prefix,
                 validation=False, eval_gaze=False, vis_data=False, use_flow=False, flow_transforms=None):
        self.sampler = sampler
        assert len(datasets) == len(tasks_per_dataset) # 1-1 association between dataset name, split file and resp tasks
        self.video_list = list()
        self.dataset_infos = dict()
        self.maximum_target_size = 0
        for i, (dataset_name, split_file, td) in enumerate(zip(datasets, split_files, tasks_per_dataset)):
            glp = gaze_list_prefix.pop(0) if 'G' in td else None
            hlp = hand_list_prefix.pop(0) if 'H' in td else None
            olp = object_list_prefix.pop(0) if 'O' in td else None
            if dataset_name == 'epick':
                data_line = EPICDataLine
                img_tmpl = 'frame_{:010d}.jpg'
                norm_val = [456., 256., 456., 256.]
                video_list = parse_samples_list(split_file, data_line)
                cls_tasks = EPIC_CLS_TASKS
                max_num_classes = LABELS_EPIC
                sub_with_flow = 'rgb\\'
            elif dataset_name == 'egtea':
                data_line = GTEADataLine
                img_tmpl = 'img_{:05d}.jpg'
                norm_val = [640., 480., 640., 480.]
                video_list = parse_samples_list(split_file, data_line)
                cls_tasks = GTEA_CLS_TASKS
                max_num_classes = LABELS_GTEA
                sub_with_flow = 'clips_frames\\'
            else:
                # undeveloped dataset yet e.g. something something or whatever
                pass
            dat_info = DatasetInfo(i, dataset_name, data_line, img_tmpl, norm_val, td, cls_tasks, max_num_classes, glp,
                                   hlp, olp, video_list, sub_with_flow)
            self.maximum_target_size = td['max_target_size'] if td['max_target_size'] > self.maximum_target_size else self.maximum_target_size
            self.dataset_infos[dataset_name] = dat_info
            self.video_list += video_list

        self.use_flow = use_flow
        self.transform = batch_transform
        self.flow_transforms = flow_transforms
        self.validation = validation
        self.eval_gaze = eval_gaze
        self.vis_data = vis_data

    def __len__(self):
        return len(self.video_list)

    def get_scale_and_shift(self, orig_width, orig_height):
        is_flipped = False
        is_training = False
        if 'RandomScale' in self.transform.transforms[0].__repr__():
            # means we are in training so get the transformations
            is_training = True
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
        return sc_w / orig_width, sc_h / orig_height, tl_x, tl_y, is_flipped, is_training

    def __getitem__(self, index):
        data_line = self.video_list[index]
        use_hands = False
        hand_track_path = None
        use_gaze = False
        gaze_track_path = None
        use_objects = False
        obj_track_path = None
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
            if 'O' in self.dataset_infos[dataset_name].td:
                use_objects = True
                path_d, path_ds, a, b, c, pid, vid_id = path.split("\\")
                obj_track_path = os.path.join(path_d, path_ds, self.dataset_infos[dataset_name].object_list_prefix,
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
        orig_norm_val = self.dataset_infos[dataset_name].norm_val
        img_tmpl = self.dataset_infos[dataset_name].img_tmpl
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=start_frame)
        sampled_frames = load_images(path, sampled_idxs, img_tmpl)
        clip_input = np.concatenate(sampled_frames, axis=2)

        sampled_flow, clip_flow = None, None
        if self.use_flow:
            sub_with_flow = self.dataset_infos[dataset_name].sub_with_flow
            flow_path = path.replace(sub_with_flow, 'flow\\')
            # for each flow channel load half the images of the rgb channel
            sampled_flow = load_flow_images(flow_path, (np.array(sampled_idxs)//2)[::2], img_tmpl)
            clip_flow = np.concatenate(sampled_flow, axis=2)

        sampled_idxs = (np.array(sampled_idxs) - start_frame).astype(np.int)

        or_h, or_w, _ = clip_input.shape

        # hands points is the final output, hand tracks is pickle, left and right track are intermediate versions
        hand_points, hand_tracks, left_track, right_track, left_track_vis, right_track_vis = None, None, None, None, None, None
        if use_hands:
            hand_tracks = load_pickle(hand_track_path)
            left_track = np.array(hand_tracks['left'], dtype=np.float32)[sampled_idxs]
            right_track = np.array(hand_tracks['right'], dtype=np.float32)[sampled_idxs]

        # gaze points is the final output, gaze data is the pickle data, gaze track is intermediate versions
        gaze_points, gaze_data, gaze_track, gaze_mask, gaze_track_vis = None, None, None, None, None
        if use_gaze:
            gaze_data = load_pickle(gaze_track_path)
            gaze_track = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)
            if 'DoubleFullSampling' not in self.sampler.__repr__():
                gaze_track = gaze_track[sampled_idxs]
                gaze_track = gaze_track[::2]
            gaze_mask = gaze_track != [0, 0]
            gaze_mask = gaze_mask[:, 0] & gaze_mask[:, 1]
            gaze_track *= orig_norm_val[:2]

        bpv = None
        if use_objects:
            object_detections = np.array(load_pickle(obj_track_path))[sampled_idxs]
            bpv = object_list_to_bpv(object_detections, 352)

        trns_norm_val = orig_norm_val
        if self.transform is not None:
            # transform the frames
            clip_input = self.transform(clip_input)
            # apply transforms to tracks
            if use_hands or use_gaze or self.use_flow:
                scale_x, scale_y, tl_x, tl_y, is_flipped, is_training = self.get_scale_and_shift(or_w, or_h)
                _, _, max_h, max_w = clip_input.shape
                trns_norm_val = [max_w, max_h, max_w, max_h]
                if use_hands:
                    left_track = apply_transform_to_track(left_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                          is_flipped)
                    right_track = apply_transform_to_track(right_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                           is_flipped)
                if use_gaze:
                    gaze_track = apply_transform_to_track(gaze_track, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                          is_flipped)
                if self.use_flow:
                    if is_training:
                        self.flow_transforms.transforms[2].set_flip(is_flipped)
                    clip_flow = self.flow_transforms(clip_flow)

        norm_val = trns_norm_val if self.transform is not None else orig_norm_val
        if use_hands:
            if self.vis_data:
                left_track_vis = left_track
                right_track_vis = right_track
            left_track = left_track[::2]
            right_track = right_track[::2]
        if use_gaze and self.vis_data:
            gaze_track_vis = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32)
            if 'DoubleFullSampling' not in self.sampler.__repr__():
                gaze_track_vis = gaze_track_vis[sampled_idxs]
            gaze_track_vis *= orig_norm_val[:2]
            gaze_track_vis = apply_transform_to_track(gaze_track_vis, scale_x, scale_y, tl_x, tl_y, trns_norm_val,
                                                      is_flipped)

        # regardless of the transforms the tracks should be normalized to [-1,1] for the dsnt layer
        left_hand_mask, right_hand_mask = None, None
        if use_hands:
            left_track = make_dsnt_track(left_track, norm_val)
            right_track = make_dsnt_track(right_track, norm_val)
            left_hand_mask = mask_coord_bounds(left_track)
            right_hand_mask = mask_coord_bounds(right_track)
        if use_gaze:
            gaze_track = make_dsnt_track(gaze_track, norm_val)
            gaze_mask = gaze_mask & mask_coord_bounds(gaze_track)

        if self.vis_data:
            visualize_item(sampled_frames, clip_input, sampled_idxs, use_hands, hand_tracks, left_track_vis,
                           right_track_vis, use_gaze, gaze_data, gaze_track_vis, or_w, or_h, norm_val,
                           self.use_flow, sampled_flow, clip_flow)

        # get the classification task labels
        labels = list()
        masks = list()
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
            masks = np.concatenate((masks, gaze_mask)).astype(np.bool)
        if use_hands:
            hand_points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(
                np.float32)
            hand_points = hand_points.flatten()
            labels = np.concatenate((labels, hand_points))
            masks = np.concatenate((masks, left_hand_mask, right_hand_mask)).astype(np.bool)
        if use_objects:
            bpv = bpv.flatten()
            labels = np.concatenate((labels, bpv))

        # this is for the dataloader only, to avoid having uneven sizes in the label/mask dimension of the batch
        if len(labels) < self.maximum_target_size:
            labels = np.concatenate((labels, [0.0]*(self.maximum_target_size-len(labels)))).astype(np.float32)
        if len(masks) < self.maximum_target_size:
            masks = np.concatenate((masks, [False]*(self.maximum_target_size-len(masks)))).astype(np.float32)

        to_return = None
        if self.validation:
            to_return = (clip_input, clip_flow, labels, masks, validation_id) if self.use_flow else (clip_input, labels, masks, validation_id)
        elif self.eval_gaze and use_gaze:
            orig_gaze = np.array([[value[0], value[1]] for key, value in gaze_data.items()], dtype=np.float32).flatten()
            to_return = (clip_input, clip_flow, labels, dataset_id, orig_gaze, validation_id) if self.use_flow else (clip_input, labels, dataset_id, orig_gaze, validation_id)
        else:
            to_return = (clip_input, clip_flow, labels, masks, dataset_id) if self.use_flow else (clip_input, labels, masks, dataset_id)

        return to_return

def vis_with_circle(img, left_point, right_point, winname):
    k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255, 0, 0), 4)
    k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0, 0, 255), 4)
    cv2.imshow(winname, k)

def vis_with_circle_gaze(img, gaze_point, winname):
    k = cv2.circle(img.copy(), (int(gaze_point[0]), int(gaze_point[1])), 10, (0, 255, 0), 4)  # green is gaze
    cv2.imshow(winname, k)

def visualize_item(sampled_frames, clip_input, sampled_idxs, use_hands, hand_tracks, left_track_vis, right_track_vis,
                   use_gaze, gaze_data, gaze_track_vis, or_w, or_h, norm_val, use_flow, sampled_flow, clip_flow):
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
        if use_flow:
            for i in range(len(sampled_flow)):
                cv2.imshow('u', sampled_flow[i][:,:,0])
                cv2.imshow('u_t', clip_flow[0, i, :, :].numpy())
                cv2.imshow('v', sampled_flow[i][:,:,1])
                cv2.imshow('v_t', clip_flow[1, i, :, :].numpy())
                cv2.waitKey(0)


if __name__ == '__main__':
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\vis_utils\21247.txt"

    # video_list_file = r"D:\Code\hand_track_classification\splits\gtea_rgb\fake_split2.txt"
    # video_list_file = r"splits\gtea_rgb_frames\fake_split3.txt"

    import torchvision.transforms as transforms
    from src.utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, \
        Normalize, Resize, CenterCrop, PredefinedHorizontalFlip
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]
    mean_1d = [0.5]
    std_1d = [0.226]

    from src.utils.argparse_utils import parse_tasks_str

    seed = 0
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), RandomHorizontalFlip(seed=seed), RandomHLS(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    train_flow_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), PredefinedHorizontalFlip(),
        ToTensorVid(dim=2), Normalize(mean=mean_1d, std=std_1d)])

    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)), ToTensorVid(),
                                          Normalize(mean=mean_3d, std=std_3d)])

    # test_sampler = MiddleSampling(num=16)
    # test_sampler = FullSampling()
    test_sampler = RandomSampling(num=16, interval=2, speed=[1.0, 1.0], seed=seed)

    #### tests
    # Note: before running tests put working directory as the "main" files
    # 1 test dataloader for epic kitchens
    tasks_str = 'N352HO352' # "V125N352" # "A2513N352H" ok # "A2513V125N352H" ok # "V125N351" ok # "V125H" ok # "A2513V125N351H" ok  # "A2513V125N351GH" crashes due to 'G'
    datasets = ['epick']
    tpd = parse_tasks_str(tasks_str, datasets)
    video_list_file = [r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt"]
    _hlp = ['hand_detection_tracks_lr005_new']
    _glp = ['gaze_tracks']
    _olp = ['noun_bpv_oh']

    # 2 test dataloader for egtea
    # tasks_str = "A106N53" ok # "N53GH" ok # "A106V19N53GH" ok
    # tpd = parse_tasks_str(tasks_str)
    # video_list_file = r"other\splits\EGTEA\fake_split3.txt"
    # _hlp = 'hand_detection_tracks_lr005'
    # _glp = 'gaze_tracks'

    # 3 test dataloader for epic + gtea
    # tasks_str = "A2513V125N352H+A106V19N53GH" ok
    # tpd = parse_tasks_str(tasks_str)
    # video_list_file = [r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt", r"other\splits\EGTEA\fake_split3.txt"]
    # _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']

    # 4 test dataloader for gtea + epic
    # tasks_str = "A106V19N53GH+A2513V125N352HO352"
    # tpd = parse_tasks_str(tasks_str, ['egtea', 'epick'])
    # video_list_file = [r"other\splits\EGTEA\fake_split1.txt", r"other\splits\EPIC_KITCHENS\epic_rgb_new_nd_val_act\epic_rgb_new_val_1_fake.txt"]
    # _hlp = ['hand_detection_tracks_lr005', 'hand_detection_tracks_lr005']
    # _glp = ['gaze_tracks']
    # _olp = ['noun_bpv_oh']

    loader = MultitaskDatasetLoader(test_sampler, video_list_file, datasets, tasks_per_dataset=tpd,
                                    batch_transform=train_transforms, gaze_list_prefix=_glp, hand_list_prefix=_hlp,
                                    object_list_prefix=_olp,
                                    validation=True, eval_gaze=False, vis_data=True, use_flow=True,
                                    flow_transforms=train_flow_transforms)

    for ind in range(len(loader)):
        data_point = loader.__getitem__(ind)
        if loader.use_flow:
            _clip_input, _clip_flow, _labels, _dataset_id, _validation_id = data_point
        else:
            _clip_input, _labels, _dataset_id, _validation_id = data_point
        print("\rItem {}: {}: {}".format(ind, _validation_id, _labels))
